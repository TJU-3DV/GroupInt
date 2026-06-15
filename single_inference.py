'''
 @FileName    : single_inference.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''

import os
import sys
import pickle
import numpy as np
import cv2
import torch
import joblib
from tqdm import tqdm

sys.path.append('./')

from camerahmr.camerahmr_core import CameraHMR_Predictor
from yolox.yolox import Predictor
from model.relation_group_v3 import relation_group_v3
from model.flow_matching_group import flow_matching_group
from utils.smpl_torch_batch import SMPLModel
from utils.compute_pred_id import solve 
from utils.cliff_module import prepare_cliff  
from process import *
from utils.vis import save_demo_results

img_root = "demo_data"
OUTPUT_DIR = 'output/demo'

yolox_model_dir = "pretrained/yolox_data/bytetrack_x_mot17.pth.tar"
yolox_thres = 0.25

GROUP_CKPT = 'pretrained/Group/relation_group.pkl'
RECON_CKPT = 'pretrained/GroupInt/flow_matching_group.pkl'


# ==================================================
# 单帧推理
# ==================================================
class SingleInference:
    def __init__(self, group_ckpt_path, recon_ckpt_path, device='cuda'):
        self.device = torch.device(device)
        print(f"Initializing inference on device: {self.device}")

        # 1. 初始化 SMPL 模型
        self.smpl = SMPLModel(
            device=self.device,
            model_path='./smpl/smpl/SMPL_NEUTRAL.pkl', 
            data_type=torch.float32
        )

        # 2. 加载分组模型
        print("Loading Group Prediction Model...")
        self.group_model = relation_group_v3(smpl=self.smpl, num_joints=21)
        self.group_model.set_device(self.device)
        self.group_model.eval()
        
        group_state = torch.load(group_ckpt_path, map_location=self.device)
        self.group_model.load_state_dict(group_state.get('model', group_state), strict=False)
        print("Group model loaded.")

        # 3. 加载重建模型
        print("Loading Reconstruction Model (Flow Matching)...")
        self.recon_model = flow_matching_group(smpl=self.smpl, num_joints=21)
        self.recon_model.set_device(self.device)
        self.recon_model.eval()
        
        recon_state = torch.load(recon_ckpt_path, map_location=self.device)
        self.recon_model.load_state_dict(recon_state.get('model', recon_state), strict=False)
        self.use_rfkp = False
        print("Reconstruction model loaded.")

    def prepare_single_frame(self, frame_data, max_people, device):
        """
        批量高效处理单帧数据：利用 prepare_cliff 一次性完成所有 bbox 的裁剪和参数计算
        """
        img_path = frame_data['img_path']
        img_h, img_w = frame_data['img_hw']
        people = frame_data['people']
        num_people = len(people)

        # ==========================================
        # Step 1: 批量提取 Bbox
        # ==========================================
        boxes = np.array([p["bbox"] for p in people], dtype=np.float32)
        
        try:
            img = cv2.imread(img_path)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(f"Failed to read image for cliff: {img_path}")
            return None

        cliff_data = prepare_cliff(img, boxes, intris=None)

        norm_imgs = cliff_data["norm_img"].unsqueeze(0)         
        centers = cliff_data["center"].unsqueeze(0)             
        scales = cliff_data["scale"].unsqueeze(0)               
        img_hs = cliff_data["img_h"].unsqueeze(0)               
        img_ws = cliff_data["img_w"].unsqueeze(0)               
        focal_lengths = cliff_data["focal_length"].unsqueeze(0) 

        # ==========================================
        # Step 2: 提取特征字段
        # ==========================================
        valid = torch.zeros((1, max_people), dtype=torch.float32)
        valid[0, :num_people] = 1.0

        img_features = torch.zeros((1, max_people, 1280), dtype=torch.float32)
        camerahmr_poses = torch.zeros((1, max_people, 24, 3, 3), dtype=torch.float32)
        camerahmr_betas = torch.zeros((1, max_people, 10), dtype=torch.float32)
        camerahmr_trans = torch.zeros((1, max_people, 3), dtype=torch.float32)
        camerahmr_focal_length = torch.ones((1, max_people), dtype=torch.float32)
        rf_kps = torch.zeros((1, max_people, 26, 3), dtype=torch.float32)

        for idx, person in enumerate(people):
            img_features[0, idx] = torch.from_numpy(person["gt_box_camerahmr_features_2"]).float()
            camerahmr_poses[0, idx] = torch.from_numpy(person["camerahmr_poses_2"]).float()
            camerahmr_betas[0, idx] = torch.from_numpy(person["camerahmr_betas_2"]).float()
            camerahmr_trans[0, idx] = torch.from_numpy(person["camerahmr_trans_2"]).float()
            camerahmr_focal_length[0, idx] = torch.from_numpy(person["camerahmr_focal_length_2"]).float()
            
            if self.use_rfkp:
                rf_kp = person["halpe_joints_2d_pred"].copy()
                center_xy = centers[0, idx].numpy()
                rf_kp[:, :2] = (rf_kp[:, :2] - center_xy) / 256.0
                rf_kps[0, idx] = torch.from_numpy(rf_kp).float()

        # ==========================================
        # Step 3: 组装数据
        # ==========================================
        data = {
            'valid': valid.to(device),
            'imgname': [img_path], 
            'features': img_features.to(device),
            'center': centers.to(device),
            'scale': scales.to(device),
            'focal_length': focal_lengths.to(device),
            'camerahmr_pose': camerahmr_poses.to(device),
            'camerahmr_shape': camerahmr_betas.to(device),
            'camerahmr_trans': camerahmr_trans.to(device),
            'camerahmr_focal_length': camerahmr_focal_length.to(device), 
            'keypoints': rf_kps.to(device),
            'group_id': torch.ones((1, max_people), dtype=torch.long, device=device) * -1,
            'img_h': img_hs.to(device),
            'img_w': img_ws.to(device),
            'img': norm_imgs.to(device)
        }
        return data

    @torch.no_grad()
    def infer_single_frame(self, frame_data, output_dir):
        """对单帧数据进行分组与重建推理"""
        num_people = len(frame_data['people'])
        if num_people == 0:
            return

        # Step 1: 数据加载与预处理
        data = self.prepare_single_frame(frame_data, num_people, self.device)
        if data is None:
            return

        # Step 2: 分组推理
        group_pred = self.group_model(data)
        pred_group_id = solve(group_pred) 
        
        # Step 3: 注入 group_id
        data['group_id'] = pred_group_id
        data = to_device(data, self.device)
        data = extract_valid_demo(data)

        # Step 4: 群体重建推理
        recon_pred = self.recon_model(data)

        # Step 5: 整理输出结果并保存
        results = {
            'imgs': data['imgname'],
            'pred_verts': recon_pred['pred_verts'].detach().cpu().numpy().astype(np.float32),
            'pred_trans': recon_pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32),
            'focal_length': recon_pred['focal_length'].detach().cpu().numpy().astype(np.float32)
        }

        save_demo_results(self, results=results, output_dir=output_dir)


def main():
    # 1. 初始化检测器 (YOLOX)
    print("Initializing YOLOX Detector...")
    detector = Predictor(yolox_model_dir, yolox_thres)

    # 2. 初始化 CameraHMR
    print("Initializing CameraHMR...")
    predictor = CameraHMR_Predictor()

    # 3. 初始化群体重建推理器
    inferencer = SingleInference(
        group_ckpt_path=GROUP_CKPT,
        recon_ckpt_path=RECON_CKPT,
        device='cuda'
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_imgs = [
        f for f in sorted(os.listdir(img_root)) 
        if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']
    ]
    
    print(f"\n Start processing {len(valid_imgs)} images...")
    for img_name in tqdm(valid_imgs):

        img_path = os.path.abspath(os.path.join(img_root, img_name))
        img_cv2 = cv2.imread(img_path)

        if img_cv2 is None:
            print(f"Failed to read image: {img_path}")
            continue

        h, w = img_cv2.shape[:2]

        # ==========================================
        # stage1：人体检测与 CameraHMR 特征提取
        # ==========================================
        results, _ = detector.predict(img_cv2, viz=False)
        boxes = results['bbox']

        if len(boxes) == 0:
            print(f"No person detected in {img_name}, skipping.")
            continue

        try:
            (pred_poses, pred_betas, pred_cam, pred_trans, focal_length, features) = \
                predictor.process_image(img_path, boxes)
        except Exception as e:
            print(f"CameraHMR failed on {img_name}: {e}")
            continue

        num_people = len(pred_poses)

        
        people = []
        for person_idx in range(num_people):
            person_data = {
                'camerahmr_poses_2': pred_poses[person_idx].detach().cpu().numpy().astype(np.float32),
                'camerahmr_betas_2': pred_betas[person_idx].detach().cpu().numpy().astype(np.float32),
                'camerahmr_trans_2': pred_trans[person_idx].detach().cpu().numpy().astype(np.float32),
                'gt_box_camerahmr_features_2': features[person_idx].detach().cpu().numpy().astype(np.float32),
                'bbox': np.asarray(boxes[person_idx], dtype=np.float32),
                'camerahmr_focal_length_2': np.array([focal_length[person_idx].item()], dtype=np.float32)
            }
            people.append(person_data)

        frame_data = {
            'img_path': img_path,
            'img_hw': (h, w),
            'people': people
        }

        # ==========================================
        # stage2：直接进行单帧推理并保存结果
        # ==========================================
        inferencer.infer_single_frame(frame_data, OUTPUT_DIR)

    print(f"\n✅ All done! Check your results in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()