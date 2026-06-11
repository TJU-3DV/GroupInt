'''
 @FileName    : single_inference.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''
import os
import torch
import numpy as np
import cv2
import pickle
import joblib

from model.relation_group_v3 import relation_group_v3  # 统一使用 v4
from model.flow_matching_group import flow_matching_group
from utils.smpl_torch_batch import SMPLModel
from utils.compute_pred_id import solve 
from utils.cliff_module import prepare_cliff  
from process import *
from utils.vis import save_demo_results

def load_pkl_safe(file_path):
    """兼容不同方式保存的 pkl 文件"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception:
        try:
            return joblib.load(file_path)
        except Exception:
            with open(file_path, 'rb') as f:
                return pickle.load(f)

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
        # 收集所有 bbox，形状为 (num_people, 4) -> [min_x, min_y, max_x, max_y]
        boxes = np.array([p["bbox"] for p in people], dtype=np.float32)
        
        # 读取整图一次 (H, W, 3)
        try:
            img = cv2.imread(img_path)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(img_path)

        # 批量裁剪、归一化并计算中心点、尺度、焦距等
        cliff_data = prepare_cliff(img, boxes, intris=None)

        # 提取结果并增加 batch 维度 (1, num_people, ...)
        norm_imgs = cliff_data["norm_img"].unsqueeze(0)         # (1, N, 3, 256, 192)
        centers = cliff_data["center"].unsqueeze(0)             # (1, N, 2)
        scales = cliff_data["scale"].unsqueeze(0)               # (1, N)
        img_hs = cliff_data["img_h"].unsqueeze(0)               # (1, N)
        img_ws = cliff_data["img_w"].unsqueeze(0)               # (1, N)
        focal_lengths = cliff_data["focal_length"].unsqueeze(0) # (1, N)

        # ==========================================
        # Step 2: 提取 pkl 中的其他特征字段
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
            
            # 处理 2D keypoints，使用 cliff 计算出的 center 进行归一化
            if self.use_rfkp:
                rf_kp = person["halpe_joints_2d_pred"].copy()
                center_xy = centers[0, idx].numpy()
                rf_kp[:, :2] = (rf_kp[:, :2] - center_xy) / 256.0
                rf_kps[0, idx] = torch.from_numpy(rf_kp).float()

        # ==========================================
        # Step 3: 生成 spatial_mask 并组装数据
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
            # 'spatial_mask': spatial_mask.to(device)
        }
        return data

    @torch.no_grad()
    def run_on_pkl(self, pkl_path, OUTPUT_DIR):
        print(f"\n Loading data from: {pkl_path}")
        params = load_pkl_safe(pkl_path)
        
        frames_data = []
        max_people = 0
        for frame in params:
            people_data = [p for k, p in frame.items() if k not in ['img_path', 'h_w']]
            if len(people_data) > max_people:
                max_people = len(people_data)
            frames_data.append({
                'img_path': frame['img_path'],
                'img_hw': frame['h_w'],
                'people': people_data
            })
        
        print(f" Found {len(frames_data)} frames. Max people per frame: {max_people}")

        for frame_idx, frame_data in enumerate(frames_data):
            print(f"\n--- Processing Frame {frame_idx + 1}/{len(frames_data)} ---")
            
            # Step 1: 数据加载
            data = self.prepare_single_frame(frame_data, max_people, self.device)

            # Step 2: 分组推理
            group_pred = self.group_model(data)
            pred_group_id = solve(group_pred) 
            
            # Step 3: 注入 group_id
            data['group_id'] = pred_group_id
            # valid_ids = data['valid'].squeeze(0).cpu().numpy()
            # pred_ids = pred_group_id.squeeze(0).cpu().numpy()
            # print(f" Predicted Group IDs for valid persons: {pred_ids[valid_ids == 1]}")

            data = to_device(data, self.device)
            data = extract_valid_demo(data)

            # Step 4: 群体重建推理
            recon_pred = self.recon_model(data)

            # Step 5: 整理输出结果
            results = {}
            results.update(imgs=data['imgname'])
            results.update(pred_verts=recon_pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_trans=recon_pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=recon_pred['focal_length'].detach().cpu().numpy().astype(np.float32))

            save_demo_results(
                self,
                results=results,
                output_dir=OUTPUT_DIR
            )
            

if __name__ == '__main__':
    PKL_PATH = 'demo_data/demo.pkl'
    GROUP_CKPT = 'pretrained/Group/relation_group.pkl'
    RECON_CKPT = 'pretrained/GroupInt/flow_matching_group.pkl'
    OUTPUT_DIR = 'output/demo'
    
    inferencer = SingleInference(
        group_ckpt_path=GROUP_CKPT,
        recon_ckpt_path=RECON_CKPT,
        device='cuda'
    )

    inferencer.run_on_pkl(PKL_PATH, OUTPUT_DIR)
    
    print(f"\n All done! Check your results in: {OUTPUT_DIR}")