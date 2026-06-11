'''
 @FileName    : Annot_CameraHMR.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''

import os
import sys
import pickle
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append('./')

from camerahmr.camerahmr_core import CameraHMR_Predictor
from yolox.yolox import Predictor

# ==================================================
# 配置
# ==================================================
img_root = "demo_data"
output_pkl = "demo_data/demo.pkl"

yolox_model_dir = "pretrained/yolox_data/bytetrack_x_mot17.pth.tar"
yolox_thres = 0.25

# ==================================================
# 初始化检测器
# ==================================================
detector = Predictor(
    yolox_model_dir,
    yolox_thres
)

# ==================================================
# 初始化 CameraHMR
# ==================================================
predictor = CameraHMR_Predictor()

# ==================================================
# 保存所有样本
# ==================================================
samples = []

imgs = sorted(os.listdir(img_root))

for img_name in tqdm(imgs):

    # 跳过非图像文件
    if img_name.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
        continue

    img_path = os.path.abspath(
        os.path.join(img_root, img_name)
    )

    img_cv2 = cv2.imread(img_path)

    if img_cv2 is None:
        print(f"Failed to read image: {img_path}")
        continue

    h, w = img_cv2.shape[:2]

    # ==================================================
    # 人体检测
    # ==================================================
    results, _ = detector.predict(
        img_cv2,
        viz=False
    )

    boxes = results['bbox']

    if len(boxes) == 0:
        print(f"No person detected in {img_name}, skipping.")
        continue

    # ==================================================
    # CameraHMR推理
    # ==================================================
    try:
        (
            pred_poses,
            pred_betas,
            pred_cam,
            pred_trans,
            focal_length,
            features
        ) = predictor.process_image(
            img_path,
            boxes
        )

    except Exception as e:
        print(f"CameraHMR failed on {img_name}: {e}")
        continues

    num_people = len(pred_poses)

    sample = {}

    # ==================================================
    # 保存每个人的数据
    # ==================================================
    for person_idx in range(num_people):

        key = f"{person_idx:03d}"

        person_data = {}

        # pose
        person_data['camerahmr_poses_2'] = (
            pred_poses[person_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # shape
        person_data['camerahmr_betas_2'] = (
            pred_betas[person_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # translation
        person_data['camerahmr_trans_2'] = (
            pred_trans[person_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # feature
        person_data['gt_box_camerahmr_features_2'] = (
            features[person_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # bbox
        person_data['bbox'] = np.asarray(
            boxes[person_idx],
            dtype=np.float32
        )

        # focal length
        person_data['camerahmr_focal_length_2'] = np.array(
            [focal_length[person_idx].item()],
            dtype=np.float32
        )

        sample[key] = person_data

    # ==================================================
    # 图像信息
    # ==================================================
    sample["h_w"] = (h, w)
    sample["img_path"] = img_path

    samples.append(sample)

# ==================================================
# 保存pkl
# ==================================================
os.makedirs(
    os.path.dirname(output_pkl),
    exist_ok=True
)

with open(output_pkl, "wb") as f:
    pickle.dump(samples, f)

print(f"\n✅ Done.")
print(f"Total images: {len(samples)}")
print(f"Saved to: {output_pkl}")