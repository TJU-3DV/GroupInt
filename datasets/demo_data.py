'''
 @FileName    : dataset_inference.py
 @Description : Dataset for model inference/demo
'''

import os
import torch
import numpy as np
from datasets.base import base

class DemoData(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        # 推理模式下，train 固定为 False
        super(DemoData, self).__init__(train=False, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)
        
        self.eval = True
        dataset = os.path.join(self.dataset_dir, 'demo.pkl')
        params = self.load_pkl(dataset)
        
        self.frames_data = []
        self.max_people = 0
        
        for seq in params:
            if len(seq) < 1:
                continue
            for frame in seq:
                img_path = frame['img_path']
                img_hw = frame['h_w']
                
                people_data = []
                for key, person in frame.items():
                    if key in ['img_path', 'h_w']:
                        continue

                    p_data = {
                        "features": np.array(person['gt_box_camerahmr_features_2'], dtype=self.np_type).reshape(-1,),
                        "bboxs": np.array(person['bbox'], dtype=self.np_type),
                        "camerahmr_poses": np.array(person['camerahmr_poses_2'], dtype=self.np_type),
                        "camerahmr_betas": np.array(person['camerahmr_betas_2'], dtype=self.np_type),
                        "camerahmr_trans": np.array(person['camerahmr_trans_2'], dtype=self.np_type),
                        "camerahmr_focal_length": np.array(person['camerahmr_focal_length_2'], dtype=self.np_type),
                        "rf_kp": np.array(person['halpe_joints_2d_pred'], dtype=self.np_type),
                        "group_id": int(person['group_id'])
                    }
                    people_data.append(p_data)
                
                # 更新单帧最大人数
                if len(people_data) > self.max_people:
                    self.max_people = len(people_data)
                    
                # 将当前帧的所有数据打包
                self.frames_data.append({
                    'img_path': img_path,
                    'img_hw': img_hw,
                    'people': people_data
                })
                
        del params
            
        self.len = len(self.frames_data)


    # Data preprocess for inference
    def create_data(self, index=0):
        load_data = {}
        frame_data = self.frames_data[index]

        batch_size = 1
        
        imgname = frame_data['img_path']
        if self.dataset_dir and not os.path.isabs(imgname):
            imgname = os.path.join(self.dataset_dir, imgname)
            
        img_h, img_w = frame_data['img_hw']
        people = frame_data['people']
        num_people = len(people)

        imgnames = [imgname]
        valid = torch.zeros((batch_size, self.max_people)).float()
        
        # 动态获取特征和 bbox 维度，防止硬编码导致维度不匹配
        feat_dim = 1280
        if num_people > 0:
            feat_dim = people[0]["features"].shape[0]
            bbox_dim = people[0]["bboxs"].shape[0]

        img_features = torch.zeros((batch_size, self.max_people, feat_dim)).float()
        centers = torch.zeros((batch_size, self.max_people, 2)).float()
        scales = torch.zeros((batch_size, self.max_people)).float()
        group_ids = torch.ones((batch_size, self.max_people)).float() * -1
        rf_kps = torch.zeros((batch_size, self.max_people, 26, 3)).float()

        camerahmr_poses = torch.zeros((batch_size, self.max_people, 24, 3, 3)).float()
        camerahmr_betas = torch.zeros((batch_size, self.max_people, 10)).float()
        camerahmr_trans = torch.zeros((batch_size, self.max_people, 3)).float()
        camerahmr_focal_length = torch.ones((batch_size, self.max_people)).float()

        img_hs = torch.zeros((batch_size, self.max_people)).float()
        img_ws = torch.zeros((batch_size, self.max_people)).float()

        for idx in range(num_people):
            valid[0][idx] = 1.
            # imgnames[0][idx] = imgname
            
            person = people[idx]
            
            # 加载数据
            img_features[0][idx] = torch.from_numpy(person["features"]).float()
            bbox = person["bboxs"]
            c_x = (bbox[0] + bbox[2]) / 2.0
            c_y = (bbox[1] + bbox[3]) / 2.0
            center = [c_x, c_y]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox_size = max(bbox_w * 256 / float(192), bbox_h)
            scale = 1.1 * bbox_size / 200.0
            scales[0][idx] = scale
            
            camerahmr_poses[0][idx] = torch.from_numpy(person["camerahmr_poses"]).float()
            camerahmr_betas[0][idx] = torch.from_numpy(person["camerahmr_betas"]).float()
            camerahmr_trans[0][idx] = torch.from_numpy(person["camerahmr_trans"]).float()
            camerahmr_focal_length[0][idx] = torch.from_numpy(person["camerahmr_focal_length"]).float()
            rf_kp = person["rf_kp"]
            rf_kp[:,:2] = (rf_kp[:,:2] - center) / 256
            rf_kps[0][idx] = torch.from_numpy(rf_kp).float()
            centers[0][idx] = torch.from_numpy(np.array(center)).float()

            group_ids[0][idx] = person["group_id"]

            img_hs[0][idx] = img_h
            img_ws[0][idx] = img_w

        # 返回推理所需的字典
        load_data['valid'] = valid
        load_data['imgname'] = imgnames
        load_data['features'] = img_features
        load_data['center'] = centers
        load_data['scale'] = scales
        
        load_data['camerahmr_pose'] = camerahmr_poses
        load_data['camerahmr_shape'] = camerahmr_betas
        load_data['camerahmr_trans'] = camerahmr_trans
        load_data['camerahmr_focal_length'] = camerahmr_focal_length

        load_data["keypoints"] = rf_kps

        load_data["group_id"] = group_ids

        load_data['img_h'] = img_hs
        load_data['img_w'] = img_ws

        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len