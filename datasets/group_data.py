'''
 @FileName    : dataset.py
 @EditTime    : 2022-09-27 16:03:55
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import torch
import numpy as np
import cv2
from datasets.base import base
import constants
from utils.imutils import img_process

class Group_Data(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        super().__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.dataset_name = name
        self.data_folder = data_folder

        if self.is_train:
            self.eval = False
            dataset_annot = os.path.join(self.dataset_dir, 'annot/train.pkl')
        else:
            self.eval = True
            dataset_annot = os.path.join(self.dataset_dir, 'annot/test.pkl')

        # 只加载一次
        self.params = self.load_pkl(dataset_annot)

        # 建立 index 映射（关键！！）
        self.index_map = []
        self.max_people = 0
        
        MAX_ALLOWED_PEOPLE = 800
        for seq_id, seq in enumerate(self.params):
            if len(seq) < 1:
                continue

            for frame_id, frame in enumerate(seq):
                person_keys = [k for k in frame.keys() if k not in ['img_path', 'h_w']]
                num_people = len(person_keys)
                
                if num_people == 0:
                    continue

                if num_people > MAX_ALLOWED_PEOPLE:
                    continue

                self.index_map.append((seq_id, frame_id))
                self.max_people = max(self.max_people, len(person_keys))

        self.len = len(self.index_map)

        print(f"Dataset loaded: {self.len} frames, max_people={self.max_people}")
    
    # Data preprocess
    def create_data(self, index):
        load_data = {}
        
        seq_id, frame_id = self.index_map[index]
        frame = self.params[seq_id][frame_id]
    
        img_h, img_w = frame['h_w']
    
        person_keys = [k for k in frame.keys() if k not in ['img_path', 'h_w']]
        num_people = len(person_keys)

        imgnames = ['empty'] * self.max_people
        valid = np.zeros((self.max_people), dtype=np.float32)
        imgs = torch.zeros((self.max_people, 3, 256, 192)).float()
        centers = torch.zeros((self.max_people, 2)).float()
        scales = torch.zeros((self.max_people)).float()
    
        img_hs = np.zeros((self.max_people), dtype=np.float32)
        img_ws = np.zeros((self.max_people), dtype=np.float32)
        focal_lengthes = np.ones((self.max_people), dtype=np.float32)
    
        group_id = torch.ones((self.max_people)).float() * -1

        spatial_mask = np.zeros((self.max_people, self.max_people), dtype=np.float32)
    
        for idx, key in enumerate(person_keys):
            if idx >= self.max_people:
                break
    
            person = frame[key]
    
            valid[idx] = 1.
    
            bbox = np.array(person['bbox']).reshape(-1,)
            intri = person['intri']
            crop_img = person['crop_img']   # 直接用
            # candidates = person.get('candidate_2', [])
    
            # ===== bbox -> center/scale =====
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
    
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox_size = max(bbox_w * 256 / float(192), bbox_h)
            scale = bbox_size / 200.0 * 1.1
    
            # ===== 图像处理 =====
            img = crop_img.astype(np.float32) / 255.0
    
            mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
            std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
            img = (img - mean) / std
    
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float()
    
            # ===== 赋值 =====
            imgs[idx] = img
            centers[idx] = torch.tensor(center).float()
            scales[idx] = scale
            group_id[idx] = person['group_id']
    
            img_hs[idx] = img_h
            img_ws[idx] = img_w
            focal_lengthes[idx] = intri[0][0]

            # for cand in candidates:
            #     j = int(cand)
            #     if j < self.max_people:   # 防止越界
            #         spatial_mask[idx, j] = 1
    
            # 自连接（一定要有）
            # spatial_mask[idx, idx] = 1

    
        load_data['valid'] = valid
        load_data['img'] = imgs
        load_data['imgname'] = imgnames
        load_data["center"] = centers
        load_data["scale"] = scales
        load_data['group_id'] = group_id
        load_data["img_h"] = img_hs
        load_data["img_w"] = img_ws
        load_data["focal_length"] = focal_lengthes
        # load_data['spatial_mask'] = torch.from_numpy(spatial_mask).float()

        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len













