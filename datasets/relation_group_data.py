'''
 @FileName    : dataset.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''

import os
import torch
import numpy as np
import cv2
from utils.geometry import estimate_translation_np
from utils.imutils import get_crop, keyp_crop2origin, surface_projection, img_crop2origin
from datasets.base import base
import constants

class Relation_Group_Data(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        super(Relation_Group_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.max_people = 0
        self.dataset_name = name
        self.joint_dataset = ['Panoptic', 'JTA']
        self.data_folder = data_folder

        if self.is_train:
            dataset_annot = os.path.join(self.dataset_dir, 'annot/train.pkl')
            self.eval = False
        else:
            self.eval = True
            dataset_annot = os.path.join(self.dataset_dir,'annot/test.pkl')

        params = self.load_pkl(dataset_annot)
        self.features, self.poses, self.shapes, self.imnames, self.img_size, self.bboxs, self.intris, self.centers, self.scales, self.pose2ds, self.joints, self.group_ids = [], [], [], [], [], [], [], [], [], [], [], []
        self.camerahmr_poses, self.camerahmr_betas, self.camerahmr_cam, self.camerahmr_trans, self.camerahmr_focal_length = [], [], [], [], []
        # find max_people across all groups
        for seq in params:
            if(len(seq) < 1):
                continue
            for frame in seq:
                group_count = {}
                for key in frame.keys():
                    if key in ['img_path', 'h_w']:
                        continue

                    # group_id = frame[key]['group_id']
                    group_id = 1
                    if group_id not in group_count:
                        group_count[group_id] = 0
                    group_count[group_id] += 1
                
                for gid in group_count:
                    self.max_people = max(self.max_people,
                                          group_count[gid])
        
        # organize dataset
        for seq in params:
            if len(seq) < 1:
                continue
            for frame in seq:
                group_dict = {}

                img_path = frame['img_path']
                img_hw = frame['h_w']

                for key in frame.keys():
                    if key in ['img_path', 'h_w']:
                        continue

                    person = frame[key]
                    # gid = person['group_id']
                    gid = 1

                    if gid not in group_dict:
                        group_dict[gid] = {
                            "features": [],
                            "pose2ds": [],
                            "poses": [],
                            "shapes": [],
                            "bboxs": [],
                            "intris": [],
                            "centers": [],
                            "scales": [],
                            "joints": [],
                            "group_id": [],
                            "camerahmr_poses": [],
                            "camerahmr_betas": [],
                            "camerahmr_cam": [],
                            "camerahmr_trans": [],
                            "camerahmr_focal_length": []
                        }
                    group_dict[gid]["pose2ds"].append(np.array(person['halpe_joints_2d_pred'],dtype=self.np_type))
                    group_dict[gid]["poses"].append(np.array(person['pose'],dtype=self.np_type))
                    group_dict[gid]["shapes"].append(np.array(person['betas'],dtype=self.np_type))
                    group_dict[gid]["intris"].append(np.array(person['intri'],dtype=self.np_type).reshape(3, 3))

                    group_dict[gid]["bboxs"].append(np.array(person['bbox'],dtype=self.np_type))
                    group_dict[gid]["features"].append(np.array(person['gt_box_camerahmr_features'],dtype=self.np_type).reshape(-1,))
                    group_dict[gid]["centers"].append(np.array(person['gt_center'],dtype=self.np_type))
                    group_dict[gid]["scales"].append(np.array(person['gt_patch_scale'],dtype=self.np_type))
                    # group_dict[gid]["group_id"].append(int(person['group_id']))

                    group_dict[gid]["camerahmr_poses"].append(
                        np.array(person['camerahmr_poses'], dtype=self.np_type)
                    )
                    group_dict[gid]["camerahmr_betas"].append(
                        np.array(person['camerahmr_betas'], dtype=self.np_type)
                    )
                    group_dict[gid]["camerahmr_cam"].append(
                        np.array(person['camerahmr_cam'], dtype=self.np_type)
                    )
                    group_dict[gid]["camerahmr_trans"].append(
                        np.array(person['camerahmr_trans'], dtype=self.np_type)
                    )
                    group_dict[gid]["camerahmr_focal_length"].append(
                        np.array(person['camerahmr_focal_length'], dtype=self.np_type)
                    )

                    if self.dataset_name in ['Panoptic']:
                        group_dict[gid]["joints"].append(np.array(person['h36m_joints_3d'],dtype=self.np_type))
                    elif self.dataset_name in self.joint_dataset:
                        group_dict[gid]["joints"].append(np.array(person['halpe_joints_3d'],dtype=self.np_type))
                
                for gid in group_dict:

                    group = group_dict[gid]

                    self.img_size.append(img_hw)
                    self.imnames.append(img_path)
                    self.features.append(group["features"])
                    self.pose2ds.append(group["pose2ds"])
                    self.poses.append(group["poses"])
                    self.shapes.append(group["shapes"])
                    self.bboxs.append(group["bboxs"])
                    self.intris.append(group["intris"])
                    self.centers.append(group["centers"])
                    self.scales.append(group["scales"])
                    self.joints.append(group["joints"])
                    self.group_ids.append(group["group_id"])
                    self.camerahmr_poses.append(group["camerahmr_poses"])
                    self.camerahmr_betas.append(group["camerahmr_betas"])
                    self.camerahmr_cam.append(group["camerahmr_cam"])
                    self.camerahmr_trans.append(group["camerahmr_trans"])
                    self.camerahmr_focal_length.append(group["camerahmr_focal_length"])
                    # self.group_ids.append(gid)
                
        del frame
        del params
        # print(self.max_people)
        self.len = len(self.features)


    def vis_input(self, image, pred_keypoints, keypoints, pose, betas, trans, valids, new_shapes, new_xs, new_ys, old_xs, old_ys, focal_length, img_h, img_w):
        # Show image
        image = image.copy()
        self.vis_img('img', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,0,255), -1)
        # self.vis_img('keyp', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(pred_keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,255,0), -1)
        self.vis_img('keyp', image)
        

        # Show SMPL
        pose = pose.reshape(-1, 72)[valids==1]
        betas = betas.reshape(-1, 10)[valids==1]
        trans = trans.reshape(-1, 3)[valids==1]
        extri = np.eye(4)
        intri = np.eye(3)
        intri[0][0] = focal_length
        intri[1][1] = focal_length
        intri[0][2] = img_w / 2
        intri[1][2] = img_h / 2
        verts, joints = self.smpl(betas, pose, trans)
        for vert in verts:
            vert = vert.detach().numpy()
            projs, image = surface_projection(vert, self.smpl.faces, extri, intri, image.copy(), viz=False)
        self.vis_img('smpl', image)

    def estimate_trans_cliff(self, joints, keypoints, center, focal_length, img_h, img_w):
        joints = joints.detach().numpy()
        # keypoints[:,:-1] = keypoints[:,:-1] * constants.IMG_RES + np.array(center)
        
        gt_cam_t = estimate_translation_np(joints, keypoints[:,:2], keypoints[:,2], focal_length=focal_length, center=[img_w/2, img_h/2])
        return gt_cam_t
    
    # Data preprocess
    def create_data(self, index=0):
        
        load_data = {}
        
        # Get augmentation parameters
        # flip, pn, rot, sc, gt_input = self.augm_params()
        flip, pn, rot, sc, gt_input = 0, np.ones(3), 0, 1, 0

        
        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        img_h, img_w = self.img_size[index]

        num_people = len(self.pose2ds[index])

        imgnames = ['empty'] * self.max_people
        valid = np.zeros((self.max_people), dtype=np.float32)
        has_3d = np.zeros(self.max_people, dtype=np.float32)
        has_smpls = np.zeros(self.max_people, dtype=np.float32)
        poses = torch.zeros((self.max_people, 72)).float()
        shapes = torch.zeros((self.max_people, 10)).float()
        vertss = torch.zeros((self.max_people, 6890, 3)).float()
        # group_ids = torch.ones((self.max_people)).float() * -1

        if self.dataset_name in ['Panoptic']:
            gt_joints = torch.zeros((self.max_people, 17, 4)).float()
        else:
            gt_joints = torch.zeros((self.max_people, 26, 4)).float()
        gt_trans = torch.zeros((self.max_people, 3)).float()
        pose2d_gt = torch.zeros((self.max_people, 26, 3)).float()
        img_features = torch.zeros((self.max_people, 1280)).float()
        centers = torch.zeros((self.max_people, 2)).float()
        scales = torch.zeros((self.max_people)).float()

        img_hs = np.zeros((self.max_people), dtype=np.float32)
        img_ws = np.zeros((self.max_people), dtype=np.float32)
        focal_lengthes = np.ones((self.max_people), dtype=np.float32)

        camerahmr_poses = torch.zeros((self.max_people, 24, 3, 3)).float()
        camerahmr_betas = torch.zeros((self.max_people, 10)).float()
        camerahmr_cam = torch.zeros((self.max_people, 3)).float()
        camerahmr_trans = torch.zeros((self.max_people, 3)).float()
        camerahmr_focal_length = torch.ones((self.max_people)).float()

        if self.dataset_name == 'INSTA':
            org_features = self.load_pkl(os.path.join(self.data_folder, 'MultiPerson8_annotations/INSTA', self.features[index]))
        else:
            org_features = self.features[index].copy()

        for idx in range(num_people):
            if idx >= self.max_people:
                break
            valid[idx] = 1.

            # Load image features
            features = org_features[idx]
            center = self.centers[index][idx].copy()
            scale = self.scales[index][idx].copy()

            focal_length = self.intris[index][idx].copy()[0][0]

            keypoints = self.pose2ds[index][idx].copy().astype(np.float32)

            # group_id = self.group_ids[index][idx]

            if self.dataset_name in self.joint_dataset:
                joints = torch.from_numpy(self.joints[index][idx].copy()).float()
                if joints.shape[1] == 3:
                    conf = (torch.abs(torch.sum(joints, dim=1)) > 0).float().reshape(-1,1)
                    joints = torch.cat([joints, conf], dim=1)
                pose = torch.zeros((72,), dtype=self.dtype)
                betas = torch.zeros((10,), dtype=self.dtype)
                trans = torch.zeros((3,), dtype=self.dtype)
                verts = torch.zeros((6890,3), dtype=self.dtype)
                has_smpl = np.zeros(1)
            else:
                pose = self.poses[index][idx].copy().reshape(72,)
                betas = self.shapes[index][idx].copy().reshape(10,)

                pose = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
                betas = torch.from_numpy(betas).float()

                temp_pose = pose.clone().reshape(-1, 72)
                temp_shape = betas.clone().reshape(-1, 10)
                temp_trans = torch.zeros((temp_pose.shape[0], 3), dtype=temp_pose.dtype, device=temp_pose.device)
                verts, joints = self.smpl(temp_shape, temp_pose, temp_trans, halpe=True)
                verts = verts.squeeze(0)
                joints = joints.squeeze(0)

                try:
                    trans = self.estimate_trans_cliff(joints, keypoints, center, focal_length, img_h, img_w)
                except:
                    # print('translation regression error')
                    trans = np.zeros((3,), dtype=np.float32)

                trans = torch.from_numpy(trans).float()
            
                conf = torch.ones((len(joints), 1)).float()
                joints = torch.cat([joints, conf], dim=1)

                has_smpl = np.ones(1)

            keypoints[:,:2] = (keypoints[:,:2] - center) / 256
            keypoints = torch.from_numpy(keypoints).float()
            center = torch.from_numpy(np.array(center)).float()


            has_3d[idx] = 1.
            has_smpls[idx] = has_smpl
            img_features[idx] = torch.from_numpy(features).float()
            vertss[idx] = verts
            gt_joints[idx] = joints
            poses[idx] = pose
            shapes[idx] = betas
            gt_trans[idx] = trans
            imgnames[idx] = imgname
            pose2d_gt[idx] = keypoints

            centers[idx] = center
            scales[idx] = sc*scale
            # group_ids[idx] = group_id

            img_hs[idx] = img_h
            img_ws[idx] = img_w
            focal_lengthes[idx] = focal_length
            

            camerahmr_poses[idx] = torch.from_numpy(
                self.camerahmr_poses[index][idx]
            ).float()

            camerahmr_betas[idx] = torch.from_numpy(
                self.camerahmr_betas[index][idx]
            ).float()

            camerahmr_cam[idx] = torch.from_numpy(
                self.camerahmr_cam[index][idx]
            ).float()

            camerahmr_trans[idx] = torch.from_numpy(
                self.camerahmr_trans[index][idx]
            ).float()

            camerahmr_focal_length[idx] = torch.from_numpy(
                self.camerahmr_focal_length[index][idx]
            ).float()


        load_data['valid'] = valid
        load_data['has_3d'] = has_3d
        load_data['has_smpl'] = has_smpls
        load_data['features'] = img_features
        load_data['verts'] = vertss
        load_data['gt_joints'] = gt_joints
        # load_data['img'] = self.normalize_img(img)
        load_data['pose'] = poses
        load_data['betas'] = shapes
        load_data['gt_cam_t'] = gt_trans
        load_data['imgname'] = imgnames
        load_data['keypoints'] = pose2d_gt

        load_data["center"] = centers
        load_data["scale"] = scales
        load_data["img_h"] = img_hs
        load_data["img_w"] = img_ws
        load_data["focal_length"] = focal_lengthes

        # load_data["group_id"] = group_ids

        load_data['camerahmr_pose'] = camerahmr_poses
        load_data['camerahmr_shape'] = camerahmr_betas
        load_data['camerahmr_cam'] = camerahmr_cam
        load_data['camerahmr_trans'] = camerahmr_trans
        load_data['camerahmr_focal_length'] = camerahmr_focal_length

        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len













