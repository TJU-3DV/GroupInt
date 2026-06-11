'''
 @FileName    : flow_matching_group.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''
import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from .MS_HGNN_batch_cap import MS_HGNN_oridinary,MS_HGNN_hyper
from collections import namedtuple
from utils.geometry import perspective_projection
from utils.rotation_conversions import *
import cv2
import math
from utils.FileLoaders import save_pkl
import os
import numpy as np
# from utils.guidance_losses import *

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])


class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=2048):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim * 2
        self.scale_number = len(args.hyper_scales)
        self.nmp_layers =args.nmp_layers
            
        # self.input_fc = nn.Linear(in_dim, self.model_dim * 4)
        # self.input_fc2 = nn.Linear(self.model_dim * 4, self.model_dim * 2)
        # self.input_fc3 = nn.Linear(self.model_dim * 2, self.model_dim)

        self.project = nn.Sequential(
            nn.LayerNorm(args.hidden_dim * 2),
            nn.Linear(args.hidden_dim * 2, 1024),
        )
    
        self.interaction_hyper = MS_HGNN_hyper(
            embedding_dim=self.model_dim,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=self.nmp_layers,
        )

    def forward(self, inputs, x_input, batch_size, agent_num, mask):
        length = inputs.shape[1]

        inputs = inputs * mask[:,None]
        x_input = x_input * mask[:,None]
        mask = mask.view(batch_size, agent_num, 1)

        ftraj_input = inputs.view(batch_size, agent_num, -1)
        x_input = x_input.view(batch_size, agent_num, -1)
        ftraj_input = torch.cat([ftraj_input, x_input], -1)

        ftraj_inter,_ = self.interaction_hyper(ftraj_input, mask)

        output_feature = ftraj_inter.view(batch_size*agent_num,-1)

        output_feature = self.project(output_feature)

        return output_feature
    
############### original flow matching ###############
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class flow_matching_group(nn.Module):
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[], learn_prior=True, nmp_layers=1)
        self.sampling_num_steps = 100
        self.integration_strength = 0.1
        self.use_guidance = True
        self.optimize_trans = True
        self.optimize_camera = False
        
        # self.pen_loss = Pen_Loss()

        # models
        scale_num = 2 + len(self.args.hyper_scales)

        self.past_encoder = PastEncoder(self.args)

        embed_dim = 2048
        out_dim = 24 * 6
        hidden_dim = 256
        feature_dim = 1024

        time_dim = 256
        sinu_pos_emb = SinusoidalPosEmb(256)
        fourier_dim = 256
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.x1_project1 = nn.Sequential(
            nn.LayerNorm(154),
            nn.Linear(154, feature_dim),
        )
        self.x1_project2 = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
        )

        self.project = nn.Sequential(
            nn.LayerNorm(feature_dim + 3),
            nn.Linear(feature_dim + 3, hidden_dim),
        )
        self.project1 = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 1024),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3 , out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3 , 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, 10),
        )

    def set_device(self, device):
        self.device = device
        self.to(device)

    def trans2cam(self, trans, center, scale, full_img_shape, focal_length):        
        ## trans.shape = torch.Size([32, 3])
        img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        w_2, h_2 = img_w / 2., img_h / 2.
        ## img_info['focal_length'] torch.Size([64])
        cam_z = (2 * focal_length) / (b * trans[:,2] + 1e-9)
        bs = b * cam_z + 1e-9
        cam_x = trans[:,0] - (2 * (cx - w_2) / bs)
        cam_y = trans[:,1] - (2 * (cy - h_2) / bs)
        cam = torch.stack([cam_z, cam_x, cam_y], dim=-1)

        return cam
        
    def check(x, name):
        if torch.isnan(x).any():
            print(f"{name} has NaN")
            exit()
    
    def forward(self, data):

        if self.training:
            batch_size, agent_num, d = data['features'].shape
            device = data['pose'].device
            dtype = data['pose'].dtype

            valid = data['valid'].reshape(-1,)

            features = data['features'].reshape(-1, d)

            center = data['center'].reshape(batch_size*agent_num, -1)
            scale = data['scale'].reshape(batch_size*agent_num,)
            img_h = data['img_h'].reshape(batch_size*agent_num,)
            img_w = data['img_w'].reshape(batch_size*agent_num,)
            focal_length = data['focal_length'].reshape(batch_size*agent_num,)
            camerahmr_focal_length = data['camerahmr_focal_length'].reshape(batch_size*agent_num,)

            valid_mask = valid.bool()

            cx, cy, b = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            # The constants below are used for normalization, and calculated from H36M data.
            # It should be fine if you use the plain Equation (5) in the paper.
            bbox_info[:, :2] /= camerahmr_focal_length.unsqueeze(-1)
            bbox_info[:, 2] /= camerahmr_focal_length
            bbox_info = bbox_info.float()

            features = self.project1(features)
            aff_features = torch.cat([features, bbox_info], 1)

            pose_valid = data['pose']        # shape: [N_valid, 72]
            betas_valid = data['betas']      # shape: [N_valid, 10]
            # trans_valid = data['gt_cam_t']   # shape: [N_valid, 3]
            # 创建全零张量
            x1_pose = torch.zeros(batch_size*agent_num, 144, dtype=dtype, device=device)
            x1_shape = torch.zeros(batch_size*agent_num, 10, dtype=dtype, device=device)
            # x1_trans = torch.zeros(batch_size*agent_num, 3, dtype=dtype, device=device)
            pose6d_valid = matrix_to_rotation_6d(axis_angle_to_matrix(pose_valid.view(-1, 24, 3))).view(-1, 144)
            # 用 valid_mask 把压缩数据填回去
            x1_pose[valid_mask] = pose6d_valid                    
            x1_shape[valid_mask] = betas_valid                    # shape: [batch_size*agent_num, 10]
            # x1_trans[valid_mask] = trans_valid                    # shape: [batch_size*agent_num, 3]
            x_1 = torch.cat([x1_pose, x1_shape], dim=-1)  # shape: [batch_size*agent_num, 157]


            camerahmr_pose = data['camerahmr_pose'].reshape(batch_size*agent_num, 24, 3, 3)
            camerahmr_shape = data['camerahmr_shape'].reshape(batch_size*agent_num, 10)
            camerahmr_trans = data['camerahmr_trans'].reshape(batch_size*agent_num, 3)
            camerahmr_pose = matrix_to_rotation_6d(camerahmr_pose.view(-1, 24, 3, 3)).view(-1, 144)
            x_0 = torch.cat([camerahmr_pose, camerahmr_shape], dim=-1)

            # 采样时间点 计算 x_t（线性插值）
            t = torch.rand(batch_size*agent_num, device=device)
            x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1

            # 计算真实flow场
            u_t = x_1 - x_0  # 恒定flow场

            # x_t = x_0

            x_input = self.x1_project1(x_t)
            x_input = self.x1_project2(x_input)
            t_emb = self.time_mlp(t.reshape(-1))

            x_input = x_input + t_emb

            # all_features = torch.cat([aff_features, x_input], 1)        # shape: [batch_size*agent_num, 157 + 2048 + 3]
            inputs = self.project(aff_features)

            # 使用超图预测flow场
            relation_features = self.past_encoder(inputs, x_input, batch_size, agent_num, valid)

            feature = torch.cat([features, relation_features], dim=1) 

            xc = torch.cat([feature, bbox_info],1)   #may need a fc before self.head
            pred_ut_pose6d = self.head(xc)
            pred_ut_shape = self.shape_head(xc)
            # pred_ut_cam = self.cam_head(xc)

            pred_u_t = torch.cat([pred_ut_pose6d, pred_ut_shape], dim=-1)

            x_recon = x_t + (1 - t).unsqueeze(-1) * pred_u_t

            pred_pose6d = x_recon[..., :144].contiguous()
            pred_shape = x_recon[..., 144:154] ## torch.Size([2, 16, 2, 10])
            pred_trans = camerahmr_trans #x_recon[..., -3:]

            pred_pose6d = pred_pose6d[valid == 1]
            pred_shape = pred_shape[valid == 1]
            pred_trans = pred_trans[valid == 1]

            pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1, 6)).view(-1, 24, 3, 3)
            pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

            img_h = img_h[valid == 1]
            img_w = img_w[valid == 1]
            scale = scale[valid == 1]
            center = center[valid == 1]
            camerahmr_focal_length = camerahmr_focal_length[valid == 1]

            num_valid = len(pred_pose)

            # convert the camera parameters from the crop camera to the full camera
            # full_img_shape = torch.stack((img_h, img_w), dim=-1)
            # pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
            temp_trans = torch.zeros((num_valid, 3), dtype=dtype, device=device)

            pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

            camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
            pred_keypoints_2d = perspective_projection(pred_joints + pred_trans[:,None,:],
                                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(num_valid, -1, -1),
                                                    translation=torch.zeros(3, device=device).unsqueeze(0).expand(num_valid, -1),
                                                    focal_length=camerahmr_focal_length,
                                                    camera_center=camera_center)
            
            pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256

            u_t = u_t[valid == 1]
            pred_u_t = pred_u_t[valid == 1]
            x_recon = x_recon[valid == 1]
            x_1 = x_1[valid == 1]

            pred = {'pred_pose':pred_pose,\
                    'pred_shape':pred_shape,\
                    'pred_cam_t':pred_trans,\
                    'pred_rotmat':pred_rotmat,\
                    'pred_verts':pred_verts,\
                    'pred_joints':pred_joints,\
                    'focal_length':camerahmr_focal_length,\
                    'pred_keypoints_2d':pred_keypoints_2d,\
                    'u_t':u_t,\
                    'pred_u_t':pred_u_t,\
                    'x_recon':x_recon,\
                    'x_1':x_1,\
                    }
            
            return pred
        
        else:
            if self.use_guidance:
                return self.sample_w_condition(
                    data,
                    num_steps=self.sampling_num_steps,
                    integration_strength=self.integration_strength,
                    use_keypoint=True,
                    use_collision=False,
                    use_crowd=True,
                    use_contact=False
                )
            else:
                return self.sample(data, num_steps=self.sampling_num_steps, integration_strength=self.integration_strength)
    
    def sample(self, data, num_steps=10, integration_strength=1e-3):
        """推理采样
        Args:
            data: 输入数据
            num_steps: 积分步数
            integration_strength: 积分强度，控制每步积分的影响程度 (0.0 ~ 1.0)
        """

        batch_size, agent_num, d = data['features'].shape
        device = data['pose'].device
        dtype = data['pose'].dtype

        valid = data['valid'].reshape(-1,)

        features = data['features'].reshape(-1, d)

        center = data['center'].reshape(batch_size*agent_num, -1)
        scale = data['scale'].reshape(batch_size*agent_num,)
        img_h = data['img_h'].reshape(batch_size*agent_num,)
        img_w = data['img_w'].reshape(batch_size*agent_num,)
        focal_length = data['focal_length'].reshape(batch_size*agent_num,)
        camerahmr_focal_length = data['camerahmr_focal_length'].reshape(batch_size*agent_num,)

        valid_mask = valid.bool()   

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] /= camerahmr_focal_length.unsqueeze(-1)
        bbox_info[:, 2] /= camerahmr_focal_length
        bbox_info = bbox_info.float()

        features = self.project1(features)

        aff_features = torch.cat([features, bbox_info], 1)

        camerahmr_pose = data['camerahmr_pose'].reshape(batch_size*agent_num, 24, 3, 3)
        camerahmr_shape = data['camerahmr_shape'].reshape(batch_size*agent_num, 10)
        camerahmr_trans = data['camerahmr_trans'].reshape(batch_size*agent_num, 3)
        camerahmr_pose = matrix_to_rotation_6d(camerahmr_pose.view(-1, 24, 3, 3)).view(-1, 144)

        x_t = torch.cat([camerahmr_pose, camerahmr_shape], dim=-1)
        
        dt = 1. / num_steps
        for step in range(num_steps):
            t = torch.ones(batch_size*agent_num, device=device) * step * dt

            x_input = self.x1_project1(x_t)
            x_input = self.x1_project2(x_input)
            t_emb = self.time_mlp(t.reshape(-1))

            x_input = x_input + t_emb

            # all_features = torch.cat([aff_features, x_input], 1)        # shape: [batch_size*agent_num, 157 + 2048 + 3]
            inputs = self.project(aff_features)

            # 预测当前时刻的flow场
            relation_features = self.past_encoder(inputs, x_input, batch_size, agent_num, valid)

            feature = torch.cat([features, relation_features], dim=1) 

            xc = torch.cat([feature, bbox_info],1)   #may need a fc before self.head
            pred_ut_pose6d = self.head(xc)
            pred_ut_shape = self.shape_head(xc)

            pred_u_t = torch.cat([pred_ut_pose6d, pred_ut_shape], dim=-1)

            # 使用积分强度调节flow场
            pred_u_t = pred_u_t * integration_strength
            
            # 积分更新
            x_t = x_t + pred_u_t * dt

        pred_pose6d = camerahmr_pose
        pred_shape = camerahmr_shape ## torch.Size([2, 16, 2, 10])
        pred_trans = camerahmr_trans #x_t[..., -3:] ## torch.Size([2, 16, 2, 3])


        pred_pose6d = pred_pose6d[valid == 1]
        pred_shape = pred_shape[valid == 1]
        pred_trans = pred_trans[valid == 1]

        pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1, 6)).view(-1, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        img_h = img_h[valid == 1]
        img_w = img_w[valid == 1]
        scale = scale[valid == 1]
        center = center[valid == 1]
        camerahmr_focal_length = camerahmr_focal_length[valid == 1]

        num_valid = len(pred_pose)

        temp_trans = torch.zeros((num_valid, 3), dtype=dtype, device=device)

        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(pred_joints + pred_trans[:,None,:],
                                                rotation=torch.eye(3, device=device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=camerahmr_focal_length,
                                                camera_center=camera_center)
        
        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256


        pred = {'pred_pose':pred_pose,\
                'pred_shape':pred_shape,\
                'pred_cam_t':pred_trans,\
                'pred_rotmat':pred_rotmat,\
                'pred_verts':pred_verts,\
                'pred_joints':pred_joints,\
                'focal_length':camerahmr_focal_length,\
                'pred_keypoints_2d':pred_keypoints_2d,\
                # 'u_t':u_t,\
                # 'pred_u_t':pred_u_t,\
                # 'x_recon':x_recon,\
                # 'x_1':x_1,\
                }
        
        return pred

    def sample_w_condition(self, data, num_steps=100, integration_strength=1e-1, use_keypoint=True, use_crowd=False, use_collision = False, use_contact = False):
        """带条件的Flow Matching采样，单图推理
        Args:
            data: 输入数据
            num_steps: 积分步数
            integration_strength: 积分强度
            use_keypoint: 是否使用关键点约束
            use_crowd: 是否使用人群结构约束
            use_collision: 是否使用碰撞约束
            use_contact: 是否使用接触约束
        """
        batch_size, agent_num, d = data['features'].shape
        device = data['features'].device
        dtype = data['features'].dtype

        valid = data['valid'].reshape(-1,)
        features = data['features'].reshape(-1, d)
        center = data['center'].reshape(batch_size*agent_num, -1)
        scale = data['scale'].reshape(batch_size*agent_num,)
        img_h = data['img_h'].reshape(batch_size*agent_num,)
        img_w = data['img_w'].reshape(batch_size*agent_num,)
        # focal_length = data['focal_length'].reshape(batch_size*agent_num,)
        camerahmr_focal_length = data['camerahmr_focal_length'].reshape(batch_size*agent_num,)
        group_id = data['group_id'].reshape(-1,)

        valid_mask = valid.bool()   

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] /= camerahmr_focal_length.unsqueeze(-1)
        bbox_info[:, 2] /= camerahmr_focal_length
        bbox_info = bbox_info.float()

        features = self.project1(features)
        aff_features = torch.cat([features, bbox_info], 1)

        camerahmr_pose = data['camerahmr_pose'].reshape(batch_size*agent_num, 24, 3, 3)
        camerahmr_shape = data['camerahmr_shape'].reshape(batch_size*agent_num, 10)
        camerahmr_trans = data['camerahmr_trans'].reshape(batch_size*agent_num, 3)
        camerahmr_pose = matrix_to_rotation_6d(camerahmr_pose.view(-1, 24, 3, 3)).view(-1, 144)

        if self.optimize_trans:
            camera_translation = camerahmr_trans.clone()
            camera_translation.requires_grad_(True)
            camera_translation_optimizer = torch.optim.Adam([camera_translation], lr=2e-1)
        else:
            camera_translation = camerahmr_trans
            camera_translation_optimizer = None

        fc = camerahmr_focal_length[0]

        x_t = torch.cat([camerahmr_pose, camerahmr_shape], dim=-1)

        # camerahmr keypoint
        ch_pose6d = x_t[..., :144].contiguous()
        ch_shape = x_t[..., 144:154] ## torch.Size([2, 16, 2, 10])
        ch_trans = camerahmr_trans

        ch_pose6d = ch_pose6d[valid == 1]
        ch_shape = ch_shape[valid == 1]
        ch_trans = ch_trans[valid == 1]

        ch_rotmat = rotation_6d_to_matrix(ch_pose6d.reshape(-1, 6)).view(-1, 24, 3, 3)
        ch_pose = matrix_to_axis_angle(ch_rotmat.view(-1, 3, 3)).view(-1, 72)

        img_h_ch = img_h[valid == 1]
        img_w_ch = img_w[valid == 1]
        scale_ch = scale[valid == 1]
        center_ch = center[valid == 1]
        num_valid = len(ch_pose)
        camerahmr_focal_length_ch = fc.unsqueeze(0).expand(num_valid)
        camera_center_ch = torch.stack([img_w_ch/2, img_h_ch/2], dim=-1)

        ch_temp_trans = torch.zeros((num_valid, 3), dtype=dtype, device=device)
        ch_verts, ch_joints = self.smpl(ch_shape, ch_pose, ch_temp_trans, halpe=True)

        ch_keypoints_2d = perspective_projection(ch_joints + ch_trans[:,None,:],
                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(num_valid, -1, -1),
                                            translation=torch.zeros(3, device=device).unsqueeze(0).expand(num_valid, -1),
                                            focal_length=camerahmr_focal_length_ch,
                                            camera_center=camera_center_ch)

        ch_keypoints_2d = (ch_keypoints_2d - center_ch[:,None,:]) / 256

        imgs = data['imgname']
        img = imgs[0]

        # 按Group组织数据
        unique_groups = torch.unique(group_id)
        num_groups = len(unique_groups)

        max_group_size = 0
        group_indices = []
        for g in unique_groups:
            idx = torch.where(group_id == g)[0]
            max_group_size = max(max_group_size, len(idx))
            group_indices.append(idx)

        group_to_flat = torch.zeros((num_groups, max_group_size), dtype=torch.long, device=device)
        group_valid_mask = torch.zeros((num_groups, max_group_size), dtype=torch.bool, device=device)

        for i, idx in enumerate(group_indices):
            curr_size = len(idx)
            group_to_flat[i, :curr_size] = idx
            group_valid_mask[i, :curr_size] = True # True表示该位置有真实数据填充，不是pad出来的

        dt = 1. / num_steps
        for step in range(num_steps):
            # print(step)
            t = torch.ones(batch_size*agent_num, device=device) * step * dt

            x_input = self.x1_project1(x_t)
            x_input = self.x1_project2(x_input)
            t_emb = self.time_mlp(t.reshape(-1))

            x_input = x_input + t_emb

            inputs = self.project(aff_features)

            # [num_groups, max_group_size, dim]
            grouped_inputs = torch.zeros((num_groups, max_group_size, inputs.shape[-1]), device=device, dtype=inputs.dtype)
            grouped_x_input = torch.zeros((num_groups, max_group_size, x_input.shape[-1]), device=device, dtype=x_input.dtype)
            grouped_valid = torch.zeros((num_groups, max_group_size), device=device, dtype=valid.dtype)

            grouped_inputs[group_valid_mask] = inputs[group_to_flat[group_valid_mask]]
            grouped_x_input[group_valid_mask] = x_input[group_to_flat[group_valid_mask]]
            grouped_valid[group_valid_mask] = valid[group_to_flat[group_valid_mask]]

            grouped_inputs_flat = grouped_inputs.view(-1, inputs.shape[-1])
            grouped_x_input_flat = grouped_x_input.view(-1, x_input.shape[-1])
            grouped_valid_flat = grouped_valid.view(-1)

            grouped_relation_flat = self.past_encoder(
                grouped_inputs_flat, 
                grouped_x_input_flat, 
                num_groups,       
                max_group_size,   
                grouped_valid_flat
            )
            grouped_relation = grouped_relation_flat.view(num_groups, max_group_size, -1)
            # 填补回原shape
            relation_features = torch.zeros((agent_num, grouped_relation.shape[-1]), device=device, dtype=grouped_relation.dtype)
            relation_features[group_to_flat[group_valid_mask]] = grouped_relation[group_valid_mask]

            feature = torch.cat([features, relation_features], dim=1) 

            xc = torch.cat([feature, bbox_info],1)   #may need a fc before self.head
            pred_ut_pose6d = self.head(xc)
            pred_ut_shape = self.shape_head(xc)

            pred_u_t = torch.cat([pred_ut_pose6d, pred_ut_shape], dim=-1)

            # 计算条件损失
            with torch.enable_grad():
                x_t_grad = x_t.detach().requires_grad_(True)

                pred_pose6d = x_t_grad[..., :144].contiguous()
                pred_shape = x_t_grad[..., 144:154]
                pred_trans = camera_translation

                pred_pose6d = pred_pose6d[valid == 1]
                pred_shape = pred_shape[valid == 1]
                pred_trans = pred_trans[valid == 1]

                pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1, 6)).view(-1, 24, 3, 3)
                pred_pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

                img_h_v = img_h[valid == 1]
                img_w_v = img_w[valid == 1]
                scale_v = scale[valid == 1]
                center_v = center[valid == 1]
                num_valid = len(pred_pose)
                camerahmr_focal_length_v = fc.unsqueeze(0).expand(num_valid)
                camera_center_v = torch.stack([img_w_v/2, img_h_v/2], dim=-1)
                λ = 1.0 / num_valid

                temp_trans = torch.zeros((num_valid, 3), dtype=dtype, device=device)
                pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

                loss = torch.zeros(1, device=device)
                loss_kp = torch.zeros(1, device=device)
                loss_crowd = torch.zeros(1, device=device)
                total_loss = torch.zeros(1, device=device)

                # if step >= 90:
                #     use_contact = True

                # 2D关键点投影损失
                if use_keypoint:
                    keypoints_2d = data['keypoints'][..., :2]
                    keypoints_conf = data['keypoints'][..., 2]

                    pred_keypoints_2d = perspective_projection(pred_joints + pred_trans[:,None,:],
                                                        rotation=torch.eye(3, device=device).unsqueeze(0).expand(num_valid, -1, -1),
                                                        translation=torch.zeros(3, device=device).unsqueeze(0).expand(num_valid, -1),
                                                        focal_length=camerahmr_focal_length_v,
                                                        camera_center=camera_center_v)
                    
                    pred_keypoints_2d = (pred_keypoints_2d - center_v[:,None,:]) / 256

                    
                    # loss_kp = torch.sum(
                    #     torch.norm(pred_keypoints_2d[:, :17] - keypoints_2d[:, :17], dim=-1)
                    #     * keypoints_conf[:, :17]
                    # )
                    
                    # 若未预先检测keypoint,也可用初始CameraHMR模型预测的keypoint
                    loss_kp = torch.sum(torch.norm(pred_keypoints_2d[:, :17] - ch_keypoints_2d[:, :17], dim=-1))
                    # print("loss_kp_mean",loss_kp / num_valid)
                    loss = loss + loss_kp  # 权重可调
                    total_loss = total_loss + λ * loss_kp

                # L_crowd损失
                if use_crowd:
                    # print("***")
                    b = pred_joints.shape[0]

                    pred_joints = pred_joints + pred_trans[:,None,:]

                    if b <= 1:
                        dis_std = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
                    else:
                        bottom = (pred_joints[:,15] + pred_joints[:,16]) / 2
                        top = pred_joints[:,17]
            
                        l = (top - bottom) / torch.norm(top - bottom, dim=1)[:,None]
                        norm = torch.mean(l, dim=0)
            
                        root = pred_joints[:,19]
            
                        proj = torch.matmul(root, norm)
            
                        dis_std = proj.std()

                    loss_crowd = dis_std
                    # print("loss_crowd:", loss_crowd)
                    loss = loss + 10 * loss_crowd
                    if step < 90:
                        total_loss = total_loss + 5 * loss_crowd
                    else:
                        total_loss = total_loss + 2 * loss_crowd

                # 接触损失
                if use_contact:
                    num_people = pred_verts.shape[0]
                    N = pred_verts.shape[0]
                    loss_contact = torch.zeros(1, device=device)
                    mat = np.zeros((num_people, num_people))
                    
                    img_name = os.path.basename(img)
                    img_id = os.path.splitext(img_name)[0]
                    # 接触矩阵
                    contact_path = f"demo_data/{img_id}.npy"
                    mat = np.load(contact_path)
                    for i in range(N):
                        for j in range(i + 1, N):
                            if mat[i][j] == 1:
                                verts_i = pred_verts[i] + pred_trans[i]
                                verts_j = pred_verts[j] + pred_trans[j]
        
                                dists = torch.cdist(verts_i.unsqueeze(0), verts_j.unsqueeze(0)).squeeze(0)
                                
                                min_dist = dists.min()
                                # 防止穿模
                                if min_dist > 0.01:
                                    loss_contact = loss_contact + min_dist

                    # print("loss_contact:", loss_contact)
                    total_loss = total_loss + 0.05 * loss_contact

                                        
            # 计算梯度并更新flow场
            if loss > 0:
                grad = torch.autograd.grad(loss, x_t_grad, retain_graph=True)[0]
                pred_u_t = pred_u_t - grad

                if self.optimize_trans:
                    camera_translation_optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)                
                    camera_translation_optimizer.step()

            # 使用积分强度调节flow场
            pred_u_t = pred_u_t * integration_strength
            
            # 积分更新
            x_t = x_t + pred_u_t * dt

        pred_pose6d = x_t[..., :144].contiguous()
        pred_shape = x_t[..., 144:154] ## torch.Size([2, 16, 2, 10])
        pred_trans = camera_translation #x_t[..., -3:] ## torch.Size([2, 16, 2, 3])
        
        pred_pose6d = pred_pose6d[valid == 1]
        pred_shape = pred_shape[valid == 1]
        pred_trans = pred_trans[valid == 1]

        pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1, 6)).view(-1, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        img_h = img_h[valid == 1]
        img_w = img_w[valid == 1]
        scale = scale[valid == 1]
        center = center[valid == 1]
        camerahmr_focal_length = camerahmr_focal_length[valid == 1]

        num_valid = len(pred_pose)

        temp_trans = torch.zeros((num_valid, 3), dtype=dtype, device=device)

        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(pred_joints + pred_trans[:,None,:],
                                                rotation=torch.eye(3, device=device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=camerahmr_focal_length,
                                                camera_center=camera_center)
        
        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256


        pred = {'pred_pose':pred_pose,\
                'pred_shape':pred_shape,\
                'pred_cam_t':pred_trans,\
                'pred_rotmat':pred_rotmat,\
                'pred_verts':pred_verts,\
                'pred_joints':pred_joints,\
                'focal_length':camerahmr_focal_length,\
                'pred_keypoints_2d':pred_keypoints_2d,\
                }
        
        return pred