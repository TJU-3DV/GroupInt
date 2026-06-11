'''
 @FileName    : relation_group_v3.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''
import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from .MS_HGNN_group import MS_HGNN_hyper
from collections import namedtuple
from utils.geometry import perspective_projection, rot6d_to_rotmat
from utils.rotation_conversions import matrix_to_axis_angle
from model.backbones.resnet import ResNet, BasicBlock
import cv2

args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])


class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=512):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)
        self.nmp_layers =args.nmp_layers
            
        # self.input_fc = nn.Linear(in_dim, self.model_dim * 4)
        # self.input_fc2 = nn.Linear(self.model_dim * 4, self.model_dim * 2)
        # self.input_fc3 = nn.Linear(self.model_dim * 2, self.model_dim)

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[0]
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[1]
            )

        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[2]
            )
        
        if len(args.hyper_scales) > 3:
            self.interaction_hyper4 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[3]
            )

    
    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N,3).type_as(x)
        category[0:5,0] = 1
        category[5:10,1] = 1
        category[10,2] = 1
        category = category.repeat(B,1,1)
        x = torch.cat((x,category),dim=-1)
        return x

    def convert_color(self, gray):
        im_color = cv2.applyColorMap(cv2.convertScaleAbs(gray, alpha=1),cv2.COLORMAP_JET)
        return im_color

    def viz_two_affinity(self, collectives, corrs):
        
        collectives = collectives.detach().cpu().numpy()
        corrs = corrs.detach().cpu().numpy()
        for collective, corr in zip(collectives, corrs):
            ratiox = 800/int(collective.shape[0])
            ratioy = 800/int(collective.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy
        
            collective = self.convert_color(collective*255)
            corr = self.convert_color(corr*255)
            # im = cv2.resize(im, dsize=None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.namedWindow('collective',0)
            cv2.resizeWindow('collective',int(collective.shape[1]*ratio),int(collective.shape[0]*ratio))
            cv2.imshow('collective',collective)
            cv2.namedWindow('affinity',0)
            cv2.resizeWindow('affinity',int(corr.shape[1]*ratio),int(corr.shape[0]*ratio))
            cv2.imshow('affinity',corr)
            cv2.waitKey()

    def viz_affinity(self, aff_map):
        viz = []
        aff_maps = aff_map.detach().cpu().numpy()
        for im in aff_maps:
            ratiox = 800/int(im.shape[0])
            ratioy = 800/int(im.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy
        
            im = self.convert_color(im*255)
            # im = cv2.resize(im, dsize=None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.namedWindow('affinity',0)
            cv2.resizeWindow('affinity',int(im.shape[1]*ratio),int(im.shape[0]*ratio))
            cv2.imshow('affinity',im)
            cv2.waitKey()
            viz.append(im)
        return viz

    def forward(self, inputs, batch_size, agent_num, mask):
        length = inputs.shape[1]

        inputs = inputs * mask[:,None]

        # inputs = self.input_fc(inputs) #.view(batch_size*agent_num, self.model_dim)
        # inputs = self.input_fc2(inputs)
        # inputs = self.input_fc3(inputs)
        # tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*agent_num)
        # tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)
  
        # ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, length*self.model_dim))
        # ftraj_input = self.input_fc3(self.add_category(ftraj_input))
        mask = mask.view(batch_size, agent_num)
        valid_mask = mask
        mask = torch.matmul(mask[:,:,None], mask[:,None,:])

        ftraj_input = inputs.view(batch_size, agent_num, -1)

        query_input = F.normalize(ftraj_input,p=2,dim=2)
        feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))

        viz_affinity = False
        if viz_affinity:
            aff_maps = self.viz_affinity(feat_corr)

        # edge_0,H_0 = self.interaction(ftraj_input, mask)

        if len(self.args.hyper_scales) > 0:
            edge_1,H_1 = self.interaction_hyper(ftraj_input,feat_corr, mask, viz=False)
        if len(self.args.hyper_scales) > 1:
            edge_2,H_2 = self.interaction_hyper2(ftraj_input,feat_corr, mask, viz=False)
        if len(self.args.hyper_scales) > 2:
            edge_3,H_3 = self.interaction_hyper3(ftraj_input,feat_corr, mask, viz=False)
        if len(self.args.hyper_scales) > 3:
            edge_4,H_4 = self.interaction_hyper4(ftraj_input,feat_corr, mask, viz=False)
        

        if len(self.args.hyper_scales) == 1:
            final_edge = edge_1
            final_H = H_1
            edge_mask = valid_mask
        elif len(self.args.hyper_scales) == 2:
            final_edge = torch.cat((edge_1,edge_2),dim=1)
            final_H = torch.cat((H_1,H_2),dim=1)
            edge_mask = torch.cat((valid_mask,valid_mask),dim=1)
        elif len(self.args.hyper_scales) == 3:
            final_edge = torch.cat((edge_1,edge_2,edge_3),dim=1)
            final_H = torch.cat((H_1,H_2,H_3),dim=1)
            edge_mask = torch.cat((valid_mask,valid_mask,valid_mask),dim=1)
        elif len(self.args.hyper_scales) == 4:
            final_edge = torch.cat((edge_1,edge_2,edge_3,edge_4),dim=1)
            final_H = torch.cat((H_1,H_2,H_3,H_4),dim=1)
            edge_mask = torch.cat((valid_mask,valid_mask,valid_mask,valid_mask),dim=1)

        return final_edge, final_H, edge_mask

class CrossPersonAttention(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid_mask):
        """
        x: (B, N, D)
        valid_mask: (B, N), 1 = valid, 0 = invalid
        """
        key_padding_mask = ~(valid_mask.bool())

        attn_out, attn_weight = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )

        x = self.norm(x + attn_out)

        return x, attn_weight

        
class relation_group_v3(nn.Module):
    def __init__(self, smpl, num_joints=21):
        super().__init__()
        self.smpl = smpl
        self.args = args(hidden_dim=256, hyper_scales=[2,3,4,5], learn_prior=True, nmp_layers=1)

        # models
        scale_num = 2 + len(self.args.hyper_scales)

        self.backbone = ResNet(
            layers = [2,2,2,2],
            block = BasicBlock
        )

        self.past_encoder = PastEncoder(self.args)

        embed_dim = 512
        hidden_dim = 256
        self.project = nn.Sequential(
            nn.LayerNorm(embed_dim + 3),
            nn.Linear(embed_dim + 3, hidden_dim),
        )
        self.cross_attn = CrossPersonAttention(
            dim=hidden_dim,
            num_heads=1
        )
        self.project1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
        )

    def set_device(self, device):
        self.device = device
        self.to(device)

    def build_edge_label(H, gt_group_id, valid_mask):
        B, E, N = H.shape
        edge_labels = torch.zeros(B, E, device=H.device)

        for b in range(B):

            for e in range(E):

                edge = H[b, e]

                # edge里的人
                edge_people = edge.bool() & valid_mask[b].bool()

                if edge_people.sum() <= 1:
                    continue

                ids = gt_group_id[b][edge_people]

                if torch.all(ids == ids[0]):
                    edge_labels[b, e] = 1

        return edge_labels

    def forward(self, data):
        batch_size, agent_num = data['scale'].shape

        imgs = data['img'].reshape(batch_size * agent_num, 3, 256, 192)
        valid = data['valid'].reshape(-1,)

        valid_idx = valid.nonzero().squeeze(1)
        valid_imgs = imgs[valid_idx]
        valid_feats = self.backbone(valid_imgs)

        features = torch.zeros(batch_size * agent_num, valid_feats.shape[-1], device=imgs.device)
        features[valid_idx] = valid_feats

        center = data['center'].reshape(batch_size*agent_num, -1)
        scale = data['scale'].reshape(batch_size*agent_num,)
        img_h = data['img_h'].reshape(batch_size*agent_num,)
        img_w = data['img_w'].reshape(batch_size*agent_num,)
        focal_length = data['focal_length'].reshape(batch_size*agent_num,)

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

        aff_features = torch.cat([features, bbox_info], 1)
        
        tokens = self.project(aff_features)
        tokens = tokens.view(batch_size, agent_num, -1)
        valid_mask = valid.view(batch_size, agent_num)
        tokens, attn_weights = self.cross_attn(tokens, valid_mask)
        inputs = tokens.view(batch_size*agent_num, -1)

        group_logit, H, edge_mask = self.past_encoder(inputs, batch_size, agent_num, valid)

        group_logit = group_logit.squeeze(-1)

        gt_group_id = data['group_id']

        edge_label = relation_group_v3.build_edge_label(H, gt_group_id, valid_mask)


        pred = {}
        pred['group_logit'] = group_logit
        pred['edge_label'] = edge_label
        pred['H'] = H

        pred['tokens'] = tokens
        pred['valid_mask'] = valid_mask
        pred['edge_mask'] = edge_mask

        # pred['pair_logit'] = pair_logit
        # pred['pair_label'] = pair_label
        # pred['pair_mask'] = pair_mask
        
        return pred

