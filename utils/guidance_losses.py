import torch.nn as nn
import torch
import numpy as np
from utils.geometry import batch_rodrigues

import time
from utils.mesh_intersection.bvh_search_tree import BVH
import utils.mesh_intersection.loss as collisions_loss
from utils.mesh_intersection.filter_faces import FilterFaces
from utils.FileLoaders import load_pkl



class Pen_Loss(nn.Module):
    def __init__(self, device, smpl):
        super(Pen_Loss, self).__init__()
        self.device = device
        self.weight = 0.1
        self.smpl = smpl

        self.search_tree = BVH(max_collisions=8)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.0001,
                                                         point2plane=False,
                                                         vectorized=True)

        self.part_segm_fn = False #"data/smpl_segmentation.pkl"
        if self.part_segm_fn:
            data = load_pkl(self.part_segm_fn)

            faces_segm = data['segm']
            ign_part_pairs = [
                "9,16", "9,17", "6,16", "6,17", "1,2",
                "33,40", "33,41", "30,40", "30,41", "24,25",
            ]

            faces_segm = torch.tensor(faces_segm, dtype=torch.long,
                                device=self.device).unsqueeze_(0).repeat([2, 1]) # (2, 13766)

            faces_segm = faces_segm + \
                (torch.arange(2, dtype=torch.long).to(self.device) * 24)[:, None]
            faces_segm = faces_segm.reshape(-1) # (2*13766, )

            # Create the module used to filter invalid collision pairs
            self.filter_faces = FilterFaces(faces_segm=faces_segm, ign_part_pairs=ign_part_pairs).to(device=self.device)

    def forward(self, verts, trans):
        loss_dict = {}

        vertices = verts + trans[:,None,:]
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                device=vertices.device).unsqueeze_(0).repeat([vertices.shape[0],
                                                                        1, 1])
        bs, nv = vertices.shape[:2] # nv: 6890
        bs, nf = face_tensor.shape[:2] # nf: 13776
        faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
        faces_idx = faces_idx.reshape(bs // 2, -1, 3)
        triangles = vertices.view([-1, 3])[faces_idx]

        print_timings = False
        if print_timings:
            start = time.time()
        collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
        if print_timings:
            torch.cuda.synchronize()
            print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

        if self.part_segm_fn:
            if print_timings:
                start = time.time()
            collision_idxs = self.filter_faces(collision_idxs)
            if print_timings:
                torch.cuda.synchronize()
                print('Collision filtering: {:5f}ms'.format((time.time() -
                                                            start) * 1000))

        if print_timings:
                start = time.time()
        pen_loss = self.pen_distance(triangles, collision_idxs)
        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

        pen_loss = pen_loss[pen_loss<2000]
        
        if len(pen_loss) > 0:
            pen_loss = torch.sigmoid(pen_loss / 2000.) - 0.5
            loss_dict['pen_loss'] = pen_loss.mean() * self.weight
        else:
            loss_dict['pen_loss'] = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        return loss_dict