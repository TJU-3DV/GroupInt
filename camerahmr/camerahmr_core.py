import cv2
import os
import json
import torch
import smplx
import trimesh
import numpy as np
from torchvision.transforms import Normalize
from utils.rotation_conversions import matrix_to_axis_angle
from camerahmr.camerahmr_model import CameraHMR
from camerahmr.utils.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH
from camerahmr.utils import recursive_to
from camerahmr.cam_model.fl_net import FLNet
from camerahmr.utils.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS
from camerahmr.utils_dataset import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 img_cv2: np.array,
                 bbox_center: np.array,
                 bbox_scale: np.array,
                 cam_int: np.array = None,
                 train: bool = False,
                 img_path = None,
                 **kwargs):
        super().__init__()
        self.img_cv2 = img_cv2
        self.img_path = img_path
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = IMAGE_SIZE
        if cam_int is not None:
            self.cam_int = cam_int
        else:
            self.cam_int = np.array([]) # DenseKP model doesn't need cam_int
        self.mean = 255. * np.array(IMAGE_MEAN)
        self.std = 255. * np.array(IMAGE_STD)
        self.normalize_img = Normalize(mean=IMAGE_MEAN,
                                    std=IMAGE_STD)
        self.center = bbox_center
        self.scale = bbox_scale
        self.personid = np.arange(len(self.center), dtype=np.int32)


    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int):

        center = self.center[idx]
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = None
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size
        cvimg = self.img_cv2

        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)


        img_patch_np = convert_cvimg_to_tensor(img_patch_cv[:, :, ::-1])
        img_patch = torch.as_tensor(img_patch_np, dtype=torch.float32)

        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c] = (img_patch[n_c] - self.mean[n_c]) / self.std[n_c]


        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['imgname'] = str(self.img_path)
        item['box_center'] = self.center[idx]
        item['box_size'] =  bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[0], cvimg.shape[1]])
        item['cam_int'] = self.cam_int
        return item


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img


class CameraHMR_Predictor(object):
    def __init__(self, 
                 smpl_model_path=SMPL_MODEL_PATH, 
                 detector=None):
        """
        detector: 外部传入检测器，需实现 detect(img) 方法，返回 boxes, scores
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.detector = detector

        self.model = self.init_model()
        self.cam_model = self.init_cam_model()
        self.smpl_model = smplx.SMPLLayer(model_path=smpl_model_path, num_betas=NUM_BETAS).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT)['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)
        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        img_h, img_w = batch['img_size'][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        return cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        _, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = np.transpose(img_full_resized.astype('float32'), (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())
        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        cam_int = np.array([[fl_h, 0, img_w / 2],
                            [0, fl_h, img_h / 2],
                            [0, 0, 1]]).astype(np.float32)
        return cam_int

    def process_image(self, img_path, boxes=None, scores=None) :
        """
        img_path: 图片路径
        boxes, scores: 可选，检测器输出，如果没传，会用 self.detector.detect()
        """
        img_cv2 = cv2.imread(str(img_path))
        
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        cam_int = self.get_cam_intrinsics(img_cv2)

        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        items = [dataset[i] for i in range(len(dataset))]

        imgs = [torch.from_numpy(item['img']) if isinstance(item['img'], np.ndarray) else item['img'] for item in items]
        batch_img = torch.stack(imgs, dim=0)

        batch = {
            'img': batch_img,
            'personid': torch.tensor([item['personid'] for item in items], dtype=torch.int64),
            'imgname': items[0]['imgname'],
            'box_center': torch.from_numpy(np.array([item['box_center'] for item in items])).float(),
            'box_size': torch.from_numpy(np.array([item['box_size'] for item in items])).float(),
            'img_size': torch.from_numpy(np.array([item['img_size'] for item in items])).float(),
            'cam_int': torch.from_numpy(np.array([item['cam_int'] for item in items])).float(),
        }
        batch = recursive_to(batch, self.device)

        img_h, img_w = batch['img_size'][0].tolist()

        with torch.no_grad():
            out_smpl_params, out_cam, focal_length_, features = self.model(batch)
            output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)
            
            pred_poses = torch.cat([out_smpl_params['global_orient'], out_smpl_params['body_pose']], dim=1)
            pred_betas = out_smpl_params['betas']
            pred_cam = out_cam
            pred_trans = output_cam_trans

            features = features.mean(dim=[2,3])

        return pred_poses, pred_betas, pred_cam, pred_trans, focal_length_, features
