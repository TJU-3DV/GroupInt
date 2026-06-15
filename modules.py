'''
 @FileName    : modules.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''
import os
import torch
import time
import yaml
from utils.imutils import vis_img
from utils.logger import Logger
from loss_func import *
import torch.optim as optim
from utils.cyclic_scheduler import CyclicLRWithRestarts
from datasets.relation_group_data import Relation_Group_Data
from datasets.group_data import Group_Data
from datasets.demo_data import DemoData
from utils.smpl_torch_batch import SMPLModel
from utils.renderer_pyrd import Renderer
import cv2
from thop import profile
from copy import deepcopy
from utils.imutils import joint_projection
from utils.gui_3d import Gui_3d
from utils.FileLoaders import save_pkl
import sys
import platform
import colorsys

def init(note='occlusion', dtype=torch.float32, mode='eval', **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join('output', note)
    out_dir = os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the log for the current experiment
    logger = Logger(os.path.join(out_dir, 'log.txt'), title="template")
    logger.set_names([note])
    logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    if mode == 'eval':
        logger.set_names(['Surface', 'MPJPE', 'PA-MPJPE', 'PCK'])
    else:
        if kwargs.get('task') == 'group':
            logger.set_names(['Epoch', 'LR', 'Train Loss', 'edge_acc', 'Precision', 'Recall', 'F1'])
        else:
            logger.set_names(['Epoch', 'LR', 'Train Loss', 'Test Loss'])

    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./smpl/smpl/SMPL_NEUTRAL.pkl', 
                        data_type=dtype,
                    )

    return out_dir, logger, model_smpl


class LossLoader():
    def __init__(self, train_loss='L1', test_loss='L1', device=torch.device('cpu'), **kwargs):
        self.train_loss_type = train_loss.split(' ')
        self.test_loss_type = test_loss.split(' ')
        self.device = device

        # Parse the loss functions
        self.train_loss = {}
        for loss in self.train_loss_type:
            if loss == 'L1':
                self.train_loss.update(L1=L1(self.device))
            if loss == 'L2':
                self.train_loss.update(L2=L2(self.device))
            if loss == 'Flow_Loss':
                self.train_loss.update(Flow_Loss=Flow_Loss(self.device))
            if loss == 'Rec_Loss':
                self.train_loss.update(Rec_Loss=Rec_Loss(self.device))
            if loss == 'Keyp2d_L1':
                self.train_loss.update(Keyp2d_L1=Keyp2d_L1(self.device))
            if loss == 'SMPL_Loss':
                self.train_loss.update(SMPL_Loss=SMPL_Loss(self.device))
            if loss == 'Keyp_Loss':
                self.train_loss.update(Keyp_Loss=Keyp_Loss(self.device))
            if loss == 'Mesh_Loss':
                self.train_loss.update(Mesh_Loss=Mesh_Loss(self.device))
            if loss == 'Joint_Loss':
                self.train_loss.update(Joint_Loss=Joint_Loss(self.device))
            if loss == 'Skeleton_Loss':
                self.train_loss.update(Skeleton_Loss=Skeleton_Loss(self.device))
            if loss == 'Joint_abs_Loss':
                self.train_loss.update(Joint_abs_Loss=Joint_abs_Loss(self.device))
            if loss == 'Shape_reg':
                self.train_loss.update(Shape_reg=Shape_reg(self.device))
            if loss == 'Pose_reg':
                self.train_loss.update(Pose_reg=Pose_reg(self.device))
            if loss == 'Joint_reg_Loss':
                self.train_loss.update(Joint_reg_Loss=Joint_reg_Loss(self.device))
            if loss == 'Plane_Loss':
                self.train_loss.update(Plane_Loss=Plane_Loss(self.device))
            if loss == 'Edge_BCE_Loss':
                self.train_loss.update(Edge_BCE_Loss=Edge_BCE_Loss(self.device))
            if loss == 'Contrast_Loss':
                self.train_loss.update(Contrast_Loss=Contrastive_Loss(self.device))
            # You can define your loss function in loss_func.py, e.g., Smooth6D, 
            # and load the loss by adding the following lines

            # if loss == 'Smooth6D':
            #     self.train_loss.update(Smooth6D=Smooth6D(self.device))

        self.test_loss = {}
        for loss in self.test_loss_type:
            if loss == 'L1':
                self.test_loss.update(L1=L1(self.device))
            if loss == 'MPJPE':
                self.test_loss.update(MPJPE=MPJPE(self.device))
            if loss == 'MPJPE_H36M':
                self.test_loss.update(MPJPE_H36M=MPJPE_H36M(self.device))
            if loss == 'MPJPE_H36M_instance':
                self.test_loss.update(MPJPE_H36M_instance=MPJPE_H36M(self.device))
            if loss == 'MPJPE_2D':
                self.test_loss.update(MPJPE_2D=MPJPE_2D(self.device))
            if loss == 'PA_MPJPE':
                self.test_loss.update(PA_MPJPE=MPJPE(self.device))
            if loss == 'MPJPE_instance':
                self.test_loss.update(MPJPE_instance=MPJPE(self.device))
            if loss == 'PCK':
                self.test_loss.update(PCK=PCK(self.device))
            if loss == 'PCK_instance':
                self.test_loss.update(PCK_instance=PCK(self.device))

    def calcul_trainloss(self, pred, gt):
        loss_dict = {}
        gt['has_smpl'] = gt['has_smpl'].squeeze(1)
        gt['has_3d'] = gt['has_3d'].squeeze(1)

        for ltype in self.train_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.train_loss['L1'](pred, gt))
            elif ltype == 'L2':
                loss_dict.update(L2=self.train_loss['L2'](pred, gt))
            elif ltype == 'Flow_Loss':
                loss_dict.update(Flow_Loss=self.train_loss['Flow_Loss'](pred['pred_u_t'], pred['u_t']))
            elif ltype == 'Rec_Loss':
                loss_dict.update(Rec_Loss=self.train_loss['Rec_Loss'](pred['x_recon'], pred['x_1']))
            elif ltype == 'Keyp2d_L1':
                loss_dict.update(Keyp2d_L1=self.train_loss['Keyp2d_L1'](pred['pred_keypoints_2d'], gt['keypoints']))
            elif ltype == 'SMPL_Loss':
                # gt['has_smpl'] = gt['has_smpl'].squeeze(1)
                SMPL_loss = self.train_loss['SMPL_Loss'](pred['pred_rotmat'], gt['pose'], pred['pred_shape'], gt['betas'], gt['has_smpl'])
                loss_dict = {**loss_dict, **SMPL_loss}
            elif ltype == 'Plane_Loss':
                Plane_Loss = self.train_loss['Plane_Loss'](pred['pred_joints'], gt['valid'])
                loss_dict = {**loss_dict, **Plane_Loss}
            elif ltype == 'Keyp_Loss':
                Keyp_loss = self.train_loss['Keyp_Loss'](pred['pred_keypoints_2d'], gt['keypoints'])
                loss_dict = {**loss_dict, **Keyp_loss}
            elif ltype == 'Mesh_Loss':
                # gt['has_smpl'] = gt['has_smpl'].squeeze(1)
                Mesh_loss = self.train_loss['Mesh_Loss'](pred['pred_verts'], gt['verts'], gt['has_smpl'])
                loss_dict = {**loss_dict, **Mesh_loss}
            elif ltype == 'Joint_Loss':
                # gt['has_3d'] = gt['has_3d'].squeeze(1)
                Joint_Loss = self.train_loss['Joint_Loss'](pred['pred_joints'], gt['gt_joints'], gt['has_3d'])
                loss_dict = {**loss_dict, **Joint_Loss}
            elif ltype == 'Skeleton_Loss':
                Skeleton_Loss = self.train_loss['Skeleton_Loss'](pred['pred_joints'])
                loss_dict = {**loss_dict, **Skeleton_Loss}
            elif ltype == 'Joint_abs_Loss':
                # gt['has_3d'] = gt['has_3d'].squeeze(1)
                pred_joints_abs = pred['pred_joints'] + pred['pred_cam_t'][:,None,:]
                gt_joints_abs = gt['gt_joints'].detach()
                gt_joints_abs[:,:,:3] = gt_joints_abs[:,:,:3] + gt['gt_cam_t'][:,None,:]
                Joint_abs_Loss = self.train_loss['Joint_abs_Loss'](pred_joints_abs, gt_joints_abs, gt['has_3d'])
                loss_dict = {**loss_dict, **Joint_abs_Loss}
            elif ltype == 'Shape_reg':
                Shape_reg = self.train_loss['Shape_reg'](pred['pred_shape'])
                loss_dict = {**loss_dict, **Shape_reg}
            elif ltype == 'Pose_reg':
                Pose_reg = self.train_loss['Pose_reg'](pred['pred_pose'])
                loss_dict = {**loss_dict, **Pose_reg}
            elif ltype == 'Joint_reg_Loss':
                # gt['has_3d'] = gt['has_3d'].squeeze(1)
                Joint_reg_Loss = self.train_loss['Joint_reg_Loss'](pred['transformer_joints'], gt['gt_joints'], gt['has_3d'])
                loss_dict = {**loss_dict, **Joint_reg_Loss}
            elif ltype == 'Edge_BCE_Loss':
                loss_dict.update(Edge_BCE_Loss = self.train_loss['Edge_BCE_Loss'](pred['group_logit'], pred['edge_label'], pred['edge_mask'].bool()))
            elif ltype == 'Contrast_Loss':
                loss_dict.update(Contrast_Loss = self.train_loss['Contrast_Loss'](pred['tokens'], gt['group_id'], pred['valid_mask']))
            # Calculate your loss here

            # elif ltype == 'Smooth6D':
            #     loss_dict.update(Smooth6D=self.train_loss['Smooth6D'](pred_pose))
            else:
                pass
        loss = 0
        for k in loss_dict:
            loss_temp = loss_dict[k]
            loss += loss_temp
            loss_dict[k] = round(float(loss_temp.detach().cpu().numpy()), 6)
        return loss, loss_dict


    def calcul_testloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred, gt))
            elif ltype == 'MPJPE':
                loss_dict.update(MPJPE=self.test_loss['MPJPE'](pred['pred_joints'], gt['gt_joints']))
            elif ltype == 'MPJPE_H36M':
                loss_dict.update(MPJPE_H36M=self.test_loss['MPJPE_H36M'](pred['pred_verts'], gt['gt_joints']))
            elif ltype == 'MPJPE_H36M_instance':
                loss_dict.update(MPJPE_H36M_instance=self.test_loss['MPJPE_H36M_instance'].forward_instance(pred['pred_verts'], gt['gt_joints']))
            elif ltype == 'MPJPE_2D':
                loss_dict.update(MPJPE_2D=self.test_loss['MPJPE_2D'](pred['viz_output'], pred['viz_keyp']))
            elif ltype == 'PA_MPJPE':
                loss_dict.update(PA_MPJPE=self.test_loss['PA_MPJPE'].pa_mpjpe(pred['pred_joints'], gt['gt_joints']))
            elif ltype == 'PCK':
                loss_dict.update(PCK=self.test_loss['PCK'](pred['pred_joints'], gt['gt_joints']))
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass
        loss = 0
        for k in loss_dict:
            if 'instance' in k:
                assert len(loss_dict) == 1
                loss = loss_dict[k]
                loss_dict[k] = round(float(loss_dict[k].mean().detach().cpu().numpy()), 3)
                break
            loss += loss_dict[k]
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 3)
        return loss, loss_dict

    def calcul_instanceloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred, gt))
            elif ltype == 'MPJPE_instance':
                loss_dict.update(MPJPE_instance=self.test_loss['MPJPE_instance'].forward_instance(pred['pred_joints'], gt['gt_joints']))
            elif ltype == 'PCK_instance':
                loss_dict.update(PCK_instance=self.test_loss['PCK_instance'].forward_instance(pred['pred_joints'], gt['gt_joints']))
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass

        return loss_dict

def get_model_info(model, tsize):

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    data = {'features':torch.zeros((1,8,2048), device=next(model.parameters()).device),
            'center':torch.zeros((8,2), device=next(model.parameters()).device),
            'scale':torch.zeros((8,), device=next(model.parameters()).device),
            'valid':torch.ones((8,), device=next(model.parameters()).device),
            'img_h':torch.zeros((8,), device=next(model.parameters()).device),
            'img_w':torch.zeros((8,), device=next(model.parameters()).device),
            'focal_length':torch.zeros((8,), device=next(model.parameters()).device)}
    flops, params = profile(deepcopy(model), inputs=(data,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= 2  # Gflops
    # flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info

class ModelLoader():
    def __init__(self, dtype=torch.float32, output='', device=torch.device('cpu'), model=None, lr=0.001, pretrain=False, pretrain_dir='', batchsize=32, task=None, data_folder='', use_prior=False, testset='', test_loss='MPJPE', **kwargs):

        self.output = output
        self.device = device
        self.batchsize = batchsize
        self.data_folder = data_folder
        self.test_loss = test_loss
        if self.test_loss in ['PCK']:
            self.best_loss = -1
        else:
            self.best_loss = 999999999

        if task == 'group':
            self.best_f1 = -1

        # load smpl model 
        self.model_smpl_gpu = SMPLModel(
                            device=torch.device('cuda'),
                            model_path='./smpl/smpl/SMPL_NEUTRAL.pkl', 
                            data_type=dtype,
                        )
        # # Setup renderer for visualization
        # self.renderer = Renderer(focal_length=5000., img_res=224, faces=self.model_smpl_gpu.faces)

        if testset == 'JTA':
            num_joint = 15
        else:
            num_joint = 26

        # Load model according to model name
        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)(self.model_smpl_gpu, num_joints=num_joint)
        print('load model: %s' %self.model_type)

        # Calculate model size
        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count: %.2fM' % (model_params / 1e6))


        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        # print("Model Summary: {}".format(get_model_info(self.model, (800, 1440))))

        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = None

        # Load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            params = torch.load(pretrain_dir)
            premodel_dict = params['model']
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print("Load pretrain parameters from %s" %pretrain_dir)
            self.optimizer.load_state_dict(params['optimizer'])
            print("Load optimizer parameters")
            
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr

        if task == 'relation' and use_prior:
            model_dict = self.model.state_dict()
            params = torch.load('pretrain_model/mix_hmr300.pkl')
            premodel_dict = params['model']
            premodel_dict = {'backbone.' + k: v for k ,v in premodel_dict.items() if 'backbone.' + k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)

            # for parameter in self.model.backbone.parameters():
            #     parameter.requires_grad == False

    def load_scheduler(self, epoch_size):
        self.scheduler = CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=self.batchsize, epoch_size=epoch_size, restart_period=10, t_mult=2, policy="cosine", verbose=True)

    def load_checkpoint(self, pretrain_dir):
        model_dict = self.model.state_dict()
        params = torch.load(pretrain_dir)
        premodel_dict = params['model']
        premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
        model_dict.update(premodel_dict)
        self.model.load_state_dict(model_dict)
        print("Load pretrain parameters from %s" %pretrain_dir)

    def save_model(self, testing_loss, epoch, task):
        # save trained model
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        model_name = os.path.join(output, 'model_%s_epoch%03d_%.6f.pkl' %(task, epoch, testing_loss))
        torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
        print('save model to %s' % model_name)

    def save_best_model(self, testing_loss, epoch, task):
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        if self.test_loss in ['PCK']:
            if self.best_loss < testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)
        else:
            if self.best_loss > testing_loss and testing_loss != -1:
                self.best_loss = testing_loss

                model_name = os.path.join(output, 'best_%s_epoch%03d_%.6f.pkl' %(task, epoch, self.best_loss))
                torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
                print('save best model to %s' % model_name)

    def save_group_model(self, precision, recall, f1, epoch, task):
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)
        
        # if f1 > self.best_f1 and f1 != -1:
        #     self.best_f1 = f1

        model_name = os.path.join(output, 'best_%s_epoch%03d_prec%.3f_rec%.3f_f1%.3f.pkl' %(task, epoch, precision, recall, f1))
        torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
        print('save best model to %s' % model_name)

    def save_camparam(self, path, intris, extris):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        f = open(path, 'w')
        for ind, (intri, extri) in enumerate(zip(intris, extris)):
            f.write(str(ind)+'\n')
            for i in intri:
                f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
            f.write('0 0 \n')
            for i in extri[:3]:
                f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')
            f.write('\n')
        f.close()

    def save_params(self, results, iter, batchsize, start_idx):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['img_h'] = results['img_h'].reshape(-1,)
        results['img_w'] = results['img_w'].reshape(-1,)
        results['focal_length'] = results['focal_length'].reshape(-1,)

        for index, (img, pred_trans, pred_pose, pred_shape, h, w, focal) in enumerate(zip(results['imgs'], results['pred_trans'], results['pred_pose'], results['pred_shape'], results['img_h'], results['img_w'], results['focal_length'])):
            if sys.platform == 'linux':
                name = img.replace(self.data_folder + '/', '').replace('.jpg', '')
            else:
                name = img.replace(self.data_folder + '\\', '').replace('.jpg', '')
            data = {}
            data['pose'] = pred_pose
            data['trans'] = pred_trans
            data['betas'] = pred_shape

            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = w / 2
            intri[1][2] = h / 2
            extri = np.eye(4)
            
            cam_path = os.path.join(self.output, 'camparams', name)
            os.makedirs(cam_path, exist_ok=True)
            self.save_camparam(os.path.join(cam_path, 'camparams.txt'), [intri], [extri])

            global_id = start_idx + index
            path = os.path.join(self.output,  name)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, '%06d.pkl' %global_id)
            save_pkl(path, data)


    def save_results(self, results, iter, batchsize, individual=False):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]
        results['gt_verts'] = results['gt_verts'] + results['gt_trans'][:,np.newaxis,:]

        b, n = results['valid'].shape
        
        valid = results['valid'].reshape(-1,)

        pred_verts = np.zeros((b * n, 6890, 3), dtype=np.float32)
        gt_verts   = np.zeros((b * n, 6890, 3), dtype=np.float32)
        focal      = np.zeros((b * n,), dtype=np.float32)

        pred_verts[valid == 1] = results['pred_verts']
        gt_verts[valid == 1] = results['gt_verts']
        focal[valid == 1] = results['focal_length']

        pred_verts = pred_verts.reshape(b, n, 6890, 3)
        gt_verts = gt_verts.reshape(b, n, 6890, 3)
        focal = focal.reshape(b, n)

        if 'MPJPE' in results.keys():
            MPJPE = np.zeros((b * n,), dtype=np.float32)
            MPJPE[valid == 1] = results['MPJPE']
            MPJPE = MPJPE.reshape(b, n)

        valid = valid.reshape(b, n)

        if not individual:
            imgs = results['imgs'][0]
        else:
            imgs = results['imgs']

        for index, (img, pred_vert, gt_vert, f, v) in enumerate(zip(imgs, pred_verts, gt_verts, focal, valid)):
            # print(img)
            if platform.system() == 'Windows':
                name = img.replace(self.data_folder + '\\', '').replace('\\', '_').replace('/', '_')
            else:
                name = img.replace(self.data_folder + '/', '').replace('\\', '_').replace('/', '_')
            
            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=f[0], center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0], faces=self.model_smpl_gpu.faces,
                                same_mesh_color=False)

            pred_vert = pred_vert[v==1]
            pred_smpl = renderer.render_front_view(pred_vert,
                                                    bg_img_rgb=img.copy())
            side_smpl = renderer.render_side_view(pred_vert)

            gt_vert = gt_vert[v==1]
            gt_smpl = renderer.render_front_view(gt_vert,
                                                    bg_img_rgb=img.copy())
            gt_side_smpl = renderer.render_side_view(gt_vert)

            background = np.zeros_like(img)

            if 'MPJPE' in results.keys():
                losses = MPJPE[index][v==1]
                for i, loss in enumerate(losses):
                    mesh_color = np.array(colorsys.hsv_to_rgb(float(i) / len(losses), 0.5, 1.0))*255
                    background = cv2.putText(background, 'MPJPE %02d: ' %i + str(loss), (50,50*i+50), cv2.FONT_HERSHEY_COMPLEX, 1, tuple(mesh_color), 1)

            pred = np.concatenate((pred_smpl, side_smpl), axis=0)
            # gt_smpl = np.concatenate((gt_smpl, gt_side_smpl), axis=0)
            # background = np.concatenate((img, background), axis=0)
            # pred = np.concatenate((background, pred, gt_smpl), axis=1)

            render_name = "%s_%02d_pred_smpl.jpg" % (name, iter * batchsize + index)
            render_name_1 = "%s_%02d.jpg" % (name, iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), pred)
            # cv2.imwrite(os.path.join(output, render_name_1), pred_smpl)

            # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            # mesh_name = os.path.join(output, 'meshes/%s_%02d_pred_mesh.obj' %(name, iter * batchsize + index))
            # self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            # mesh_name = os.path.join(output, 'meshes/%s_%02d_gt_mesh.obj' %(name, iter * batchsize + index))
            # self.model_smpl_gpu.write_obj(gt_verts, mesh_name)
            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

    def save_demo_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_verts'] = results['pred_verts'] + results['pred_trans'][:,np.newaxis,:]

        for index, (img, pred_verts, focal, input) in enumerate(zip(results['imgs'], results['pred_verts'], results['focal_length'], results['origin_input'])):
            # print(img)
            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)

            pred_smpl = renderer.render_front_view(pred_verts[np.newaxis,:,:],
                                                    bg_img_rgb=img.copy())

            for kp in input:
                pred_smpl = cv2.circle(pred_smpl, tuple(kp[:2].astype(np.int)), 5, (0,0,255), -1)

            render_name = "%05d_pred_smpl.jpg" % (iter * batchsize + index)
            cv2.imwrite(os.path.join(output, render_name), pred_smpl)

            mesh_name = os.path.join(output, 'meshes/%05d_pred_mesh.obj' %(iter * batchsize + index))
            self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)

class DatasetLoader():
    def __init__(self, trainset=None, testset=None, data_folder='./data', dtype=torch.float32, smpl=None, task=None, model='hmr', **kwargs):
        self.data_folder = data_folder
        self.trainset = trainset.split(' ')
        self.testset = testset.split(' ')
        self.dtype = dtype
        self.smpl = smpl
        self.task = task
        self.model = model

    def load_trainset(self):
        train_dataset = []
        for i in range(len(self.trainset)):
            if self.task == 'relation':
                train_dataset.append(Relation_Group_Data(True, self.dtype, self.data_folder, self.trainset[i], self.smpl))
            elif self.task == 'group':
                train_dataset.append(Group_Data(True, self.dtype, self.data_folder, self.trainset[i], self.smpl))
                
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        return train_dataset

    def load_testset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            if self.task == 'relation':
                test_dataset.append(Relation_Group_Data(False, self.dtype, self.data_folder, self.testset[i], self.smpl))
            elif self.task == 'group':
                test_dataset.append(Group_Data(False, self.dtype, self.data_folder, self.testset[i], self.smpl))
                
        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset

    def load_demo_data(self):
        test_dataset = []
        for i in range(len(self.testset)):
            if self.task == 'relation':
                test_dataset.append(DemoData(False, self.dtype, self.data_folder, self.testset[i], self.smpl))

        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset
