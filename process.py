'''
 @FileName    : process.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''

import torch
import numpy as np
import cv2
from tqdm import tqdm
import time

def extract_valid(data):
    batch_size, agent_num, d = data['keypoints'].shape[:3]
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'] #.reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length'] #.reshape(batch_size*agent_num,)[valid == 1]

    data['valid_img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]
    data['has_3d'] = data['has_3d'].reshape(batch_size*agent_num,1)[valid == 1]
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*agent_num,1)[valid == 1]
    data['verts'] = data['verts'].reshape(batch_size*agent_num, 6890, 3)[valid == 1]
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*agent_num, -1, 4)[valid == 1]
    data['pose'] = data['pose'].reshape(batch_size*agent_num, 72)[valid == 1]
    data['betas'] = data['betas'].reshape(batch_size*agent_num, 10)[valid == 1]
    data['keypoints'] = data['keypoints'].reshape(batch_size*agent_num, 26, 3)[valid == 1]
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*agent_num, 3)[valid == 1]

    data['ori_imgname'] = data['imgname']
    imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    data['imgname'] = imgname.tolist()

    return data

def extract_valid_group(data):
    batch_size, agent_num = data['scale'].shape[:3]
    valid = data['valid'].reshape(-1,)
    data['img'] = data['img']

    data['center'] = data['center']
    data['scale'] = data['scale']
    data['group_id'] = data['group_id']
    data['img_h'] = data['img_h']
    data['img_w'] = data['img_w']
    data['focal_length'] = data['focal_length']

    data['ori_imgname'] = data['imgname']
    imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    data['imgname'] = imgname.tolist()

    return data

def extract_valid_demo(data):
    batch_size, agent_num, d = data['features'].shape
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center']
    data['scale'] = data['scale']
    data['img_h'] = data['img_h']
    data['img_w'] = data['img_w']
    data['keypoints'] = data['keypoints'].reshape(batch_size*agent_num, 26, 3)[valid == 1]
    # data['focal_length'] = data['focal_length']

    # imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    # data['imgname'] = imgname.tolist()

    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname']} 
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname']}
    data = {**imnames, **data}

    return data


def group_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        data = to_device(data, device)
        data = extract_valid_group(data)

        # forward
        pred = model.model(data)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach() #/ batchsize
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        train_loss += loss_batch

    return train_loss/len_data

def connected_components(adj):
    """
    adj: (N, N) 0/1
    return group_id: (N,)
    """
    N = adj.shape[0]
    visited = torch.zeros(N, dtype=torch.bool, device=adj.device)
    group_id = torch.full((N,), -1, dtype=torch.long, device=adj.device)

    gid = 0

    for i in range(N):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        group_id[i] = gid

        while stack:
            node = stack.pop()
            neighbors = torch.where(adj[node] > 0)[0]

            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    group_id[nb] = gid
                    stack.append(nb)

        gid += 1

    return group_id

def compute_group_tp_fp_fn(pred_gid, gt_gid, threshold=0.5):
    """
    pred_gid: (N,) tensor
    gt_gid:   (N,) tensor

    return:
        TP, FP, FN
    """

    pred_groups = []
    for gid in pred_gid.unique():
        members = torch.where(pred_gid == gid)[0]
        pred_groups.append(set(members.tolist()))

    gt_groups = []
    for gid in gt_gid.unique():
        members = torch.where(gt_gid == gid)[0]
        gt_groups.append(set(members.tolist()))

    TP = 0
    FP = 0
    matched_gt = set()

    for pred_set in pred_groups:

        best_match = -1
        best_score = 0

        for i, gt_set in enumerate(gt_groups):

            inter = len(pred_set & gt_set)
            if inter == 0:
                continue

            score = inter / max(len(pred_set), len(gt_set))

            if score > threshold and score > best_score:
                best_score = score
                best_match = i

        if best_match >= 0 and best_match not in matched_gt:
            TP += 1
            matched_gt.add(best_match)
        else:
            FP += 1

    FN = len(gt_groups) - len(matched_gt)

    return TP, FP, FN

def compute_group_tp_fp_fn_new(pred_gid, gt_gid, crit='half'):
    """
    pred_gid: (N,) tensor
    gt_gid:   (N,) tensor

    return:
        TP, FP, FN
    """

    # ---- 1. 构造 group list ----
    pred_groups = []
    for gid in pred_gid.unique():
        members = torch.where(pred_gid == gid)[0]
        pred_groups.append(set(members.tolist()))

    gt_groups = []
    for gid in gt_gid.unique():
        members = torch.where(gt_gid == gid)[0]
        gt_groups.append(set(members.tolist()))

    # ---- 2. 计算 TP（核心：完全复刻 group_eval）----
    TP = 0

    for gt_set in gt_groups:
        gt_card = len(gt_set)

        for pred_set in pred_groups:
            pred_card = len(pred_set)

            inters = gt_set & pred_set
            inters_card = len(inters)

            if crit == 'half':
                if pred_card == 2 and gt_card == 2:
                    if len(gt_set - pred_set) == 0:
                        TP += 1
                elif inters_card / max(gt_card, pred_card) > 1/2:
                    TP += 1

            elif crit == 'card':
                if pred_card == 2 and gt_card == 2:
                    if len(gt_set - pred_set) == 0:
                        TP += 1
                elif inters_card / max(gt_card, pred_card) > 2/3:
                    TP += 1

            elif crit == 'dpmm':
                if pred_card == 2 and gt_card == 2:
                    if len(gt_set - pred_set) == 0:
                        TP += 1
                elif inters_card / max(gt_card, pred_card) > 0.6:
                    TP += 1

            elif crit == 'all':
                if len(gt_set - pred_set) == 0:
                    TP += 1

    # ---- 3. FP / FN（完全照搬）----
    FP = len(pred_groups) - TP
    FN = len(gt_groups) - TP

    return TP, FP, FN


def group_test(model, loss_func, loader, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)

    model.model.eval()

    total_TP = 0
    total_FP = 0
    total_FN = 0

    total_edge_pred = 0
    total_edge_correct = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            data = to_device(data, device)
            data = extract_valid_group(data)

            # forward
            pred = model.model(data)

            # 取出 logits 和 gt
            group_logit = pred['group_logit']      # (B, s * N)
            H = pred['H']                          # (B, s * N)
            valid_mask  = pred['valid_mask']       # (B, N)
            edge_mask = pred['edge_mask']         # (B, s * N)
            gt_group_id = data['group_id']         # (B, N)

            B = group_logit.shape[0]
            
            edge_prob = torch.sigmoid(group_logit)
            edge_pred = edge_prob > 0.4


            batch_TP = 0
            batch_FP = 0
            batch_FN = 0

            for b in range(B):

                valid = valid_mask[b].bool()
                edge_valid = edge_mask[b].bool()
                n_valid = valid.sum()

                if n_valid <= 1:
                    continue
                
                H_b = H[b][edge_valid][:, valid]      # (E, Nv)
                edge_pred_b = edge_pred[b][edge_valid]
                gt_gid_b = gt_group_id[b][valid]
                edge_prob_b = edge_prob[b][edge_valid]

                EE = H_b.shape[0]
                for e in range(EE):

                    members = torch.where(H_b[e] > 0)[0]
                    gids = gt_gid_b[members]

                    edge_gt = torch.all(gids == gids[0])
                    edge_pd = edge_pred_b[e]

                    if edge_gt == edge_pd:
                        total_edge_correct += 1

                    total_edge_pred += 1

                # build adjacency
                adj = torch.zeros(n_valid, n_valid, device=device)

                active_edges = torch.where(edge_pred_b)[0]

                for e in active_edges:

                    members = torch.where(H_b[e] > 0)[0]

                    for i in members:
                        for j in members:
                            adj[i, j] = 1

                pred_gid = connected_components(adj)

                # num_groups = pred_gid.unique().shape[0]

                TP, FP, FN = compute_group_tp_fp_fn_new(pred_gid, gt_gid_b)

                batch_TP += TP
                batch_FP += FP
                batch_FN += FN

            total_TP += batch_TP
            total_FP += batch_FP
            total_FN += batch_FN

            precision = batch_TP / (batch_TP + batch_FP + 1e-8)
            recall    = batch_TP / (batch_TP + batch_FN + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)

            print(f'batch {batch_idx}/{len(loader)} | TP:{batch_TP} FP:{batch_FP} FN:{batch_FN}')
            print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}') 

    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall    = total_TP / (total_TP + total_FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    edge_acc = total_edge_correct / (total_edge_pred + 1e-8)

    print('=' * 40)
    print(f'Edge Acc: {edge_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1:        {f1:.4f}')
    print('=' * 40)

    return edge_acc, precision, recall, f1

def relation_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        data = to_device(data, device)
        data = extract_valid(data)

        # forward
        pred = model.model(data)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach() #/ batchsize
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        train_loss += loss_batch

    return train_loss/len_data

def relation_test(model, loss_func, loader, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    global_idx = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['keypoints'].shape[0]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            
            if True:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['valid_img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['valid_img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=pred['focal_length'].detach().cpu().numpy().astype(np.float32))
                num_person = len(pred['pred_pose'])
                model.save_params(results, i, batchsize, global_idx)
                global_idx += num_person


            if i < 1:
                results = {}
                results.update(imgs=data['ori_imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=pred['focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(valid=data['valid'].detach().cpu().numpy().astype(np.float32))

                if 'MPJPE_instance' in cur_loss_dict.keys() or 'MPJPE_H36M_instance' in cur_loss_dict.keys():
                    results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))

                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    # model.save_results(results, i, batchsize)

            loss_batch = loss.detach().mean() #/ batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def relation_demo(model, loader, device=torch.device('cpu')):

    print('-' * 10 + 'model demo' + '-' * 10)
    print(f"[DEBUG] loader 的长度 (batch数量): {len(loader)}") 
    model.model.eval()
    print("enter demo function")
    with torch.no_grad():
        for i, data in enumerate(loader):
            print("ENTER LOOP")
            print("****")
            print(data['features'].shape)
            batchsize = data['features'].shape[0]
            data = to_device(data, device)
            data = extract_valid_demo(data)

            # forward
            print("2")
            pred = model.model(data)
            print("3")

            results = {}
            results.update(imgs=data['imgname'])
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=pred['focal_length'].detach().cpu().numpy().astype(np.float32))
            model.save_demo_results(results, i, batchsize)

