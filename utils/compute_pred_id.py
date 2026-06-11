import torch

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


def solve(pred):
    """
    根据分组模型的输出，求解最终的 group_id
    """
    group_logit = pred['group_logit']      # (B, E)
    H = pred['H']                          # (B, E, N)
    valid_mask = pred['valid_mask']        # (B, N)
    edge_mask = pred['edge_mask']          # (B, E)

    B, N = valid_mask.shape
    device = valid_mask.device

    # 初始化最终的 group_id，无效人员默认为 -1
    final_group_id = torch.full((B, N), -1, dtype=torch.long, device=device)

    # 1. 阈值分割得到预测的超边
    edge_prob = torch.sigmoid(group_logit)
    edge_pred = edge_prob > 0.4

    for b in range(B):
        valid = valid_mask[b].bool()
        edge_valid = edge_mask[b].bool()
        n_valid = valid.sum().item()

        # 边界情况：没有有效人员或只有 1 个有效人员
        if n_valid <= 1:
            if n_valid == 1:
                valid_idx = torch.where(valid)[0].item()
                final_group_id[b, valid_idx] = 0
            continue

        # 2. 提取当前 batch 中有效的超边和有效人员
        H_b = H[b][edge_valid][:, valid]          # (E_valid, N_valid)
        edge_pred_b = edge_pred[b][edge_valid]    # (E_valid,)

        # 3. 初始化局部邻接矩阵 (N_valid, N_valid)
        adj = torch.zeros(n_valid, n_valid, dtype=torch.bool, device=device)

        # 4. 找出所有预测为 True 的活跃超边，并构建连通关系
        active_edges = torch.where(edge_pred_b)[0]
        for e in active_edges:
            members = torch.where(H_b[e] > 0)[0]
            if len(members) > 0:
                # 优化：使用 PyTorch 广播代替双重 for 循环，将 members 中的节点两两相连
                adj[members.unsqueeze(1), members.unsqueeze(0)] = True

        # 5. 计算连通分量，得到局部 valid 人员的 group_id
        pred_gid_valid = connected_components(adj)

        # 6. 将局部的 group_id 映射回全局 (B, N) 维度
        final_group_id[b, valid] = pred_gid_valid

    return final_group_id