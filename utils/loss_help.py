import torch

def find_top_k_min_k_positions(tensor, k_max, k_min):
    """
    在一个4维张量中找到最大的k个数和最小的k个数的位置。
    假设输入张量的形状为 [B, C, H, W]。

    参数:
        tensor (torch.Tensor): 输入的4维张量。
        k (int): 要寻找的最大和最小元素的数量。

    返回:
        tuple: 一个包含两个列表的元组。
            - 第一个列表是最大的k个元素的位置，每个位置是一个元组 (b, c, h, w)。
            - 第二个列表是最小的k个元素的位置，每个位置是一个元组 (b, c, h, w)。
    """
    # 1. 检查输入张量的维度
    if tensor.dim() != 4:
        raise ValueError("输入张量必须是4维的，形状为 [B, C, H, W]。")

    B, C, H, W = tensor.shape

    # 2. 将张量展平成二维 (B*C, H*W)，以便进行全局排序
    #    我们也可以展平成一维，但保持 (B*C, H*W) 有助于后续恢复索引
    flattened_tensor = tensor.view(-1, H * W)

    # 3. 找到全局最大的 k 个元素及其索引
    #    torch.topk 默认返回最大的元素
    topk_values, topk_indices_flat = torch.topk(flattened_tensor, k_max, dim=1)

    # 4. 找到全局最小的 k 个元素及其索引
    #    设置 largest=False 以返回最小的元素
    min_k_values, min_k_indices_flat = torch.topk(flattened_tensor, k_min, dim=1, largest=False)

    # 5. 将展平的索引转换回原始的4维坐标 (b, c, h, w)
    def convert_indices(indices_flat):
        positions = []
        # indices_flat 的形状是 (B*C, k)
        for bc in range(indices_flat.shape[0]):
            for idx in range(indices_flat.shape[1]):
                # 获取在展平后的 H*W 维度上的索引
                flat_idx = indices_flat[bc, idx].item()

                # 计算对应的 b, c, h, w
                b = bc // C
                c = bc % C
                h = flat_idx // W
                w = flat_idx % W

                positions.append((b, c, h, w))
        return positions

    top_k_positions = convert_indices(topk_indices_flat)
    min_k_positions = convert_indices(min_k_indices_flat)

    return top_k_positions, min_k_positions


def get_hard_negative_positions(result, k=10, ignore_radius=4):
    """
    寻找背景中得分最高的 K 个点 (Hard Negatives)。

    参数:
        result: (B, 1, H, W) 模型的输出 (可以是 Logits 也可以是 Score，推荐传入 Logits)
        k: 选几个点
        ignore_radius: 忽略中心的半径 (保护真实目标不被选中)
    """
    B, C, H, W = result.shape
    flattened = result.view(B, -1)

    # 1. 创建中心 Mask (保护真实目标)
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H, device=result.device),
                          torch.arange(W, device=result.device), indexing='ij')

    # 计算到中心的距离平方
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    # 设定掩码半径，覆盖中心区域
    is_target = dist_sq < (ignore_radius ** 2)

    # 2. 将目标区域的得分设为 -inf，防止被选中
    flattened_masked = flattened.clone()
    flattened_masked[:, is_target.view(-1)] = -float('inf')

    # 3. 选取 Top-K (最大的 K 个背景点)
    # 这些点是当前模型最容易混淆的地方，攻击它们效率最高
    top_vals, top_indices = torch.topk(flattened_masked, k, dim=1)

    # 4. 转换坐标格式 (b, c, h, w)
    attack_positions = []
    # 假设 Batch Size 可能 > 1，这里做个循环适配
    for b in range(B):
        for idx in top_indices[b]:
            h_idx = (idx // W).item()
            w_idx = (idx % W).item()
            attack_positions.append((b, 0, h_idx, w_idx))

    return attack_positions