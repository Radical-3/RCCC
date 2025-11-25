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