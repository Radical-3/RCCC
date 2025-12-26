import cv2
import numpy as np
import torch


def visualize_heatmap(image, score_map, bbox=None, save_path=None, alpha=0.5):
    """
    可视化热力图并叠加到原图上
    :param image: 原始图像 (Tensor [C, H, W] 或 Numpy [H, W, C])
    :param score_map: 跟踪器的得分图 (Tensor 或 Numpy)
    :param bbox: 预测框 [x, y, w, h] (可选)
    :param save_path: 保存路径
    :param alpha: 热力图透明度
    """
    # 1. 处理图像格式
    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().permute(1, 2, 0).numpy()  # 转为HWC
        image = (image * 255).astype(np.uint8)  # 假设输入是0-1归一化的
        # 如果原图是BGR (OpenCV格式) 不需要转RGB，如果是RGB则需要
        image = np.ascontiguousarray(image)

    img_h, img_w = image.shape[:2]

    # 2. 处理得分图格式
    if isinstance(score_map, torch.Tensor):
        score_map = score_map.detach().cpu().squeeze().numpy()

    # 归一化得分图到 0-255
    score_map_norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    score_map_norm = (score_map_norm * 255).astype(np.uint8)

    # 3. 生成热力图
    # 将 16x16 的热力图放大到原图尺寸
    heatmap = cv2.resize(score_map_norm, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    # 应用伪彩色 (COLORMAP_JET: 蓝-低, 红-高)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 4. 叠加图片
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    # 5. 画框 (如果提供了bbox)
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框

    # 6. 保存或显示
    if save_path:
        cv2.imwrite(save_path, overlay)
    else:
        # 如果不保存，可以在这里 cv2.imshow 调试
        cv2.imshow('Heatmap Visualization', overlay)
        cv2.waitKey(1)
        pass

def load_style_image(config, logger):
    logger.info(f"Loading style image from: {config.style_image_path}")
    style_img_np = cv2.imread(config.style_image_path)
    style_img_np = cv2.cvtColor(style_img_np, cv2.COLOR_BGR2RGB)
    # 转为 Tensor (1, 3, H, W) 并归一化
    style_img_tensor = torch.from_numpy(style_img_np).float() / 255.0
    style_img_tensor = style_img_tensor.permute(2, 0, 1).unsqueeze(0).to(config.device)
    # 注意：这里还没有 Resize，建议在 loop 里根据渲染尺寸 resize，或者固定一个尺寸


# ---新增辅助函数：生成重叠的滑动窗口 ---
def get_sliding_window_crops(mask, crop_size, stride):
    """
    生成覆盖物体的滑动窗口坐标列表。
    mask: (1, 1, H, W)
    crop_size: int (e.g., 224)
    stride: int (e.g., 112, 建议是 crop_size 的一半)
    """
    h, w = mask.shape[2], mask.shape[3]
    crops = []

    # 遍历高度和宽度
    # 这里的 range 保证了“全选”所有区域
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):

            # 检查这一块是否有物体（通过 Mask 判断）
            # 提取这一块的 Mask
            mask_patch = mask[:, :, y:y + crop_size, x:x + crop_size]

            # 如果这一块里 90% 以上都是黑色背景，就跳过，省显存
            # valid_pixels_ratio = mask_patch.mean()
            # if valid_pixels_ratio < 0.1:
            #     continue

            # 或者只要有一点点物体就保留 (为了边缘伪装)
            if mask_patch.max() > 0.1:
                crops.append((y, x))

    return crops