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