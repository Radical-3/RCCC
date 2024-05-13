import numpy as np
import torch


def convert_to_numpy(images):
    # 如果图片不是numpy格式，就转换成numpy格式
    if isinstance(images, torch.Tensor):
        images = images[:, :, :, [2, 1, 0]].mul(255).detach().squeeze().cpu().numpy().astype(np.uint8)
    return images


# 去掉图片alpha通道
def remove_alpha(images):
    # 如果图片最后一个维度是4通道，删除最后一个alpha通道
    if images.shape[-1] == 4:
        return images[..., :3]
    return images
