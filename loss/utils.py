import torch


def random_light(config, image):
    light = torch.rand(1, device=config.device).uniform_(config.min_light, config.max_light)
    image = image + light
    image.clamp_(0, 1)
    return image

