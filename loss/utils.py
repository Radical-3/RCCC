import torch


def transform(config, image):
    if config.add_brightness:
        image = random_brightness(config, image)
    if config.add_contrast:
        image = random_contrast(config, image)
    return image


def random_brightness(config, image):
    brightness = torch.rand(1, device=config.device).uniform_(config.min_brightness, config.max_brightness)
    image = image + brightness
    image.clamp_(0, 1)
    return image


def random_contrast(config, image):
    contrast = torch.rand(1, device=config.device).uniform_(config.min_contrast, config.max_contrast)
    image = image * contrast
    image.clamp_(0, 1)
    return image
