import numpy as np
import torch

from torch import optim
from tqdm import tqdm

from config import Config
from dataloader.dataset import Dataset
from detector import Detector_Controller
from log import logger
from mesh import Mesh
from camo import Camo
from render import Renderer
from loss import Loss, transform


def training_camouflage():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)
    detector = Detector_Controller(config, "yolov5")
    loss = Loss(config, detector)

    dataset = Dataset(config)
    rd = Renderer(config)
    ms = Mesh(config)
    camo = Camo(config, ms.shape())

    camo.load_mask()
    camo.requires_grad(True)
    optimizer = optim.Adam([camo.item()], lr=config.lr, amsgrad=True)

    for epoch in range(config.epochs):
        total_loss = list()

        with tqdm(dataset, desc=f"Epoch:{epoch}") as pbar:
            for data in pbar:
                ms.set_camo(camo)
                mesh = ms.item()

                dist, elev, azim = data[4][0, :].float()
                background = data[1].to(config.device).to(torch.float32) / 255
                mask = data[2].to(config.device).to(torch.float32)

                rd.set_camera_position(dist, elev, azim)
                image_without_background = rd.render(mesh)
                image_backup = image_without_background.clone().to(config.device)

                image_without_background = transform(config, image_without_background)
                image = image_without_background * mask + background * (1 - mask)
                result = detector.run(image)

                loss_maximum_probability_score = loss.maximum_probability_score(result)
                loss_total_variation = loss.total_variation(image_backup.squeeze(), data[2])

                loss_value = loss_maximum_probability_score + loss_total_variation
                loss_value.backward()

                total_loss.append(loss_value.item())
                optimizer.step()
                optimizer.zero_grad()
                camo.clamp()
                pbar.set_postfix(total_loss=np.mean(total_loss), loss=loss_value.item())

    if config.save_camo_to_pth:
        camo.save_camo_pth()

    if config.save_camo_to_txt:
        camo.save_camo_txt()

    if config.save_camo_to_png:
        ms.set_camo(camo)
        ms.make_texture_map_from_atlas()
