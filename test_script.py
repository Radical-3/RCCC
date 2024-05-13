import time

import cv2
import numpy as np
import torch

from torch import optim
from tqdm import tqdm

from config import Config
from dataloader import Dataset
from detector import Detector_Controller
from log import logger
from mesh import Mesh
from camo import Camo
from render import Renderer
from loss import Loss, transform

config = Config(logger, './config/base.yaml').item()
logger.set_config(config)
detector = Detector_Controller(config, "yolov5")
Loss = Loss(config, detector)

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
            rd.set_camera_position(dist, elev, azim)
            background = data[1].to(config.device).to(torch.float32) / 255
            mask = data[2].to(config.device).to(torch.float32)
            image_without_background = rd.render(mesh)
            image_without_background = transform(config, image_without_background)

            image_backup = image_without_background.clone().to(config.device)
            image = image_without_background * mask + background * (1 - mask)
            result = detector.run(image)
            loss_maximum_probability_score = Loss.maximum_probability_score(result) * config.mps_weight
            loss_total_variation = torch.max(Loss.total_variation(image_backup.squeeze(), data[2]) * config.tv_weight,
                                             torch.tensor(0.1, device=config.device))
            loss = loss_maximum_probability_score + loss_total_variation
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            camo.clamp()
            pbar.set_postfix(total_loss=np.mean(total_loss), loss=loss.item())


if config.save_camo_to_pth:
    camo.save_camo()

if config.save_camo_to_txt:
    triangle_colors = list()
    colors = torch.round(camo.item().detach().mul(255)).cpu().numpy()
    for i, data in enumerate(tqdm(colors)):
        for j in range(data.shape[0]):
            for k in range(data.shape[1]):
                row = [i, j, k]
                row.extend(data[j, k])
                triangle_colors.append(row)
    triangle_colors = np.array(triangle_colors)
    with open("output/camo.txt", "w") as f:
        np.savetxt(f, triangle_colors, fmt="%d\t%d\t%d\t%d\t%d\t%d")

if config.save_camo_to_png:
    ms.set_camo(camo)
    ms.make_texture_map_from_atlas()



