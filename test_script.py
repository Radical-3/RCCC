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
from utils import convert_to_numpy

config = Config(logger, './config/base.yaml').item()
logger.set_config(config)
detector = Detector_Controller(config, "yolov5")

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
            image = image_without_background * mask + background * (1 - mask)
            result = detector.run(image)
            loss_maximum_probability_score = Loss.maximum_probability_score(result) * config.mps_weight
            while True:
                cv2.imshow("output", convert_to_numpy(image))
                if cv2.waitKey(1) == ord('q'):
                    break

# rd.set_camera_position(8.0, 47.40421576000235, 120.72393414768057)
#
# image = rd.render(mesh)
# image = convert_to_numpy(image)
# cv2.imwrite("output/test.png", image)


