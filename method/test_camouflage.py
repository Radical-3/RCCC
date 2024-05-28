import os

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

from camo import Camo
from config import Config
from dataloader import Dataset
from detector import Detector_Controller
from log import logger
from mesh import Mesh
from render import Renderer
from utils import convert_to_numpy


def test_camouflage():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)
    detector = Detector_Controller(config)

    dataset = Dataset(config, config.test_dataset_path)
    renderer = Renderer(config)
    mesh = Mesh(config)
    mesh_with_camo = Mesh(config)

    camo = Camo(config, mesh_with_camo.shape())
    camo.load_camo()
    camo.load_mask()
    mesh_with_camo.set_camo(camo)

    mesh = mesh
    mesh_with_camo = mesh_with_camo

    os.makedirs(config.test_result_path, exist_ok=True)
    with torch.no_grad():
        with tqdm(dataset, desc=f"test") as pbar:
            for data in pbar:
                dist, elev, azim = data[4][0, :].float()
                background = data[1].to(config.device).to(torch.float32) / 255
                mask = data[2].to(config.device).to(torch.float32)

                renderer.set_camera_position(dist, elev, azim)
                image_without_background = renderer.render(mesh.item())
                image_without_background_with_camo = renderer.render(mesh_with_camo.item())

                image = image_without_background * mask + background * (1 - mask)
                image_with_camo = image_without_background_with_camo * mask + background * (1 - mask)

                image = convert_to_numpy(image)
                image_with_camo = convert_to_numpy(image_with_camo)

                result = list()
                result.append(detector.detect(image)[0])
                result.append(detector.detect(image_with_camo)[0])

                cv2.imwrite(os.path.join(config.test_result_path, f"{data[0].item()}.png"), np.hstack(result))

        for elev in np.linspace(5, 75, 10):
            images = list()
            with tqdm(np.linspace(0, 360, 360), desc=f"rotation gif({elev:.1f})") as pbar:
                for azim in pbar:
                    rd.set_camera_position(dist=6, elev=elev, azim=azim)

                    image = rd.render(mesh)
                    image_with_camo = rd.render(mesh_with_camo)

                    image = convert_to_numpy(image)
                    image_with_camo = convert_to_numpy(image_with_camo)

                    result = list()
                    result.append(detector.detect(image)[0])
                    result.append(detector.detect(image_with_camo)[0])

                    images.append(cv2.cvtColor(np.hstack(result), cv2.COLOR_BGR2RGB))

            path = os.path.join(config.test_result_path, f"rotation_{elev:.1f}.gif")
            logger.info(f"start drawing a gif to {path}")
            with imageio.get_writer(path, mode='I', duration=0.01) as writer:
                for img in images:
                    writer.append_data(img)
            logger.info(f"finish drawing a gif to {path}")
