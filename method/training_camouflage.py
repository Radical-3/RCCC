import os

import numpy as np
import torch
import yaml

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


def training_camouflage():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)
    detector = Detector_Controller(config)
    loss = Loss(config, detector)

    dataset = Dataset(config, config.train_dataset_path)
    renderer = Renderer(config)
    mesh = Mesh(config)
    camo = Camo(config, mesh.shape())

    camo.load_mask()
    if config.continue_train:
        camo.load_camo()
    camo.requires_grad(True)
    optimizer = optim.Adam([camo.item()], lr=float(config.lr), amsgrad=True)

    for epoch in range(config.epochs):
        total_loss = list()

        with tqdm(dataset, desc=f"Epoch:{epoch}/{config.epochs}", ncols=200) as pbar:
            for data in pbar:
                mesh.set_camo(camo)

                dist, elev, azim = data[4][0, :].float()
                background = data[1].to(config.device).to(torch.float32) / 255
                mask = data[2].to(config.device).to(torch.float32)

                renderer.set_camera_position(dist, elev, azim)
                image_without_background = renderer.render(mesh.item())
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

                pbar.set_postfix(total_loss=f"{np.mean(total_loss):.3f}", loss=f"{loss_value.item():.3f}")

    if config.enable_save:
        save_path = os.path.join("output", config.save_folder)
        logger.info(f"save the result to {save_path}")

        count = 1
        while os.path.exists(save_path + f"_{count}"):
            count += 1
        save_path = save_path + f"_{count}"
        os.makedirs(save_path)

        if config.save_camo_to_pth:
            camo.save_camo_pth(save_path)

        if config.save_camo_to_png:
            mesh.set_camo(camo)
            mesh.make_texture_map_from_atlas(save_path)

        if config.save_config:
            with open(os.path.join(save_path, config.save_config_name), 'w') as file:
                yaml.dump(config.__dict__, file, default_flow_style=False, allow_unicode=True, default_style='"',
                          sort_keys=False)
            logger.info(f"save the config as the 'yaml' in {save_path}")
