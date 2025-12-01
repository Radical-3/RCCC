
import cv2
import numpy

import torch


from utils import convert_to_numpy, find_top_k_min_k_positions

from tqdm import tqdm


from mesh import Mesh
from camo import Camo
from render import Renderer
from loss import Loss, transform

from detector.neural_networks.track.OSTrack.tracking.seq_list import seq_list
from log import logger
from config import Config


def test_camera_position():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)
    renderer = Renderer(config)
    mesh = Mesh(config)
    camo = Camo(config, mesh.shape())
    camo.load_mask()
    if config.continue_train:
        camo.load_camo()
    dataset = seq_list(config)
    for seq in dataset:
        pbar = tqdm(range(0, len(seq.frames), 1), desc=f"Dataset {seq.name}")
        for i in pbar:
            mesh.set_camo(camo)
            data_np_temp = numpy.load(seq.frames[i], allow_pickle=True)
            data_temp = [torch.tensor(item) for item in data_np_temp]
            dist, elev, azim = data_temp[4].float()
            background_temp = data_temp[1].to(config.device).to(torch.float32) / 255
            mask_temp = data_temp[2].to(config.device).to(torch.float32)
            relative_remove = data_temp[5].float().tolist()

            # 对模板图像添加对抗伪装
            renderer.set_camera_position(dist, elev, azim, at=relative_remove)
            image_without_background_temp = renderer.render(mesh.item())

            x = convert_to_numpy(background_temp.unsqueeze(0))
            cv2.imshow('x', x)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            image_without_background_temp = transform(config, image_without_background_temp)
            y = convert_to_numpy(image_without_background_temp)
            cv2.imshow('y', y)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            image_temp = image_without_background_temp * mask_temp + background_temp * (1 - mask_temp)
            z = convert_to_numpy(image_temp)

            cv2.imshow('z', z)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

