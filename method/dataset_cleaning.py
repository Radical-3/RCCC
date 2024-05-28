import os

import torch
from tqdm import tqdm

from config import Config
from dataloader import Dataset
from detector import Detector_Controller
from log import logger
from mesh import Mesh
from render import Renderer


def dataset_cleaning():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)

    detector = Detector_Controller(config)
    rd = Renderer(config)
    dataset = Dataset(config, config.dataset_path)

    mesh = Mesh(config).item()
    count = 0
    with torch.no_grad():
        with open(config.cleaning_result_path, 'w+') as F:
            with tqdm(dataset, desc=f"Cleaning") as pbar:
                for data in pbar:
                    is_garbage = False
                    pbar.set_postfix(bad_samples=count)

                    dist, elev, azim = data[4][0, :].float()
                    background = data[1].to(config.device).to(torch.float32) / 255
                    mask = data[2].to(config.device).to(torch.float32)

                    rd.set_camera_position(dist, elev, azim)
                    image_without_background = rd.render(mesh)

                    image = image_without_background * mask + background * (1 - mask)
                    result_background = detector.run(background, nms=True)
                    result = detector.run(image, nms=True)

                    if len(result) != 1 or len(result_background) != 1:
                        is_garbage = True
                    elif result[0][4] < config.clean_conf_threshold or result_background[0][4] < config.clean_conf_threshold:
                        is_garbage = True
                    elif len(data[3][0]) < 1:
                        is_garbage = True
                    else:
                        result = result[0]
                        label = data[3][0].to(config.device).to(torch.float32)
                        x1 = max(label[0], result[0])
                        y1 = max(label[1], result[1])
                        x2 = min(label[2], result[2])
                        y2 = min(label[3], result[3])

                        area_bbox1 = (label[2] - label[0] + 1) * (label[3] - label[1] + 1)
                        area_bbox2 = (result[2] - result[0] + 1) * (result[3] - result[1] + 1)

                        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
                        union = area_bbox1 + area_bbox2 - intersection

                        iou = intersection / union
                        if iou < config.clean_iou_threshold:
                            is_garbage = True

                    if is_garbage:
                        filename = str(int(data[0].item())) + ".npy"
                        F.write(filename + '\n')
                        count += 1
                        if config.cleaning:
                            filepath = os.path.join(config.dataset_path, filename)
                            os.remove(filepath)
                        else:
                            continue
