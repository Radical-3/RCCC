import os
import torch
from tqdm import tqdm
from config import Config
from dataloader import Dataset
from detector import Detector_Controller
from log import logger
from mesh import Mesh
from render import Renderer
from utils.cleaning import is_garbage_sample


def dataset_cleaning():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)

    detector = Detector_Controller(config)
    rd = Renderer(config)
    dataset = Dataset(config, config.cleaning_dataset_path)
    mesh = Mesh(config).item()
    count = 0

    with torch.no_grad():
        with open(config.cleaning_result_path, 'w+') as F:
            with tqdm(dataset, desc="Cleaning") as pbar:
                for data in pbar:
                    if is_garbage_sample(data, config, detector, rd, mesh):
                        filename = f"{int(data[0].item())}.npy"
                        F.write(filename + '\n')
                        count += 1
                        if config.cleaning:
                            filepath = os.path.join(config.cleaning_dataset_path, filename)
                            os.remove(filepath)
                    pbar.set_postfix(bad_samples=count)
    logger.info(f"The cleaning is complete. A total of {count} samples were cleaned")
