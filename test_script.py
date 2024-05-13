import cv2
import torch
from torch import optim

from detector.detector_controller import detector_controller
from config import Config
from log import logger

config = Config(logger, './config/base.yaml').item()
logger.set_config(config)
detector = detector_controller(config, "yolov7")
image = cv2.imread("data/dataset/image4.png")
image_tensor = torch.load("data/dataset/1.pt")
image_tensor.require_grad = True
optimizer = optim.Adam([image_tensor], lr=0.01, amsgrad=True)
for i in range(3):
    # result = detector.detect(image)
    result = detector.run(image_tensor)
    class_confs = result[:, :, 5:5 + 80]
    class_confs = torch.nn.Softmax(dim=2)(class_confs)
    class_confs = class_confs[:, :, 2]
    objectiveness_score = result[:, :, 4]
    confs_if_object = objectiveness_score * class_confs
    max_conf, _ = torch.max(confs_if_object, dim=1)
    loss = max_conf
    loss.backward()
