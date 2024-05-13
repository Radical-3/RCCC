import types
import torch
from log import logger


class detector:
    def __init__(self, config: types.SimpleNamespace):
        self._config = config
        self._device = torch.device("cpu")
        if config.use_cuda:
            self._device = torch.device(config.device)
        self._image_size = config.img_sz
        self._conf_thres = config.conf_thres
        self._iou_thres = config.iou_thres
        self._max_det = config.max_det
        self._view_img = config.view_img
        self._save_boxes = config.save_boxes
        self._save_conf = config.save_conf
        self._classes = config.classes
        self._agnostic_nms = config.agnostic_nms
        self._visualize = config.visualize
        self._augment = config.augment
        self._line_thickness = config.line_thickness
        self._hide_labels = config.hide_labels
        self._hide_conf = config.hide_conf
        self._save_crop = config.save_crop
        self._save_img = config.save_img

    def detect(self, im0s):
        logger.error("You should implement this method in your subclass")
        raise NotImplementedError()

    def run(self, img, nms):
        logger.error("You should implement this method in your subclass")
        raise NotImplementedError()

    def get_names(self):
        logger.error("You should implement this method in your subclass")
        raise NotImplementedError()
