import numpy as np
import torch

from .neural_networks.yolov8.ultralytics.engine.results import Results
from .neural_networks.yolov8.ultralytics.nn.tasks import attempt_load_weights
from .neural_networks.yolov8.ultralytics.data.augment import LetterBox
from .neural_networks.yolov8.ultralytics.utils import ops
from .neural_networks.yolov8.ultralytics.utils.ops import non_max_suppression
from .detector import Detector


class Yolov8(Detector):
    def __init__(self, config):
        super().__init__(config)

        self.__weights = config.yolov8_weights
        self.__model = attempt_load_weights(self.__weights, device=self._device)
        self.__names = self.__model.module.names if hasattr(self.__model, "module") else self.__model.names
        self.__stride = max(int(self.__model.stride.max()), 32)
        self.__verbose = config.verbose
        # 如果图片大小是单一数字，就*2  【960】->【960，960】
        if len(self._image_size) == 1:
            self._image_size = self._image_size * 2

    def detect(self, im0s):
        assert im0s is not None, 'Image Not Available '
        profilers = (
            ops.Profile(device=self._device),
            ops.Profile(device=self._device),
            ops.Profile(device=self._device),
        )
        # Preprocess
        with profilers[0]:
            letterbox = LetterBox(new_shape=self._image_size, stride=self.__stride, auto=True)
            img = letterbox(image=im0s)
            img = (img[..., ::-1] / 255.0).astype(np.float32)
            img = img.transpose(2, 0, 1)[None]
            img = torch.from_numpy(img).to(self._device)
        # Inference
        with profilers[1]:
            predict = self.__model(img, augment=self._augment)
        # Postprocess
        with profilers[2]:
            preds = non_max_suppression(predict, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                        max_det=self._max_det)

        results = []
        for pred in preds:
            orig_img = im0s
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, names=self.__names, boxes=pred))

        results[0].speed = {
            "preprocess": profilers[0].dt * 1e3,
            "inference": profilers[1].dt * 1e3,
            "postprocess": profilers[2].dt * 1e3,
        }

        if self.verbose:
            box_details_list = []
            for i in range(pred.shape[0]):
                box_details = [
                    f"({int(pred[i, 0].item())}, {int(pred[i, 1].item())}, {int(pred[i, 2].item())}, {int(pred[i, 3].item())}, {float(pred[i, 4].item())}, {int(pred[i, 5].item())})"]
                box_details_list.append(f"Boxes: {', '.join(box_details)}")

            s = results[0].verbose() + f"{results[0].speed['inference']:.1f}ms\n" + '\n'.join(box_details_list)

            # LOGGER.info('\n' + s)
            print('\n' + s)
            return s

    def run(self, img, nms=False):
        # 如果图像是三维的，就增加一维-batch维，变成四维
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        pred = self.__model(img, augment=self._augment)[0]
        objectiveness_score = torch.ones(1, 1, pred.shape[2]).to(self._device)
        pred = torch.cat((pred[:, :4, :], objectiveness_score, pred[:, 4:, :]), dim=1)
        pred = pred.permute(0, 2, 1)
        if nms:
            pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                       max_det=self._max_det)[0]

        return pred

    def get_names(self):
        return self.__names
