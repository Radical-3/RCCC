import numpy as np
from detector.neural_networks.yolov9.utils.augmentations import letterbox
from detector.neural_networks.yolov9.utils.plots import Annotator, colors
import torch
from detector.detector import Detector
from detector.neural_networks.yolov9.models.experimental import attempt_load
from detector.neural_networks.yolov9.utils.general import check_img_size, Profile, non_max_suppression, \
    scale_boxes, LOGGER, xyxy2xywh


class Yolov9(Detector):
    def __init__(self, config):
        super().__init__(config)
        self.__weights = config.yolov9_weights
        self.__model = attempt_load(self.__weights, device=self._device)
        self.__stride = max(int(self.__model.stride.max()), 32)
        self.__names = self.__model.module.names if hasattr(self.__model, 'module') else self.__model.names
        if len(self._image_size) == 1:
            self._image_size = self._image_size * 2
        self.__image_size = check_img_size(self._image_size, s=self.__stride)

    def detect(self, im0s):
        assert im0s is not None, 'Image Not Available '
        s = 'image' + ':'
        dt = (Profile(), Profile(), Profile())
        img = letterbox(im0s, self.__image_size, stride=self.__stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        with dt[0]:
            img = torch.from_numpy(img).to(self._device)
            img = img.float()
            img /= 255
            if len(img.shape) == 3:
                img = img[None]
        with dt[1]:
            pred = self.__model(img, augment=self._augment, visualize=self._visualize)
        with dt[2]:
            pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                       max_det=self._max_det)
        result = []
        result_lb = 0
        im0 = im0s.copy()
        s += '%gx%g ' % img.shape[2:]
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        annotator = Annotator(im0, line_width=self._line_thickness, example=str(self.__names))
        det = pred[0]
        if len(det):
            result_lb = 1
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()
                s += f"{n} {self.__names[int(c)]}{'s' * (n > 1)}, "
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if self._save_boxes:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if self._save_conf else (cls, *xywh)  # label format
                    # print('\n' + str(line))
                if self._save_img or self._view_img:
                    c = int(cls)  # integer class
                    label = f'{self.__names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        # Print results
        t = tuple(x.t / 1 * 1E3 for x in dt)
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.__image_size)}' % t)
        im_result = im0
        result.append(im_result)
        result.append(result_lb)
        return result

    def run(self, img, nms):
        if len(img.shape) == 3:
            img = img[None]
        pred = self.__model(img, augment=self._augment, visualize=self._visualize)[0]
        objectiveness_score = torch.ones(1, 1, pred.shape[2]).to(self._device)
        pred = torch.cat((pred[:, :4, :], objectiveness_score, pred[:, 4:, :]), dim=1)
        pred = pred.permute(0, 2, 1)
        if nms:
            pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                       max_det=self._max_det)[0]
        return pred

    def get_names(self):
        return self.__names
