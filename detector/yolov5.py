import torch
import numpy as np

from .neural_networks.yolov5.utils.augmentations import letterbox
from .neural_networks.yolov5.models.experimental import attempt_load
from .neural_networks.yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes, \
    xyxy2xywh
from .neural_networks.yolov5.utils.torch_utils import time_sync
from .neural_networks.yolov5.utils.plots import Annotator, colors

from .detector import Detector

'''
    本类用于生成目标检测网络-yolov5,对目标图像进行检测返回结果
'''


class Yolov5(Detector):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        self.__weights = config.yolov5_weights
        # 如果有模型权重，就在加载模型，并赋值给w
        w = str(self.__weights[0] if isinstance(self.__weights, list) else self.__weights)
        # 初始化步长，初始化检测类别名字列表
        self.__stride, self.__names = 64, [f'class{i}' for i in range(1000)]
        # 如果在模型权重中，有torchscript语句，就调用torch.jit.load(w)方法加载，否则就调用attempt_load方法加载
        self.__model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.__weights, device=self._device)
        # 读取模型后，获得模型最大步长，作为本类的步长
        self.__stride = int(self.__model.stride.max())
        # 加载模型中附带的检测类的名字
        self.__names = self.__model.module.names if hasattr(self.__model, 'module') else self.__model.names

        # 如果图片大小是单一数字，就*2  【960】->【960，960】
        if len(self._image_size) == 1:
            self._image_size = self._image_size * 2
        # 检查图片大小，图片大小必须能够整除步长
        self.__image_size = check_img_size(self._image_size, s=self.__stride)
        # 预加载
        if self._device.type != 'cpu':
            self.__model(torch.zeros(1, 3, *self.__image_size, ).to(self._device).type_as(
                next(self.__model.parameters())))  # run once

    # 检测函数，接收图片，调用网络并进行检测，返回检测结果
    def detect(self, im0s):
        # 检测图片是否存在
        assert im0s is not None, 'Image Not Available '
        # 加载图片，重新设置图片格式
        img = letterbox(im0s, self.__image_size, stride=self.__stride)[0]
        # 将图片通道从HWC转换为CHW，最后一个RGB通道从BGR转换到RGB
        img = img.transpose((2, 0, 1))[::-1]
        # 将处理后的图像数据转换为一个连续的内存块表示
        img = np.ascontiguousarray(img)
        # 初始化时间列表
        dt = [0.0, 0.0, 0.0]
        # t1时刻
        t1 = time_sync()
        # 将图片从numpy格式转换为tensor格式
        img = torch.from_numpy(img).to(self._device)
        # 转换为浮点型数据
        img = img.float()
        # 归一化
        img = img / 255.0
        # 如果图像是三维的，就增加一维-batch维，变成四维
        if len(img.shape) == 3:
            img = img[None]
        # t2时刻
        t2 = time_sync()
        # 计算第一部分时间
        dt[0] += t2 - t1
        # 将数据带入网络进行推理，得到网络输出结果
        pred = self.__model(img, augment=self._augment, visualize=self._visualize)[0]
        # t3时刻
        t3 = time_sync()
        # 计算推理时间
        dt[1] += t3 - t2
        # 接收网络输出的数据、置信度阈值、IOU阈值等参数，进行非极大值抑制，去除冗余的检测框
        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                   max_det=self._max_det)
        # 计算NMS时间
        dt[2] += time_sync() - t3
        # 下面部分都是处理检测结果，在图片上添加检测框和数据操作
        result = []
        result_lb = 0
        # 接收NMS后的结果
        det = pred[0]
        s = ''
        # 复制图片用于画框展示
        im0 = im0s.copy()
        # 将图片的长宽赋值给s字符串
        s += '%gx%g ' % img.shape[2:]
        # 取出图片长宽，生成一个【whwh】的张量，用于边界框信息转换
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        # 创建一个用于图像标注的 annotator 对象，输入了图片，并指定了标注线的宽度和类别示例信息
        annotator = Annotator(im0, line_width=self._line_thickness, example=str(self.__names))
        # 如果检测到目标，根据检测结果画图展示
        if len(det):
            result_lb = 1
            # 将边界框坐标从原始图像尺寸缩放到输入图片的尺寸，并对结果四舍五入
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
            # print('det', det)
            # 遍历检测结果中的每个类别，输出结果
            for c in det[:, -1].unique():
                # 统计每个类别的检测结果数量
                n = (det[:, -1] == c).sum()
                # 将每个类别的检测结果数量和类别名称添加到字符串s中
                s += f"{n} {self.__names[int(c)]}{'s' * (n > 1)}, "
            # 记录结果，遍历所有检测到的边界框
            for *xyxy, conf, cls in reversed(det):
                # 如果要保存框的信息，就把边界框的信息写入文件
                if self._save_boxes:
                    # 将两点坐标的形式转换为一点坐标加长宽的形式
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # 如果要记录置信度，就记录类别、坐标、置信度，否则只记录其他两个
                    line = (cls, *xywh, conf) if self._save_conf else (cls, *xywh)
                    print(line)
                # 如果要保存图片或者裁剪或者看图片
                if self._save_img or self._save_crop or self._view_img:
                    # 当前检测类别的索引
                    c = int(cls)
                    # 如果隐藏标签就不输出，如果隐藏置信度，就只输出名字，否则类别置信度全输出
                    label = None if self._hide_labels else (
                        self.__names[c] if self._hide_conf else f'{self.__names[c]} {conf:.2f}')
                    # 将坐标，标签置信度给画图对象annotator
                    annotator.box_label(xyxy, label, color=colors(c, True))
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        # 得到画框处理后的结果
        im_result = annotator.result()
        # 放入结果列表
        result.append(im_result)
        result.append(result_lb)
        return result

    # 直接输出检测网络输出结果，不进行其他处理，用于训练
    def run(self, img, nms=False):
        # 如果图像是三维的，就增加一维-batch维，变成四维
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        # 送入神经网络模型，前向传播，得到结果，并返回
        pred = self.__model(img, augment=self._augment, visualize=self._visualize)[0]
        if nms:
            pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                       max_det=self._max_det)[0]
        return pred

    def get_names(self):
        return self.__names
