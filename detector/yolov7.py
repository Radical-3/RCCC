import time
import numpy as np
import torch

from numpy import random
from detector.neural_networks.yolov7.models.experimental import attempt_load
from detector.neural_networks.yolov7.utils.datasets import letterbox
from detector.neural_networks.yolov7.utils.general import check_img_size, non_max_suppression, \
    scale_coords
from detector.neural_networks.yolov7.utils.plots import plot_one_box
from detector.detector import detector
'''
    本类用于生成目标检测网络-yolov7,对目标图像进行检测返回结果
'''


class yolov7(detector):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        self.__weights = config.yolov7_weights
        # 加载模型
        self.__model = attempt_load(self.__weights, map_location=self._device)
        # 读取模型后，获得模型最大步长，作为本类的步长
        self.__stride = int(self.__model.stride.max())
        # 加载模型中附带的检测类的名字
        self.__names = self.__model.module.names if hasattr(self.__model, 'module') else self.__model.names
        # 如果图片大小是单一数字，就*2  【960】->【960，960】
        if len(self._image_size) == 1:
            self._image_size = self._image_size * 2
        # 检查图片大小，图片大小必须能够整除步长
        self.__image_size = check_img_size(self._image_size, s=self.__stride)  # check image size
        # 预加载
        if self._device.type != 'cpu':
            self.__model(
                torch.zeros(1, 3, *self.__image_size, ).to(self._device).type_as(
                    next(self.__model.parameters())))  # run once
        # 不同类使用不同的颜色画边界框
        self.__colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.__names]

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
        t1 = time.time()
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
        t2 = time.time()
        # 计算第一部分时间
        dt[0] += t2 - t1
        # 将数据带入网络进行推理，得到网络输出结果
        pred = self.__model(img, augment=self._augment)[0]
        # t3时刻
        t3 = time.time()
        # 计算推理时间
        dt[1] += t3 - t2
        # 接收网络输出的数据、置信度阈值、IOU阈值等参数，进行非极大值抑制，去除冗余的检测框
        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms)
        # 计算NMS时间
        dt[2] += time.time() - t3
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
        # 如果检测到目标，根据检测结果画图展示
        if len(det):
            result_lb = 1
            # 将边界框坐标从原始图像尺寸缩放到输入图片的尺寸，并对结果四舍五入
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
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
                if self._save_img or self._view_img:
                    # 将类别名和置信度赋值给lable
                    label = f'{self.__names[int(cls)]} {conf:.2f}'
                    # 使用plot添加边界框和置信度，类别
                    plot_one_box(xyxy, im0, label=label, color=self.__colors[int(cls)], line_thickness=1)
        # 输出相关信息
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        # 得到画框处理后的结果
        im_result = im0
        # 放入结果列表
        result.append(im_result)
        result.append(result_lb)
        return result

    # 直接输出检测网络输出结果，不进行其他处理，用于训练
    def run(self, img, nms):
        # 如果图像是三维的，就增加一维-batch维，变成四维
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        # 送入神经网络模型，前向传播，得到结果，并返回
        pred = self.__model(img, augment=self._augment)[0]
        if nms:
            pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms)
        return pred
