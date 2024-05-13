import time
import torch
import numpy as np

from detector.neural_networks.yolov6.yolov6.utils.events import LOGGER, load_yaml
from detector.neural_networks.yolov6.yolov6.layers.common import DetectBackend
from detector.neural_networks.yolov6.yolov6.data.data_augment import letterbox
from detector.neural_networks.yolov6.yolov6.utils.nms import non_max_suppression
from detector.neural_networks.yolov6.yolov6.core.inferer import Inferer

from .detector import Detector
'''
    本类用于生成目标检测网络-yolov6,对目标图像进行检测返回结果
'''


class Yolov6(Detector):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        self.__weights = config.yolov6_weights
        self.__yaml = config.yaml
        self.__half = config.half
        self.__dict__.update(locals())
        # 加载模型参数
        self.__model = DetectBackend(self.__weights, device=self._device)
        # 读取模型后，获得模型步长
        self.__stride = self.__model.stride
        # 加载类别名称
        self.__class_names = load_yaml(self.__yaml)['names']
        # 将模型转换为部署状态
        self.model_switch(self.__model.model)
        # 执行了模型参数的精度转换，以便在部署时使用更低的浮点精度
        if self.__half & (self._device.type != 'cpu'):
            self.__model.model.half()
        else:
            self.__model.model.float()
            self.__half = False
        # 如果图片大小是单一数字，就*2  【960】->【960，960】
        if len(self._image_size) == 1:
            self._image_size = self._image_size * 2
        # 检查图片大小，图片大小必须能够整除步长
        self.__image_size = Inferer.check_img_size(self._image_size, s=self.__stride)  # check image size
        # 预加载
        if self._device.type != 'cpu':
            self.__model(torch.zeros(1, 3, *self._image_size).to(self._device).type_as(
                next(self.__model.model.parameters())))  # warmup

    # 转换模型的部署状态
    @staticmethod
    def model_switch(model):
        from detector.neural_networks.yolov6.yolov6.layers.common import RepVGGBlock
        # 遍历模型中的所有模块
        for layer in model.modules():
            # 如果有此模块，转换部署状态
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None
        # 记录一个日志信息
        LOGGER.info("Switch model to deploy modality.")

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
        pred_results = self.__model(img)
        # t3时刻
        t3 = time.time()
        # 计算推理时间
        dt[1] += t3 - t2
        # 接收网络输出的数据、置信度阈值、IOU阈值等参数，进行非极大值抑制，去除冗余的检测框
        det = non_max_suppression(pred_results, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                  max_det=self._max_det)[0]
        # 计算NMS时间
        dt[2] += time.time() - t3
        # 下面部分都是处理检测结果，在图片上添加检测框和数据操作
        # 取出图片长宽，生成一个【whwh】的张量，用于边界框信息转换
        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
        # 复制图片用于画框展示
        img_ori = im0s.copy()

        # 检查图片数据是否符合
        assert img_ori.data.contiguous, ('Image needs to be contiguous. Please apply to input images with '
                                         'np.ascontiguousarray(im).')
        # 检查字体文件
        Inferer.font_check()
        s = ''
        # 将图片的长宽赋值给s字符串
        s += '%gx%g ' % img.shape[2:]
        # 如果检测到目标，根据检测结果画图展示
        if len(det):
            result_lb = 1
            # 将边界框坐标从原始图像尺寸缩放到输入图片的尺寸，并对结果四舍五入
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], im0s.shape).round()
            # print('det', det)
            # 遍历检测结果中的每个类别,输出结果
            for c in det[:, -1].unique():
                # 统计每个类别的检测结果数量
                n = (det[:, -1] == c).sum()
                # 将每个类别的检测结果数量和类别名称添加到字符串s中
                s += f"{n} {self.__class_names[int(c)]}{'s' * (n > 1)}, "

            # 记录结果，遍历所有检测到的边界框
            for *xyxy, conf, cls in reversed(det):
                # 如果要保存框的信息，就把边界框的信息写入文件
                if self._save_boxes:
                    # 将两点坐标的形式转换为一点坐标加长宽的形式
                    xywh = (Inferer.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # 如果要记录置信度，就记录类别、坐标、置信度，否则只记录其他两个
                    line = (cls, *xywh, conf) if self._save_conf else (cls, *xywh)
                    print(line)
                # 如果要保存图片或者裁剪或者看图片
                if self._save_img or self._save_crop or self._view_img:
                    # 当前检测类别的索引
                    class_num = int(cls)
                    # 如果隐藏标签就不输出，如果隐藏置信度，就只输出名字，否则类别置信度全输出
                    label = None if self._hide_labels else (
                        self.__class_names[
                            class_num] if self._hide_conf else f'{self.__class_names[class_num]} {conf:.2f}')
                    # 调用函数进行画框
                    Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label,
                                               color=Inferer.generate_colors(class_num, True))
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # 将图片转换为numpy格式
            img_src = np.asarray(img_ori)
            return img_src, result_lb

    # 直接输出检测网络输出结果，不进行其他处理，用于训练
    def run(self, img, nms):
        # 如果图像是三维的，就增加一维-batch维，变成四维
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        # 送入神经网络模型，前向传播，得到结果，并返回
        pred = self.__model(img)
        if nms:
            pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                       max_det=self._max_det)
        return pred

    def get_names(self):
        return self.__class_names
