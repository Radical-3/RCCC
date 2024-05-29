from detector.yolov5 import Yolov5
from detector.yolov6 import Yolov6
from detector.yolov7 import Yolov7
from detector.yolov8 import Yolov8

'''
    本模块用于对检测模型的调用，通过输入模型名字，输出对应模型的检测结果
'''


class Detector_Controller:
    # 初始化函数
    def __init__(self, config):
        # 加载配置文件，模型名称
        self.config = config

        # 通过模型名分别实例化不同的检测类
        match self.config.detector:
            case "yolov5":
                self.detector = Yolov5(self.config)
            case "yolov6":
                self.detector = Yolov6(self.config)
            case "yolov7":
                self.detector = Yolov7(self.config)
            case "yolov8":
                self.detector = Yolov8(self.config)
            case _:
                raise NameError("This detector is not yet supported")

    # 直接得到模型检测网络输出的tensor数据
    def run(self, image, nms=False):
        # 调用run函数进行检测并返回检测结果
        image = image.permute(0, 3, 1, 2)
        # 调用run函数进行检测并返回检测结果
        result_run = self.detector.run(image, nms=nms)
        return result_run

    # 调用模型进行检测，返回进行过NMS后的数据
    def detect(self, image):
        result_detect = self.detector.detect(image)
        return result_detect

    def get_names(self):
        return self.detector.get_names()
