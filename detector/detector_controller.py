from detector.yolov5 import yolov5
from detector.yolov6 import yolov6
from detector.yolov7 import yolov7

'''
    本模块用于对检测模型的调用，通过输入模型名字，输出对应模型的检测结果
'''


class detector_controller:
    # 初始化函数
    def __init__(self, config, model_name):
        # 加载配置文件，模型名称
        self.config = config
        self.model_name = model_name
        # 通过模型名分别实例化不同的检测类
        if self.model_name == "yolov5":
            self.detector = yolov5(self.config)
        elif self.model_name == "yolov6":
            self.detector = yolov6(self.config)
        elif self.model_name == "yolov7":
            self.detector = yolov7(self.config)

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
