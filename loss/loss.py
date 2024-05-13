import numpy as np
import torch


class Loss:
    def __init__(self, config, model):
        # 初始化实例，获取配置文件，目标检测模型，目标检测模型的类别名，运行使用的计算设备
        self.__config = config
        self.__model = model
        self.__names = model.get_names()

    # 最大识别概率损失
    def maximum_probability_score(self, result):
        # result = torch.Size([1, 80055, 85])
        # 获取需要攻击的类别
        classes = self.__config.train_classes
        # 确保传入的目标识别网络的计算结果符合[x,y,w,h,存在物体的概率，每个类别的概率(coco数据集共有80种类别)]
        assert (result.size(-1) == (5 + len(self.__names)))
        # 将锚框中每个类别的概率，单独截断出来
        class_confs = result[:, :, 5:5 + len(self.__names)]

        if classes is not None:
            # 若制定了攻击类别，则使用softmax函数，对所有类别的概率进行归一化，并获取指定攻击类别在全部识别概率中的占比。
            class_confs = torch.nn.Softmax(dim=2)(class_confs)
            class_confs = class_confs[:, :, classes]
        else:
            # 若未指定攻击类别，则直接获取识别结果中，，进行攻击
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]

        # 获取识别结果的第四项，即存在物体概率
        objectiveness_score = result[:, :, 4]
        # 保证两者维度相同，（batch-size为1时，会导致两者维度不同，objectiveness_score的batch-size维度缺失）
        if class_confs.shape[-1] != objectiveness_score.shape[-1]:
            objectiveness_score = objectiveness_score.unsqueeze(-1)

        # 若仅攻击物体存在概率，则结果只返回物体存在概率，否则返回存在概率和分类概率的乘积
        if self.__config.only_objectness:
            confs_if_object = objectiveness_score
        else:
            confs_if_object = objectiveness_score * class_confs

        # 最后返回所有锚框中 最大的概率得分的锚框的概率得分
        max_conf, _ = torch.max(confs_if_object, dim=1)
        return max_conf

    # 平滑函数
    @staticmethod
    def total_variation(image, mask):
        # 计算patch第二维即宽上相邻两像素之间的差值，使用绝对值函数保证正数，并求和,得到一个一维张量
        # 每个值对应列上元素差值，并将每一列的差值求和得到宽上所有列差值的总和
        variation_w = torch.sum(torch.abs(image[:, 1:, :] - image[:, :-1, :] + 0.000001), 0)
        variation_w = torch.sum(torch.sum(variation_w, 0), 0)
        # 计算patch第一维即高上相邻两像素之间的差值，使用绝对值函数保证正数，并求和，得到一个一维张量
        # 每个值对应行上元素差值，并将每一行的差值求和得到高上所有行差值的总和
        variation_h = torch.sum(torch.abs(image[1:, :, :] - image[:-1, :, :] + 0.000001), 0)
        variation_h = torch.sum(torch.sum(variation_h, 0), 0)
        variation = variation_w + variation_h
        num = image.size(0) * image.size(1) - np.count_nonzero(mask)
        return variation / num
