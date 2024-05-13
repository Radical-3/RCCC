import os

import numpy
from torch.utils.data import Dataset as torch_dataset, DataLoader
from tqdm import tqdm

'''
    本模块用于数据集的读取以及dataset的生成
'''


# 定义LoadData类 ，作用：将本地中的数据读取为一个dataset
class Dataset(torch_dataset):
    def __init__(self, config):
        # 初始化数据的标签列表
        data_labels = list()
        # 遍历数据文件
        for file in os.listdir(config.dataset_path):
            # 在npy或npz后缀的文件中，将文件名中第一部分放在标签列表中
            if file.split(".")[-1] in ["npy", "npz"]:
                data_labels.append(file.split(".")[0])
        # 保存文件后缀名
        suffix_name = "." + os.listdir(config.dataset_path)[0].split(".")[-1]
        dataset = None
        # 用于返回数据的字典
        for data_label in tqdm(data_labels, desc="Loading data"):
            # data为读取的文件里面的数据
            data = numpy.load(os.path.join(config.dataset_path, data_label + suffix_name), allow_pickle=True)
            # numpy.vstack(a,b)将a和b数组向下叠加
            dataset = numpy.vstack((dataset, data)) if dataset is not None else data
        self.dataset = dataset
        self.dataloader = DataLoader(self, batch_size=config.batch_size)

    # 获得数据
    def __getitem__(self, index):
        identifier, image, image_mask, label, camera_position = self.dataset[index]
        return identifier, image, image_mask, label, camera_position

    # 得到数据长度
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # 返回数据加载器的迭代器
        return iter(self.dataloader)
