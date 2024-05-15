import os
import types
import numpy
from torch.utils.data import Dataset as torch_dataset
from torch.utils.data import DataLoader as torch_dataloader

from dataloader import Dataset_Produce

'''
    本模块用于数据集的读取以及dataset的生成
'''


# 定义LoadData类 ，作用：将本地中的数据读取为一个dataset
class Dataset(torch_dataset):
    def __init__(self, config: types.SimpleNamespace):
        self.__data_labels = list()
        self.__suffix_name = list()
        self.__data = list()
        self.__dataloader_item = None
        self.__dataset_iterator = None
        self.__dataset = None
        self.__config = config
        self.get_local_filename(config.dataset_path)
        self.iterator_dataset(config)
        self.get_dataloader(self.__dataset_iterator, config)

    def __len__(self):
        return len(self.__data_labels)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index == self.__config.dataset_size - 1:
            try:
                result = self.__data[self.__index]
                self.__index = 0
                self.get_dataloader(self.__dataset_iterator, self.__config)
                return result
            except (IndexError, TypeError):
                raise StopIteration
        else:
            try:
                result = self.__data[self.__index]
                self.__index += 1
                return result
            except (IndexError, TypeError):
                raise StopIteration

    def get_local_filename(self, path):
        # 初始化数据的标签列表
        data_labels = list()
        # 遍历数据文件
        for file in os.listdir(path):
            # 在npy或npz后缀的文件中，将文件名中第一部分放在标签列表中
            if file.split(".")[-1] in ["npy", "npz"]:
                data_labels.append(file.split(".")[0])
        # 保存文件后缀名
        suffix_name = "." + os.listdir(path)[0].split(".")[-1]
        self.__data_labels = data_labels
        self.__suffix_name = suffix_name

    def read_local_data(self, path, size):
        count = 0
        select_labels = self.__data_labels[(size * count):(size * (count + 1))]
        while len(select_labels) == size:
            dataset = None
            # 用于返回数据的字典
            try:
                select_labels = self.__data_labels[(size * count):(size * (count + 1))]
            except IndexError:
                select_labels = self.__data_labels[(size * count):]
            if len(select_labels) > 0:
                for data_label in select_labels:
                    # data为读取的文件里面的数据
                    data = numpy.load(os.path.join(path, data_label + self.__suffix_name), allow_pickle=True)
                    # numpy.vstack(a,b)将a和b数组向下叠加
                    dataset = numpy.vstack((dataset, data)) if dataset is not None else data
            count = count + 1
            yield dataset

    def iterator_dataset(self, config):
        self.__dataset_iterator = self.read_local_data(config.dataset_path, config.dataset_size)

    def next_dataset(self, dataset_iterator):
        try:
            dataset_item = next(dataset_iterator)
            if dataset_item is not None:
                self.__dataset = Dataset_Produce(dataset_item)
            else:
                self.__dataset = None
        except StopIteration:
            self.__dataset = None

    def get_dataloader(self, dataset_iterator, config):
        self.next_dataset(dataset_iterator)
        if self.__dataset is not None:
            self.__dataloader_item = torch_dataloader(self.__dataset, batch_size=config.batch_size)
            self.__data = list(self.__dataloader_item)
        else:
            self.__data = None
            self.__dataloader_item = None
