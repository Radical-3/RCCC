import os
import numpy
from torch.utils.data import Dataset as torch_dataset
from torch.utils.data import DataLoader as torch_dataloader
from tqdm import tqdm

from .dataset_produce import Dataset_Produce

'''
    本模块用于数据集的读取以及dataset的生成
'''


# 定义LoadData类 ，作用：将本地中的数据读取为一个dataset
class Dataset(torch_dataset):
    def __init__(self, config, dataset_path):
        self.__config = config
        self.__dataset_path = dataset_path

        self.__data_labels = list()
        self.__suffix_name = list()
        self.__data = list()

        self.__dataloader_item = None
        self.__dataset_iterator = None
        self.__dataset = None

        self.__load_all = self.__config.all_dataset
        self.__local_filename(self.__dataset_path)
        self.__dataset_size = len(self.__data_labels) if self.__load_all else self.__config.dataset_size

        if self.__load_all:
            self.__dataset_iterator = self.__local_data(self.__dataset_path, self.__dataset_size)
            self.__dataloader(self.__dataset_iterator, self.__config)

    def __len__(self):
        return len(self.__data_labels)

    def __iter__(self):
        self.__index = 0
        if not self.__load_all:
            self.__dataset_iterator = self.__local_data(self.__dataset_path, self.__dataset_size)
            self.__dataloader(self.__dataset_iterator, self.__config)
        return self

    def __next__(self):
        if self.__load_all:
            try:
                result = self.__data[self.__index]
                self.__index += 1
                return result
            except (IndexError, TypeError):
                raise StopIteration
        else:
            if self.__index == self.__dataset_size - 1:
                try:
                    result = self.__data[self.__index]
                    self.__index = 0
                    self.__dataloader(self.__dataset_iterator, self.__config)
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

    def __local_filename(self, path):
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

    def __local_data(self, path, load_size):
        count = 0
        select_labels = self.__data_labels[(load_size * count):(load_size * (count + 1))]
        while len(select_labels) == load_size:
            dataset = None
            # 用于返回数据的字典
            try:
                select_labels = self.__data_labels[(load_size * count):(load_size * (count + 1))]
            except IndexError:
                select_labels = self.__data_labels[(load_size * count):]
            if len(select_labels) > 0:
                if self.__load_all:
                    select_labels = tqdm(select_labels, desc="Loading all data")
                for data_label in select_labels:
                    data = numpy.load(os.path.join(path, data_label + self.__suffix_name), allow_pickle=True)
                    dataset = numpy.vstack((dataset, data)) if dataset is not None else data
            count = count + 1
            yield dataset

    def __next_dataset(self, dataset_iterator):
        try:
            dataset_item = next(dataset_iterator)
            if dataset_item is not None:
                self.__dataset = Dataset_Produce(dataset_item)
            else:
                self.__dataset = None
        except StopIteration:
            self.__dataset = None

    def __dataloader(self, dataset_iterator, config):
        self.__next_dataset(dataset_iterator)
        if self.__dataset is not None:
            self.__dataloader_item = torch_dataloader(self.__dataset,
                                                      batch_size=config.batch_size,
                                                      shuffle=self.__config.shuffle)
            self.__data = list(self.__dataloader_item)
        else:
            self.__data = None
            self.__dataloader_item = None
