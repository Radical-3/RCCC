from torch.utils.data import Dataset as Torch_Dataset


'''
    本模块用于数据集的读取以及dataset的生成  
'''


# 定义LoadData类 ，作用：将本地中的数据读取为一个dataset
class Dataset_Produce(Torch_Dataset):
    def __init__(self, dataset):
        # 检查传入的数据是否是列表，如果不是，将其包装在一个列表中
        if dataset[0].size == 1:
            dataset = [dataset]
        self.__dataset = dataset

    # 获得数据
    def __getitem__(self, index):
        identifier, image, image_mask, label, camera_position, relative_remove = self.__dataset[index]
        return identifier, image, image_mask, label, camera_position, relative_remove

    # 得到数据长度
    def __len__(self):
        return len(self.__dataset)
