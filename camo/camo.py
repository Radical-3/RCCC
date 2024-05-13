import numpy
import torch


class Camo:
    def __init__(self, config, shape):
        self.__config = config
        self.__device = torch.device("cpu")
        if config.use_cuda:
            self.__device = torch.device(config.device)
        self.__camo = torch.rand(shape).to(self.__device)
        self.__camo_mask = None

    def item(self):
        return self.__camo

    def camo_mask(self):
        return self.__camo_mask

    def requires_grad(self, status):
        self.__camo.requires_grad_(status)

    def clamp(self):
        self.__camo.data.clamp_(0, 1)

    def load_camo(self):
        self.__camo = torch.load(self.__config.test_camo_pth_path)

    def load_mask(self):
        camo_mask = torch.zeros(self.__camo.shape, device=self.__device)
        with open(self.__config.face_path, "r") as f:
            face_idx = numpy.array(list(map(int, f.readlines())))
        for face_id in face_idx:
            camo_mask[int(face_id), :, :, :] = 1
        self.__camo_mask = camo_mask

    def save_camo_pth(self):
        torch.save(self.__camo, self.__config.save_camo_pth_path)

    def save_camo_txt(self):
        triangle_colors = list()
        colors = torch.round(self.__camo.detach().mul(255)).cpu().numpy()
        for i, data in enumerate(colors):
            for j in range(data.shape[0]):
                for k in range(data.shape[1]):
                    row = [i, j, k]
                    row.extend(data[j, k])
                    triangle_colors.append(row)
        triangle_colors = numpy.array(triangle_colors)
        with open(self.__config.save_camo_txt_path, "w") as f:
            numpy.savetxt(f, triangle_colors, fmt="%d\t%d\t%d\t%d\t%d\t%d")