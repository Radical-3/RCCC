import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import torchvision.models as models

# 复用之前定义的VGG19Style和预处理函数
class VGG19Style(nn.Module):
    def __init__(self):
        super(VGG19Style, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        for param in vgg19.parameters():
            param.requires_grad = False
        self.features = nn.Sequential()
        for layer in vgg19:
            if isinstance(layer, nn.MaxPool2d):
                self.features.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                self.features.append(layer)
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.layer_indices = {'conv1_1':1, 'conv2_1':6, 'conv3_1':11, 'conv4_1':20, 'conv5_1':29}
    def forward(self, x):
        features = {}
        for name, idx in self.layer_indices.items():
            for i in range(idx + 1):
                x = self.features[i](x)
            features[name] = x
        return features

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 原仓库常用尺寸，可调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 反向预处理（将tensor转回PIL图像，用于保存结果）
def deprocess(tensor):
    tensor = tensor.clone().squeeze(0)
    tensor *= torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    tensor += torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    tensor = torch.clamp(tensor, 0, 1)  # 限制像素值在0-1
    return transforms.ToPILImage()(tensor)