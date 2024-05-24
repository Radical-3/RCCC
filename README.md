# Recognition Confuse Camo Coating(RCCC)

## Introduction

本仓库用于存放3D伪装迷彩涂装的相关代码和数据集，包括3D伪装迷彩涂装的生成代码、相关数据集的读取代码、3D伪装迷彩涂装的数据集等，使用组织内Carla-Control-Hub生成的内部格式数据集。

目前支持攻击的识别算法：

- [x] YOLOv5
- [x] YOLOv6
- [x] YOLOv7

## Installation

1. 本项目基于Python3.10.14开发,使用git克隆本项目，或者在当前页面直接下载zip文件
    ```shell
    python --version
   
    git clone
   
    ```
2. 项目基于pytorch3d 0.7.5版本开发，需要在安装依赖的情况后，再次安装pytorch和pytorch3d
   ```shell
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia # 安装pytorch2.1.0版本
   
   conda install -c fvcore -c iopath -c conda-forge fvcore iopath # 安装pytorch3d依赖包
   
   conda install pytorch3d -c pytorch3d # 安装pytorch3d

   pip install --upgrade iopath==0.1.10# 托管在conda云上的 0.1.9 iopath存在bug # 需要使用pip安装0.1.10的iopath
   ```
3. 使用git克隆本项目，或者在当前页面直接下载zip文件
    ```shell
    git clone
    ```
4. 对于不同识别算法需要安装不同的依赖包，如YOLOv5需要安装yolov5中的依赖包
    ```shell
    pip install -r requirements.txt
    
    pip install -r detector/neural_networks/yolov5/requirements.txt
    ```

## Usage

1. 生成伪装迷彩涂装
    ```shell
    python test.py
    ```
2. 修改配置文件`base.yaml`中可以直接改变生成的伪装迷彩涂装的参数
   例如：
   修改`base.yaml`中的`epoch`参数可以改变生成的伪装迷彩涂装的训练代数
3. 通过修改`test.py`中的模型名称可以改变生成的伪装迷彩涂装的模型
4. 生成的伪装迷彩涂装会保存在`output`文件夹中

## Contributing

本项目由[@黑羽彻](https://github.com/ZhaoZhiqiao)
开发维护, [@Caesar545](https://github.com/Caesar545)、[@He Fasheng](https://github.com/clown001)负责注释和文档编写。  
若项目存在任何问题，欢迎提交issue或者pull request，也可以直接通过邮箱联系[@黑羽彻](https://github.com/ZhaoZhiqiao):
zhao_zhiqiao@outlook.com。

## License

本项目为非开源的私有项目，未经允许禁止转载
