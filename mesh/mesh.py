import math
import os.path
import warnings

import cv2
import numpy as np
import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesAtlas
from pytorch3d.structures import Meshes

from log import logger
from utils import remove_alpha

'''
    本模块用于将mesh文件分解成各物体的image图像，通过输入3d模型的obj文件，输出每个物体的纹理图集
'''


def create_texture_map(uv_pos, texture, image, name, save_path):
    w, h = image.shape[0:2]
    uv_pos[..., 0] = (uv_pos[..., 0] * h).round()
    uv_pos[..., 1] = (uv_pos[..., 1] * w).round()
    uv_pos = uv_pos.detach().cpu().numpy().astype(np.int32)
    texture = texture.mul(255).detach().cpu().numpy().round().astype(np.int32)
    if h == 1 and w == 1 and len(np.unique(texture)) <= 3:
        image[0, 0] = np.unique(texture)
    else:
        for F in range(uv_pos.shape[0]):
            for R1 in range(uv_pos.shape[1]):
                for R2 in range(uv_pos.shape[2]):
                    cv2.fillConvexPoly(image, uv_pos[F, R1, R2], texture[F, R1, R2].tolist())
    image = np.flip(image, axis=0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, f"{name}_base_color.png"), image)
    logger.info(f"save the {name}_base_color as the 'png' in {save_path}")


class Mesh:
    # 类初始化函数
    def __init__(self, config):
        # 获取配置文件，运行使用的计算设备
        self.__config = config

        self.__device = torch.device("cpu")
        if config.use_cuda:
            self.__device = torch.device(config.device)

        # 忽视程序中的内置警告信息
        warnings.filterwarnings("ignore")

        self.__coordinate_bias = None

        # 调用load_obj函数，接收模型的obj文件，是否加载材质文件，是否创建每个面的纹理，每个面的纹理映射分辨率
        # 输出模型的顶点张量，面顶点索引张量，存储网格的辅助信息
        self.__vert, self.__faces, self.__aux = load_obj(self.__config.model_path,
                                                         device=self.__device,
                                                         load_textures=True,
                                                         create_texture_atlas=True,
                                                         texture_atlas_size=self.__config.texture_size,
                                                         texture_wrap="repeat")

        self.__fix_orientation()

        self.__atlas = self.__aux.texture_atlas
        self.__atlas_backup = self.__aux.texture_atlas.clone()

        idx = self.__faces.materials_idx.cpu().numpy()
        self.__face_material_names = np.array(list(self.__aux.texture_images.keys()))[idx]
        self.__face_material_names[idx == -1] = ""

    # =================================================================
    # [新增] 核心修复函数：站立 + 旋转90度 + 翻转Y + 缩放
    # =================================================================
    def __fix_orientation(self, angle_deg=90.0, scale_factor=1):
        """
        修正 CARLA 模型到 PyTorch3D 的坐标系差异
        1. 站立 (Y-up -> Z-up)
        2. 旋转 90 度 (解决朝向问题)
        3. 缩放 (解决大小问题)
        4. 镜像翻转 Y 轴 (解决左右反向问题)
        """
        # 1. 站立修正
        rot_mat_stand = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ], device=self.__device)
        self.__vert = torch.matmul(self.__vert, rot_mat_stand)

        # 2. 旋转修正 (逆时针 90 度)
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rot_mat_z = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0]
        ], device=self.__device)
        self.__vert = torch.matmul(self.__vert, rot_mat_z)

        # 3. 缩放修正
        self.__vert = self.__vert * scale_factor

        # 4. 镜像修正 (只翻转 Y)
        # 这一步把"右下"修正为"右上"
        self.__vert[:, 1] = -self.__vert[:, 1]

        logger.info("Mesh orientation, rotation, scale, and mirror fixed.")

    def set_camo(self, camo):
        self.__atlas = camo.item() * camo.camo_mask() + self.__atlas_backup * (1 - camo.camo_mask())

    # 返回tensor格式的纹理atlas
    def shape(self):
        return self.__atlas.shape

    # 重置纹理信息atlas
    def reset_atlas(self):
        self.__atlas = self.__atlas_backup

    def item(self):
        return Meshes(verts=[self.__vert], faces=[self.__faces.verts_idx], textures=TexturesAtlas(atlas=[self.__atlas]))

    def make_texture_map_from_atlas(self, save_path):
        self.__atlas = self.__atlas.detach()
        faces_verts_uvs = self.__aux.verts_uvs[self.__faces.textures_idx] if len(self.__aux.verts_uvs) > 0 else None
        save_path = str(os.path.join(save_path, self.__config.save_camo_png_name))
        os.makedirs(save_path)
        if faces_verts_uvs is None:
            logger.error("No UVs found in the mesh file")
            raise ValueError()

        for material_name, image in list(self.__aux.texture_images.items()):
            image = remove_alpha(image)
            image = np.zeros(image.shape)
            faces_material_ind = torch.from_numpy(self.__face_material_names == material_name).to(self.__device)
            texture = self.__atlas[faces_material_ind, :, :]
            self.calculate_coordinate_bias()

            uvs_subset = faces_verts_uvs[faces_material_ind, :, :] % 1.0
            uvs_subset[uvs_subset > 1] = 0
            uv_pos = (uvs_subset[:, None, None, None] * self.__coordinate_bias[..., None]).sum(-2)
            create_texture_map(uv_pos, texture, image, material_name, save_path)

    def calculate_coordinate_bias(self):
        correctness = [True]
        if self.__coordinate_bias is not None:
            correctness.append(self.__coordinate_bias.shape[0] == 3)
            correctness.append(self.__coordinate_bias.shape[1] == self.__config.texture_size)
            correctness.append(self.__coordinate_bias.shape[2] == self.__config.texture_size)
            correctness.append(self.__coordinate_bias.shape[3] == 3)
        if self.__coordinate_bias is None or not all(correctness):
            R = self.__config.texture_size
            rng = torch.arange(R, device=self.__device)
            Y, X = meshgrid_ij(rng, rng)
            grid = torch.stack([X, Y], dim=-1)
            below_diag = grid.sum(-1) < R

            bary = list()
            bary.append(torch.zeros((R, R, 3), device=self.__device))
            slc = torch.arange(2, device=self.__device)[:, None]
            bary[0][below_diag, slc] = (grid[below_diag] / R).T
            bary[0][~below_diag, slc] = ((R - 1.0 - grid[~below_diag] + 1.0) / R).T
            # bary[0][~below_diag, slc] = bary[0][~below_diag, slc].flip(0)
            bary[0][..., -1] = 1 - bary[0][..., :2].sum(dim=-1)

            bary.append(torch.zeros((R, R, 3), device=self.__device))
            grid_below_diag = torch.stack(((grid[below_diag][:, 0] + 1.0) / R, (grid[below_diag][:, 1]) / R), dim=-1)
            grid_above_diag = torch.stack(
                ((R - 1.0 - grid[~below_diag][:, 0] + 1.0) / R, (R - 1.0 - grid[~below_diag][:, 1]) / R), dim=-1)
            bary[1][below_diag, slc] = grid_below_diag.T
            bary[1][~below_diag, slc] = grid_above_diag.T
            # bary[1][~below_diag, slc] = bary[1][~below_diag, slc].flip(0)
            bary[1][..., -1] = 1 - bary[1][..., :2].sum(dim=-1)

            bary.append(torch.zeros((R, R, 3), device=self.__device))
            grid_below_diag = torch.stack(((grid[below_diag][:, 0]) / R, (grid[below_diag][:, 1] + 1) / R), dim=-1)
            grid_above_diag = torch.stack(
                ((R - 1.0 - grid[~below_diag][:, 0]) / R, (R - 1.0 - grid[~below_diag][:, 1] + 1) / R), dim=-1)
            bary[2][below_diag, slc] = grid_below_diag.T
            bary[2][~below_diag, slc] = grid_above_diag.T
            # bary[2][~below_diag, slc] = bary[2][~below_diag, slc].flip(0)
            bary[2][..., -1] = 1 - bary[2][..., :2].sum(dim=-1)
            self.__coordinate_bias = torch.stack(bary, dim=-2)
