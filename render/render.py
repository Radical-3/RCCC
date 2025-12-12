import types

import numpy as np
import torch

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights, camera_position_from_spherical_angles, PerspectiveCameras,
)

from utils import remove_alpha

'''
    本模块使用pytorch3d中的renderer渲染器实现对3d模型的渲染，接收3d模型，输出渲染后的图片
'''


class Renderer:
    # 类初始化函数
    def __init__(self, config: types.SimpleNamespace):
        self.__config = config
        self.__device = torch.device("cpu")

        if config.use_cuda:
            self.__device = torch.device(config.device)

        self.__cameras = None

        self.__lights = None
        self.__raster_settings = None
        self.set_lights()
        self.set_raster_settings()

    def set_camera(self, eye_tensor, at_tensor, up_tensor, aspect_ratio=1.0):
        R, T = look_at_view_transform(
            eye=eye_tensor,
            at=at_tensor,
            up=up_tensor
        )

        self.__cameras = FoVPerspectiveCameras(R=R, T=T, fov=59.7, aspect_ratio=aspect_ratio, device=self.__device)
        # 2. [核心修改] 让光源跟随相机位置
        # eye_tensor 就是相机在当前渲染坐标系下的位置
        # 直接把灯放在相机这里，就像打开了闪光灯一样，永远照亮车
        # if eye_tensor is not None:
        #     # 如果觉得正面光太硬，可以稍微把灯往上抬一点，例如：
        #     light_pos = eye_tensor.clone()
        #     # light_pos[..., 2] += 10.0
        #     light_pos[..., 1] += 10.0
        #     self.set_lights(location=light_pos)
        #
        #     # 直接使用相机位置作为光源
        #     # self.set_lights(location=eye_tensor)

    # 设置相机位置，生成camera类
    def set_camera_position(self, dist, elev, azim, at=((0, 0, 0),)):
        # R, T = look_at_view_transform(dist * self.__config.scale, elev, azim, device=self.__device)
        R, T = look_at_view_transform(dist * self.__config.scale, elev, azim, at=at, degrees=True, device=self.__device)
        self.__cameras = FoVPerspectiveCameras(R=R, T=T, device=self.__device)

    def set_raster_settings(self):
        self.__raster_settings = RasterizationSettings(
            image_size=self.__config.render_image_size,
            blur_radius=self.__config.blur_radius,
            faces_per_pixel=self.__config.faces_per_pixel,
            bin_size=self.__config.bin_size,
            max_faces_opengl=self.__config.max_faces_opengl,
            max_faces_per_bin=self.__config.max_faces_per_bin,
            perspective_correct=self.__config.perspective_correct,
            clip_barycentric_coords=self.__config.clip_barycentric_coords,
            cull_backfaces=self.__config.cull_backfaces,
            z_clip_value=self.__config.z_clip_value,
            cull_to_frustum=self.__config.cull_to_frustum
        )

    def set_lights(self, location=None):
        if location is None:
            # 如果没传位置，使用配置文件中的默认位置
            self.__lights = PointLights(device=self.__device, location=[self.__config.light])
        else:
            # [修改] 增强健壮性，确保 location 格式符合 PointLights 要求
            if torch.is_tensor(location):
                # PointLights location 需要 (N, 3)，如果传入的是 (3,) 则升维
                if location.dim() == 1:
                    location = location.unsqueeze(0)
                # 确保在正确的 device 上
                location = location.to(self.__device)
                self.__lights = PointLights(device=self.__device, location=location)
            else:
                self.__lights = PointLights(device=self.__device, location=[location])

    def render(self, mesh):
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.__cameras,
                raster_settings=self.__raster_settings
            ),
            shader=SoftPhongShader(
                device=self.__device,
                cameras=self.__cameras,
                lights=self.__lights
            )
        )
        image = renderer(mesh)
        image = remove_alpha(image)
        return image
