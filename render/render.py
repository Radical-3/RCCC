import types
import torch

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
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

    # 设置相机位置，生成camera类
    def set_camera_position(self, dist, elev, azim):
        R, T = look_at_view_transform(dist * self.__config.scale, elev, azim, device=self.__device)
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
            self.__lights = PointLights(device=self.__device, location=[self.__config.light])
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
