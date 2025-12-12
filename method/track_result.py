# 使用跟踪器跟踪视频帧，得到跟踪结果
import math
import os

import cv2
import numpy
import numpy as np
import torch

from utils import convert_to_numpy
from torch import optim
from tqdm import tqdm

from config import Config
from dataloader import Dataset
from detector import Detector_Controller
from log import logger
from mesh import Mesh
from camo import Camo
from render import Renderer
from loss import Loss, transform

from detector.neural_networks.track.OSTrack.tracking.seq_list import seq_list
from log import logger
from config import Config
from detector.neural_networks.track.OSTrack.lib.test.evaluation.tracker import Tracker
def get_params(relative_cam, relative_veh, scale=1, device="cuda"):
    # 解析数据
    cam_pos = np.array(relative_cam[0])
    cam_rot = relative_cam[1]
    veh_pos = np.array(relative_veh[0])
    veh_yaw = relative_veh[1][1]
    if isinstance(veh_yaw, torch.Tensor): veh_yaw = veh_yaw.item()

    # [核心修正] 必须恢复 Y 轴取反! 否则左右是反的 (镜像)
    def to_right_handed(pos):
        return np.array([pos[0], -pos[1], pos[2]])

    cam_pos_rh = to_right_handed(cam_pos)
    veh_pos_rh = to_right_handed(veh_pos)

    # 角度修正: 配合 Y 轴取反，Yaw 也要取反
    veh_yaw_corrected = -veh_yaw
    cam_yaw_corrected = -cam_rot[1]
    cam_pitch_corrected = cam_rot[0]

    # 相对位移
    diff_pos_world = (cam_pos_rh - veh_pos_rh) * scale

    # 逆向旋转
    theta_rad = math.radians(-veh_yaw_corrected)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    def rotate_vector(vec):
        x, y, z = vec
        x_new = x * cos_t - y * sin_t
        y_new = x * sin_t + y * cos_t
        return [x_new, y_new, z]

    eye_local = rotate_vector(diff_pos_world)

    # 计算 At (LookAt)
    p_rad = math.radians(cam_pitch_corrected)
    y_rad = math.radians(cam_yaw_corrected)
    forward_world = np.array([
        math.cos(p_rad) * math.cos(y_rad),
        math.cos(p_rad) * math.sin(y_rad),
        math.sin(p_rad)
    ])

    look_target_world = cam_pos_rh + forward_world * 10.0
    diff_target = (look_target_world - veh_pos_rh) * scale
    at_local = rotate_vector(diff_target)

    up_local = [0.0, 0.0, 1.0]

    return (torch.tensor([eye_local], dtype=torch.float32, device=device),
            torch.tensor([at_local], dtype=torch.float32, device=device),
            torch.tensor([up_local], dtype=torch.float32, device=device))

def track_result():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)
    detector = Detector_Controller(config)

    renderer = Renderer(config)
    mesh = Mesh(config)
    camo = Camo(config, mesh.shape())

    camo.load_mask()
    camo.load_camo()

    tracker = Tracker(config.tracker_name, config.tracker_param, config.dataset_name, config.run_id)
    params = tracker.get_parameters()
    params.debug = config.debug
    ostracker = tracker.create_tracker(params)
    dataset = seq_list(config)

    for seq in dataset:
        # 创建用于保存bbox的文件
        bbox_output_dir = os.path.join("./output", "bbox_results")
        os.makedirs(bbox_output_dir, exist_ok=True)
        bbox_file_path = os.path.join(bbox_output_dir, f"{seq.name}.txt")

        mesh.set_camo(camo)
        data_temp_np = numpy.load(seq.frames[0], allow_pickle=True)
        data_temp = [torch.tensor(item) for item in data_temp_np]
        relative_data = data_temp[5].float().squeeze()
        relative_cam = (relative_data[:3].tolist(), relative_data[3:6].tolist())
        relative_veh = (relative_data[6:9].tolist(), relative_data[9:12].tolist())
        eye_tensor, at_tensor, up_tensor = get_params(relative_cam, relative_veh, 1)
        background = data_temp[1].to(config.device).to(torch.float32) / 255
        mask = data_temp[2].to(config.device).to(torch.float32)

        renderer.set_camera(eye_tensor, at_tensor, up_tensor)
        image_without_background = renderer.render(mesh.item())

        # x = convert_to_numpy(image_without_background)
        # cv2.imshow('x', x)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # image_without_background = transform(config, image_without_background)
        # y = convert_to_numpy(image_without_background)
        # cv2.imshow('y', y)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image = image_without_background * mask + background * (1 - mask)
        image = image.squeeze(0)
        # image = convert_to_numpy(image)

        # cv2.imshow('z', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        init_info = seq.init_info()
        ostracker.my_initialize(image, init_info)

        # 打开文件准备写入bbox
        with open(bbox_file_path, 'w') as bbox_file:
            # 写入第一帧的初始bbox，保留3位小数
            init_bbox = init_info['init_bbox']
            formatted_bbox = [f"{val:.3f}" for val in init_bbox]
            bbox_file.write(f"{','.join(formatted_bbox)}\n")

            with tqdm(enumerate(seq.frames[1:], start=1), total=len(seq.frames[1:]), desc="Processing Frames") as pbar:
                for frame_num, frame_path in pbar:
            # for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
                    mesh.set_camo(camo)
                    data_np = numpy.load(frame_path, allow_pickle=True)
                    data = [torch.tensor(item) for item in data_np]
                    relative_data = data[5].float().squeeze()
                    relative_cam = (relative_data[:3].tolist(), relative_data[3:6].tolist())
                    relative_veh = (relative_data[6:9].tolist(), relative_data[9:12].tolist())
                    eye_tensor, at_tensor, up_tensor = get_params(relative_cam, relative_veh, 1)
                    background = data[1].to(config.device).to(torch.float32) / 255
                    mask = data[2].to(config.device).to(torch.float32)

                    renderer.set_camera(eye_tensor, at_tensor, up_tensor)
                    image_without_background = renderer.render(mesh.item())

                    image_without_background = transform(config, image_without_background)
                    # x = convert_to_numpy(image_without_background)
                    # cv2.imshow('x', x)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    image = image_without_background * mask + background * (1 - mask)
                    image = image.squeeze(0)
                    # image = convert_to_numpy(image)

                    result, bbox = ostracker.my_track(image)

                    # 将bbox写入文件，保留3位小数
                    formatted_bbox = [f"{val:.3f}" for val in bbox]
                    bbox_file.write(f"{','.join(formatted_bbox)}\n")
                    bbox_file.flush()  # 确保立即写入磁盘

                    # 将浮点数转换为整数，因为绘制图像时需要整数像素值
                    # x, y, w, h = map(int, bbox)
                    # # 使用 cv2.rectangle 在图像上绘制矩形框，参数分别是图像，左上角坐标，右下角坐标，颜色和线条宽度
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.imshow('x', image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

