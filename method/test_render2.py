import math

import cv2
import numpy
import numpy as np

import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

from utils import convert_to_numpy, find_top_k_min_k_positions

from tqdm import tqdm


from mesh import Mesh
from camo import Camo
from render import Renderer
from loss import Loss, transform

from detector.neural_networks.track.OSTrack.tracking.seq_list import seq_list
from log import logger
from config import Config
def fix_mesh_orientation(pytorch3d_mesh, device):
    verts = pytorch3d_mesh.verts_packed()

    # 1. 先把车放平 (解决"站立"问题，这个是对的，不动)
    rot_mat_stand = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ], device=device)
    verts = torch.matmul(verts, rot_mat_stand)

    # 2. [核心修正] 暴力旋转 90 度
    # 既然翻转没用，说明是轴定义差了 90 度
    # 现在的现象：车头朝右下 (4点钟)
    # 目标现象：车头朝右上 (1-2点钟)
    # 操作：逆时针旋转 90 度 (Counter-Clockwise)

    angle_deg = 90.0  # <--- 如果还不对，改成 -90.0 绝对就对了

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # 绕 Z 轴旋转矩阵
    rot_mat_z = torch.tensor([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ], device=device)

    verts = torch.matmul(verts, rot_mat_z)

    # 3. 之前的一维翻转 (Flip) 先注释掉，避免干扰旋转
    # 如果旋转后方向对了但是左右反了，再把下面两行解开
    # verts[:, 0] = -verts[:, 0]
    # verts[:, 1] = -verts[:, 1]

    new_mesh = pytorch3d_mesh.update_padded(verts.unsqueeze(0))
    return new_mesh


# ==============================================================================
# 2. 参数计算: 恢复 Y 轴取反 (核心!)
# ==============================================================================
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

# 测试渲染的车的位置是否正确
def test_camera_position2():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)
    renderer = Renderer(config)
    mesh = Mesh(config)
    # mesh_item = fix_mesh_orientation(mesh.item(), config.device)
    camo = Camo(config, mesh.shape())
    camo.load_mask()
    if config.continue_train:
        camo.load_camo()
    dataset = seq_list(config)
    for seq in dataset:
        pbar = tqdm(range(0, len(seq.frames), 1), desc=f"Dataset {seq.name}")
        for i in pbar:
            mesh.set_camo(camo)
            data_np_temp = numpy.load(seq.frames[i], allow_pickle=True)
            data_temp = [torch.tensor(item) for item in data_np_temp]
            background_temp = data_temp[1].to(config.device).to(torch.float32) / 255
            mask_temp = data_temp[2].to(config.device).to(torch.float32)
            # 将12个元素的tensor拆分为两个符合get_params输入要求的参数
            relative_data = data_temp[5].float().squeeze()
            relative_cam = (relative_data[:3].tolist(), relative_data[3:6].tolist())
            relative_veh = (relative_data[6:9].tolist(), relative_data[9:12].tolist())
            eye_tensor, at_tensor, up_tensor = get_params(relative_cam, relative_veh, 1)
            # eye_tensor, at_tensor, up_tensor = get_params(relative_cam, relative_veh, 1)

            # 对模板图像添加对抗伪装
            renderer.set_camera(eye_tensor, at_tensor, up_tensor)
            image_without_background_temp = renderer.render(mesh.item())

            # 展示背景图像 x 并让用户框选区域
            x = convert_to_numpy(background_temp.unsqueeze(0))
            cv2.imshow("x", x)
            bbox_x = cv2.selectROI("x", x, fromCenter=False, showCrosshair=True)  # 返回 (x, y, w, h)
            cv2.destroyAllWindows()
            print(f"[ROI] bbox_x = {bbox_x}  # (x_min, y_min, w, h)")

            # ----------------------------------------------------

            # 展示渲染图像 y 并让用户框选区域
            image_without_background_temp = transform(config, image_without_background_temp)
            y = convert_to_numpy(image_without_background_temp)
            cv2.imshow("y", y)
            bbox_y = cv2.selectROI("y", y, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            print(f"[ROI] bbox_y = {bbox_y}  # (x_min, y_min, w, h)")
            image_temp = image_without_background_temp * mask_temp + background_temp * (1 - mask_temp)
            z = convert_to_numpy(image_temp)

            cv2.imshow('z', z)
            cv2.waitKey(0)
            cv2.destroyAllWindows()