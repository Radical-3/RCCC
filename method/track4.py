# 使用新的损失函数：正确位置的得分变小，边缘位置的得分变大
# 需要用到搜索图像的正确位置还是说使用原始搜索图像预测一下
import math
import os

import cv2
import numpy
import numpy as np
import torch
from torch.optim import lr_scheduler

from utils import convert_to_numpy, find_top_k_min_k_positions
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
from . import metrics
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

def track4():
    config = Config(logger, './config/base.yaml').item()
    logger.set_config(config)
    detector = Detector_Controller(config)
    loss = Loss(config, detector)

    renderer = Renderer(config)
    mesh = Mesh(config)
    camo = Camo(config, mesh.shape())

    camo.load_mask()
    if config.continue_train:
        camo.load_camo()
    camo.requires_grad(True)
    optimizer = optim.Adam([camo.item()], lr=float(config.lr), amsgrad=True)
    # optimizer = torch.optim.SGD(
    # [camo.item()],
    # lr = float(config.lr),
    # momentum = 0.9,  # 使用0.9的动量
    # weight_decay = 1e-5  # 添加权重衰减
    # )

    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',  # 模式为 'min'，因为我们希望最小化损失
    #     factor=0.8,  # 学习率衰减因子，每次减半
    #     patience=100,  # 如果损失5个epoch没有改善，就降低学习率
    #     verbose=True,  # 打印学习率变化信息
    #     threshold=1e-4,  # 损失变化的阈值，只有变化大于此值才算"改善"
    #     cooldown=0,  # 冷却期，降低学习率后，等待这么多epoch再重新监控
    #     min_lr=1e-6  # 学习率的下限，不能低于此值
    # )

    tracker = Tracker(config.tracker_name, config.tracker_param, config.dataset_name, config.run_id)
    params = tracker.get_parameters()
    params.debug = config.debug
    ostracker = tracker.create_tracker(params)
    dataset = seq_list(config)

    # 初始化指标存储
    metrics.init_metrics("./output/loss")

    for epoch in range(config.epochs):
        # 为每个数据集创建损失列表
        dataset_losses = {}
        for seq in dataset:
            dataset_losses[seq.name] = {
                "total": [],
                "score": [],
                "ciou": []
            }
        epoch_total_loss = list()
        for seq in dataset:
            pbar = tqdm(range(0, len(seq.frames), 2), desc=f"Epoch {epoch + 1}/{config.epochs} Dataset {seq.name}")
            for i in pbar:
                mesh.set_camo(camo)
                data_np_temp = numpy.load(seq.frames[i], allow_pickle=True)
                data_temp = [torch.tensor(item) for item in data_np_temp]
                dist, elev, azim = data_temp[4].float()
                background_temp = data_temp[1].to(config.device).to(torch.float32) / 255
                mask_temp = data_temp[2].to(config.device).to(torch.float32)
                # 将12个元素的tensor拆分为两个符合get_params输入要求的参数
                relative_data = data_temp[5].float().squeeze()
                relative_cam = (relative_data[:3].tolist(), relative_data[3:6].tolist())
                relative_veh = (relative_data[6:9].tolist(), relative_data[9:12].tolist())
                eye_tensor, at_tensor, up_tensor = get_params(relative_cam, relative_veh, 1)
                init_info = {'init_bbox': seq.ground_truth_rect[i]}

                data_np = numpy.load(seq.frames[i + 1], allow_pickle=True)
                data = [torch.tensor(item) for item in data_np]
                background = data[1].to(config.device).to(torch.float32) / 255

                # 使用干净搜索图像和干净模板图像预测正确结果
                ostracker.my_initialize(background_temp, init_info)
                result_clean, bbox_clean = ostracker.my_track_tensor(background)
                top_k_pos, min_k_pos = find_top_k_min_k_positions(result_clean, 2, 10)

                # 对模板图像添加对抗伪装
                renderer.set_camera(eye_tensor, at_tensor, up_tensor)
                image_without_background_temp = renderer.render(mesh.item())

                # x = convert_to_numpy(background_temp.unsqueeze(0))
                # cv2.imshow('x', x)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                image_without_background_temp = transform(config, image_without_background_temp)
                # y = convert_to_numpy(image_without_background)
                # cv2.imshow('y', y)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                image_temp = image_without_background_temp * mask_temp + background_temp * (1 - mask_temp)
                # z = convert_to_numpy(image_temp)
                #
                # cv2.imshow('z', z)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                image_temp = image_temp.squeeze(0)
                # image_temp.requires_grad = True

                # init_info = seq.my_init_info(i)
                # 使用添加伪装的模板图像初始化跟踪器
                ostracker.my_initialize(image_temp, init_info)

                # 对搜索图像添加伪装
                # 将12个元素的tensor拆分为两个符合get_params输入要求的参数
                relative_data = data[5].float().squeeze()
                relative_cam = (relative_data[:3].tolist(), relative_data[3:6].tolist())
                relative_veh = (relative_data[6:9].tolist(), relative_data[9:12].tolist())
                eye_tensor, at_tensor, up_tensor = get_params(relative_cam, relative_veh, 1)
                mask = data[2].to(config.device).to(torch.float32)

                renderer.set_camera(eye_tensor, at_tensor, up_tensor)
                image_without_background = renderer.render(mesh.item())

                image_backup = image_without_background.clone().to(config.device)
                image_without_background = transform(config, image_without_background)
                # x = convert_to_numpy(image_without_background)
                # cv2.imshow('x', x)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                image = image_without_background * mask + background * (1 - mask)
                # x = convert_to_numpy(image)
                # cv2.imshow('x', x)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                image = image.squeeze(0)
                # image.requires_grad = True
                # image = convert_to_numpy(image)

                result, bbox = ostracker.my_track_tensor(image)
                # 将浮点数转换为整数，因为绘制图像时需要整数像素值
                # x, y, w, h = map(int, bbox)
                # # 使用 cv2.rectangle 在图像上绘制矩形框，参数分别是图像，左上角坐标，右下角坐标，颜色和线条宽度
                # image_show = convert_to_numpy(image.unsqueeze(0))
                # cv2.rectangle(image_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow('x', image_show)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # loss_maximum_probability_score = loss.track_maximum_probability_score(result)
                loss_maximum_probability_score = loss.track_top_k_min_k_probability_score(result, top_k_pos, min_k_pos)
                # 保持梯度流，避免创建新的tensor打断计算图
                loss_ciou = loss.calculate_ciou_loss(bbox, bbox_clean)
                # # 改成了得分损失+距离损失
                # track_loss_score = loss.track_score_loss(result) * 10
                # track_loss_distance = loss.track_distance_loss(result)
                # loss_track = track_loss_score + track_loss_distance

                # loss_ostrack = loss.ostrack_loss(torch.tensor(bbox).unsqueeze(0), result.unsqueeze(0), torch.from_numpy(seq.ground_truth_rect[i+1]).unsqueeze(0))

                loss_total_variation = loss.total_variation(image_backup.squeeze(), data[2])

                # loss_value = loss_track + loss_total_variation
                # 对损失进行加权，使它们在同一数量级上
                # 统一损失权重，避免某些损失主导训练过程
                loss_value = loss_maximum_probability_score + loss_total_variation + loss_ciou
                # loss_value = loss_ostrack + loss_total_variation
                optimizer.zero_grad()
                # retain_graph=True加上这个，解决了两次传播的问题，但是不知道对结果有没有影响
                # 额，发现是每一次没有重新设置camo
                loss_value.backward()

                # 打印梯度信息
                # print("Camo gradient norm:", camo.item().grad.norm())  # 查看梯度的L2范数
                # print("Camo gradient max:", camo.item().grad.max())  # 查看梯度最大值
                # print("Camo gradient min:", camo.item().grad.min())  # 查看梯度最小值

                # loss_value.backward(retain_graph=True)
                dataset_losses[seq.name]["total"].append(loss_value.item())
                dataset_losses[seq.name]["score"].append(loss_maximum_probability_score.item())
                dataset_losses[seq.name]["ciou"].append(loss_ciou.item())
                epoch_total_loss.append(loss_value.item())
                epoch_average_loss = np.mean(dataset_losses[seq.name]["total"]) if dataset_losses[seq.name]["total"] else 0
                # scheduler.step(epoch_average_loss)
                # print(torch.all(camo.item().grad == 0))
                # print(image_temp.grad)
                # print(image.grad)
                # print(torch.all(image_temp.grad == 0))
                # print(torch.all(image.grad == 0))
                optimizer.step()
                camo.clamp()

                pbar.set_postfix(total_loss=f"{epoch_average_loss:.3f}",
                                 score_loss=f"{np.mean(dataset_losses[seq.name]['score']):.3f}",
                                 ciou_loss=f"{np.mean(dataset_losses[seq.name]['ciou']):.3f}",
                                 epoch_total_loss=f"{np.mean(epoch_total_loss):.3f}")
                # print(frame_num)

        # 记录当前epoch的平均损失（针对每个数据集）
        for dataset_name, losses in dataset_losses.items():
            if len(losses["total"]) > 0:
                avg_total_loss = np.mean(losses["total"])
                avg_score_loss = np.mean(losses["score"])
                avg_ciou_loss = np.mean(losses["ciou"])
                metrics.update_losses(dataset_name, epoch + 1, avg_total_loss, avg_score_loss, avg_ciou_loss)

        # 每100轮保存一次图片、损失和camo
        if (epoch + 1) % 100 == 0:
            # 创建保存目录
            save_dir = f"./output/checkpoint_epoch_{epoch + 1}"
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存camo
            camo.save_camo_pth(save_dir)
            
            # 保存纹理图
            mesh.set_camo(camo)
            mesh.make_texture_map_from_atlas(save_dir)
            
            # 保存当前损失图表
            metrics.plot_losses(save_dir)
            # metrics.save_losses_to_csv(save_dir)

    if config.save_camo_to_pth:
        camo.save_camo_pth("./output/tracker4")

    # 在所有epoch训练完成后绘制损失曲线
    metrics.plot_losses("./output/loss")
    
    if config.save_camo_to_png:
        mesh.set_camo(camo)
        mesh.make_texture_map_from_atlas("./output/tracker4")