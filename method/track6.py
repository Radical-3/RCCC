import math
import os
import cv2
import numpy
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import random  # <--- [新增] 必须导入 random

from config import Config
from dataloader import Dataset
from detector import Detector_Controller
from log import logger
from mesh import Mesh
from camo import Camo
from render import Renderer
from loss import Loss, transform

from detector.neural_networks.track.OSTrack.tracking.seq_list import seq_list
from detector.neural_networks.track.OSTrack.lib.test.evaluation.tracker import Tracker
from utils import get_hard_negative_positions
from . import metrics

# 修改训练模式，不是相邻帧训练
# 搜索帧从1开始以2为步长进行选取，然后模板帧从搜索帧前80帧内随机选取


# ... (保留 get_params 函数不变) ...
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
def track6():
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

    tracker = Tracker(config.tracker_name, config.tracker_param, config.dataset_name, config.run_id)
    params = tracker.get_parameters()
    params.debug = config.debug
    ostracker = tracker.create_tracker(params)
    dataset = seq_list(config)

    # 初始化指标存储
    metrics.init_metrics("./output/loss")

    # [新增] 定义最大随机间隔，建议 50-100，根据视频帧率调整
    # 间隔越大，训练越难，但生成的伪装对抗长时跟踪越有效
    MAX_TEMPLATE_GAP = 80

    for epoch in range(config.epochs):
        dataset_losses = {}
        for seq in dataset:
            dataset_losses[seq.name] = {
                "total": [],
                "score": [],
                "iou": []
            }
        epoch_total_loss = list()

        for seq in dataset:
            # 这里的 range(0, len-1, 2) 确保 i+1 不越界
            # 稍微调整了范围，防止 i+1 超过 len(seq.frames)
            pbar = tqdm(range(0, len(seq.frames) - 1, 2), desc=f"Epoch {epoch + 1}/{config.epochs} Dataset {seq.name}")

            for i in pbar:
                mesh.set_camo(camo)

                # =========================================================
                # [核心修改 START]: 随机间隔选取模板帧
                # =========================================================

                # 1. 确定搜索帧索引 (保持为当前进度的下一帧，模拟实时流)
                search_idx = i + 1

                # 2. 随机确定模板帧索引 (关键步骤！)
                # 逻辑：从 [max(0, i - MAX_TEMPLATE_GAP), i] 范围内随机选一帧
                # 这样既包含短时(相邻帧)，也包含长时(相隔50帧)，增加鲁棒性
                min_template_idx = max(0, i - MAX_TEMPLATE_GAP)
                template_idx = random.randint(min_template_idx, i)

                # 3. 加载模板帧数据 (使用 template_idx)
                data_np_temp = numpy.load(seq.frames[template_idx], allow_pickle=True)
                data_temp = [torch.tensor(item) for item in data_np_temp]

                background_temp = data_temp[1].to(config.device).to(torch.float32) / 255
                mask_temp = data_temp[2].to(config.device).to(torch.float32)

                # 解析模板帧的相机参数
                relative_data_temp = data_temp[5].float().squeeze()
                relative_cam_temp = (relative_data_temp[:3].tolist(), relative_data_temp[3:6].tolist())
                relative_veh_temp = (relative_data_temp[6:9].tolist(), relative_data_temp[9:12].tolist())
                eye_tensor_temp, at_tensor_temp, up_tensor_temp = get_params(relative_cam_temp, relative_veh_temp, 1)

                # [重要] 必须使用模板帧对应的 GT 来初始化
                init_info = {'init_bbox': seq.ground_truth_rect[template_idx]}

                # =========================================================
                # [核心修改 END]
                # =========================================================

                # 4. 加载搜索帧数据 (使用 search_idx)
                data_np = numpy.load(seq.frames[search_idx], allow_pickle=True)
                data = [torch.tensor(item) for item in data_np]
                background = data[1].to(config.device).to(torch.float32) / 255

                # --- 预测 Clean 结果 (作为对照) ---
                # 使用带伪装逻辑前的纯净图做初始化和跟踪
                ostracker.my_initialize(background_temp, init_info)
                result_clean, bbox_clean = ostracker.my_track_tensor(background)

                # 获取攻击目标位置 (Hard Negative Mining)
                clean_top_vals, clean_top_inds = torch.topk(result_clean.view(1, -1), 1)

                # 动态获取特征图宽度，防止硬编码 16 出错
                feat_w = result_clean.shape[-1]
                top_k_pos = []
                for idx in clean_top_inds[0]:
                    top_k_pos.append((0, 0, (idx // feat_w).item(), (idx % feat_w).item()))

                target_position = get_hard_negative_positions(result_clean, 6, 1)

                # --- 渲染模板帧 (Template) ---
                renderer.set_camera(eye_tensor_temp, at_tensor_temp, up_tensor_temp)
                image_without_background_temp = renderer.render(mesh.item())

                # 数据增强 (Transform) - 训练时必须开启！
                image_without_background_temp = transform(config, image_without_background_temp)

                image_temp = image_without_background_temp * mask_temp + background_temp * (1 - mask_temp)
                image_temp = image_temp.squeeze(0)

                # 使用带伪装的模板初始化
                ostracker.my_initialize(image_temp, init_info)

                # --- 渲染搜索帧 (Search) ---
                relative_data = data[5].float().squeeze()
                relative_cam = (relative_data[:3].tolist(), relative_data[3:6].tolist())
                relative_veh = (relative_data[6:9].tolist(), relative_data[9:12].tolist())
                eye_tensor, at_tensor, up_tensor = get_params(relative_cam, relative_veh, 1)
                mask = data[2].to(config.device).to(torch.float32)

                renderer.set_camera(eye_tensor, at_tensor, up_tensor)
                image_without_background = renderer.render(mesh.item())
                image_backup = image_without_background.clone().to(config.device)

                # 数据增强
                image_without_background = transform(config, image_without_background)

                image = image_without_background * mask + background * (1 - mask)
                image = image.squeeze(0)

                # --- 跟踪与 Loss 计算 ---
                # 注意：这里 return_score_logits=True 获取的是 logits 用于 Loss
                # 但 OSTrack 内部其实还是会过一次汉宁窗 (如果在 my_track_tensor 里写了的话)
                result, bbox = ostracker.my_track_tensor(image, return_score_logits=True)

                # 计算得分损失 (Logit Attack)
                loss_maximum_probability_score = loss.track_logit_attack_loss(result, top_k_pos, target_position) * 40

                # 计算 IoU 损失
                loss_iou = loss.iou_attack_loss(bbox, bbox_clean)

                # 计算平滑损失 (Total Variation)
                loss_total_variation = loss.total_variation(image_backup.squeeze(), data[2])

                # 总 Loss
                loss_value = loss_maximum_probability_score + loss_total_variation + loss_iou

                optimizer.zero_grad()
                loss_value.backward()

                dataset_losses[seq.name]["total"].append(loss_value.item())
                dataset_losses[seq.name]["score"].append(loss_maximum_probability_score.item())
                dataset_losses[seq.name]["iou"].append(loss_iou.item())
                epoch_total_loss.append(loss_value.item())

                epoch_average_loss = np.mean(dataset_losses[seq.name]["total"]) if dataset_losses[seq.name][
                    "total"] else 0

                optimizer.step()
                camo.clamp()

                pbar.set_postfix(total_loss=f"{epoch_average_loss:.3f}",
                                 score_loss=f"{np.mean(dataset_losses[seq.name]['score']):.3f}",
                                 iou_loss=f"{np.mean(dataset_losses[seq.name]['iou']):.3f}",
                                 epoch_total_loss=f"{np.mean(epoch_total_loss):.3f}")

        # ... (后续保存逻辑保持不变) ...
        for dataset_name, losses in dataset_losses.items():
            if len(losses["total"]) > 0:
                avg_total_loss = np.mean(losses["total"])
                avg_score_loss = np.mean(losses["score"])
                avg_iou_loss = np.mean(losses["iou"])
                metrics.update_losses(dataset_name, epoch + 1, avg_total_loss, avg_score_loss, avg_iou_loss)

        if (epoch + 1) % 100 == 0:
            save_dir = f"./output/checkpoint_epoch_{epoch + 1}"
            os.makedirs(save_dir, exist_ok=True)
            camo.save_camo_pth(save_dir)
            mesh.set_camo(camo)
            mesh.make_texture_map_from_atlas(save_dir)
            metrics.plot_losses(save_dir)

    if config.save_camo_to_pth:
        camo.save_camo_pth("./output/tracker5")
    metrics.plot_losses("./output/loss")
    if config.save_camo_to_png:
        mesh.set_camo(camo)
        mesh.make_texture_map_from_atlas("./output/tracker5")