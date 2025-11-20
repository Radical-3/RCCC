
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


# 实验是不是因为搜索图像和模板图像都是有camo参与，所以导致没有梯度
def track3():
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

    for epoch in range(config.epochs):
        total_loss = list()
        for seq in dataset:
            for i in range(0, len(seq.frames), 2):
                mesh.set_camo(camo)
                data_np_temp = numpy.load(seq.frames[i], allow_pickle=True)
                data_temp = [torch.tensor(item) for item in data_np_temp]
                dist, elev, azim = data_temp[4].float()
                background_temp = data_temp[1].to(config.device).to(torch.float32) / 255

                # image = convert_to_numpy(image)

                # cv2.imshow('z', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # init_info = seq.my_init_info(i)
                init_info = {'init_bbox': seq.ground_truth_rect[i]}
                ostracker.my_initialize(background_temp, init_info)


                data_np = numpy.load(seq.frames[i+1], allow_pickle=True)
                data = [torch.tensor(item) for item in data_np]
                dist, elev, azim = data[4].float()
                background = data[1].to(config.device).to(torch.float32) / 255
                mask = data[2].to(config.device).to(torch.float32)

                renderer.set_camera_position(dist, elev, azim)
                image_without_background = renderer.render(mesh.item())
                image_backup = image_without_background.clone().to(config.device)

                image_without_background = transform(config, image_without_background)
                # x = convert_to_numpy(image_without_background)
                # cv2.imshow('x', x)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                image = image_without_background * mask + background * (1 - mask)
                image = image.squeeze(0)
                # image = convert_to_numpy(image)

                result, bbox = ostracker.my_track(image)
                # 将浮点数转换为整数，因为绘制图像时需要整数像素值
                x, y, w, h = map(int, bbox)
                # # 使用 cv2.rectangle 在图像上绘制矩形框，参数分别是图像，左上角坐标，右下角坐标，颜色和线条宽度
                image_show = convert_to_numpy(image.unsqueeze(0))
                cv2.rectangle(image_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('x', image_show)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                loss_maximum_probability_score = loss.track_maximum_probability_score(result)
                # loss_maximum_probability_score = loss.track_top_k_probability_score(result)
                # 改成了得分损失+距离损失
                # track_loss_score = loss.track_score_loss(result) * 10
                # track_loss_distance = loss.track_distance_loss(result)
                # loss_track = track_loss_score + track_loss_distance
                loss_total_variation = loss.total_variation(image_backup.squeeze(), data[2])

                # loss_value = loss_track + loss_total_variation
                loss_value = loss_maximum_probability_score + loss_total_variation
                optimizer.zero_grad()
                # retain_graph=True加上这个，解决了两次传播的问题，但是不知道对结果有没有影响
                # 额，发现是每一次没有重新设置camo
                loss_value.backward()
                # loss_value.backward(retain_graph=True)
                total_loss.append(loss_value.item())
                print(torch.all(camo.item().grad == 0))
                optimizer.step()
                camo.clamp()

                # pbar.set_postfix(total_loss=f"{np.mean(total_loss):.3f}", loss=f"{loss_value.item():.3f}", score_loss=f"{track_loss_score.item():.3f}")
                # print(frame_num)

    if config.save_camo_to_pth:
        camo.save_camo_pth("./output")

    if config.save_camo_to_png:
        mesh.set_camo(camo)
        mesh.make_texture_map_from_atlas("./output")
