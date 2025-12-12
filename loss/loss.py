import math

import numpy as np
import torch
from detector.neural_networks.track.OSTrack.lib.utils.box_ops import box_xywh_to_xyxy
from detector.neural_networks.track.OSTrack.lib.utils.heapmap_utils import generate_heatmap
from detector.neural_networks.track.OSTrack.lib.utils.focal_loss import FocalLoss
# from detector.neural_networks.track.OSTrack.lib.utils.box_ops import giou_loss
from torch.nn import L1Loss
class Loss:
    def __init__(self, config, model):
        # 初始化实例，获取配置文件，目标检测模型，目标检测模型的类别名，运行使用的计算设备
        self.__config = config
        self.__model = model
        self.__names = model.get_names()

    def iou_attack_loss(self, pred_bbox, target_bbox):
        """
        计算纯 IoU 损失，专门用于对抗攻击（Adversarial Attack）。
        目标是最小化此 Loss，从而使 IoU 趋向于 0（让跟踪器跟丢）。

        参数:
        - pred_bbox: (N, 4) 张量，预测框，格式为 (x, y, w, h)
        - target_bbox: (N, 4) 张量，目标框，格式为 (x, y, w, h)

        返回:
        - loss: IoU 的均值 (攻击越成功，此值越接近 0)
        - iou_val: 当前的 IoU 均值 (用于监控)
        """

        # 1. 坐标转换: (x, y, w, h) -> (x1, y1, x2, y2)
        # pred_bbox shape: [batch_size, 4]
        pred_x1 = pred_bbox[..., 0]
        pred_y1 = pred_bbox[..., 1]
        pred_x2 = pred_bbox[..., 0] + pred_bbox[..., 2]
        pred_y2 = pred_bbox[..., 1] + pred_bbox[..., 3]

        target_x1 = target_bbox[..., 0]
        target_y1 = target_bbox[..., 1]
        target_x2 = target_bbox[..., 0] + target_bbox[..., 2]
        target_y2 = target_bbox[..., 1] + target_bbox[..., 3]

        # 2. 计算相交区域 (Intersection)
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        # 这里的 clamp(min=0) 非常重要，如果没有重叠，宽或高为负数，乘积可能变正数，导致逻辑错误
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # 3. 计算并集区域 (Union)
        pred_area = pred_bbox[..., 2] * pred_bbox[..., 3]
        target_area = target_bbox[..., 2] * target_bbox[..., 3]

        # 防止分母为 0 添加 1e-6
        union_area = pred_area + target_area - inter_area + 1e-6

        # 4. 计算 IoU
        iou = inter_area / union_area

        # 5. 定义损失
        # -----------------------------------------------------------
        # 重要：因为你是做攻击（想让 IoU 变小），所以 Loss 直接等于 IoU。
        # 如果 Loss 趋向于 0，说明没有重叠，攻击成功。
        # -----------------------------------------------------------
        loss = iou

        # 进阶版（可选）：如果你发现 loss 下降太慢，可以使用平方，增加高 IoU 时的梯度
        # loss = iou ** 2

        return loss * 10

    def giou_loss(self, boxes1, boxes2):
        """
        计算 Generalized IoU (GIoU) 损失.

        参数:
        - boxes1: (1, 4) 张量，格式为 (xmin, ymin, w, h)
        - boxes2: (1, 4) 张量，格式为 (xmin, ymin, w, h)

        返回:
        - giou_loss: GIoU 损失，值越小表示两个框越相似。
        """

        # 将 (xmin, ymin, w, h) 转换为 (xmin, ymin, xmax, ymax)
        boxes1_x1y1x2y2 = torch.cat([boxes1[:, :2], boxes1[:, :2] + boxes1[:, 2:]], dim=-1)
        boxes2_x1y1x2y2 = torch.cat([boxes2[:, :2], boxes2[:, :2] + boxes2[:, 2:]], dim=-1)

        # 计算相交区域
        inter_x1 = torch.max(boxes1_x1y1x2y2[:, 0], boxes2_x1y1x2y2[:, 0])
        inter_y1 = torch.max(boxes1_x1y1x2y2[:, 1], boxes2_x1y1x2y2[:, 1])
        inter_x2 = torch.min(boxes1_x1y1x2y2[:, 2], boxes2_x1y1x2y2[:, 2])
        inter_y2 = torch.min(boxes1_x1y1x2y2[:, 3], boxes2_x1y1x2y2[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 计算每个框的面积
        area1 = (boxes1_x1y1x2y2[:, 2] - boxes1_x1y1x2y2[:, 0]) * (boxes1_x1y1x2y2[:, 3] - boxes1_x1y1x2y2[:, 1])
        area2 = (boxes2_x1y1x2y2[:, 2] - boxes2_x1y1x2y2[:, 0]) * (boxes2_x1y1x2y2[:, 3] - boxes2_x1y1x2y2[:, 1])

        # 计算并集区域
        union_area = area1 + area2 - inter_area

        # 计算 IoU
        iou = inter_area / union_area

        # 计算封闭区域的坐标
        enclose_x1 = torch.min(boxes1_x1y1x2y2[:, 0], boxes2_x1y1x2y2[:, 0])
        enclose_y1 = torch.min(boxes1_x1y1x2y2[:, 1], boxes2_x1y1x2y2[:, 1])
        enclose_x2 = torch.max(boxes1_x1y1x2y2[:, 2], boxes2_x1y1x2y2[:, 2])
        enclose_y2 = torch.max(boxes1_x1y1x2y2[:, 3], boxes2_x1y1x2y2[:, 3])

        # 计算封闭区域的面积
        enclose_area = torch.clamp(enclose_x2 - enclose_x1, min=0) * torch.clamp(enclose_y2 - enclose_y1, min=0)

        # 计算 GIoU
        giou = iou - (enclose_area - union_area) / enclose_area
        giou_loss = 1 - giou  # 损失越小表示两个框越接近

        return giou_loss.mean(), iou.mean()

    # 原来的里面的坐标都是0到1之间的，这里的都是真实的
    # giou_loss 是 0 ,focal_loss报错
    def ostrack_loss(self, pred_bbox, score_map, true_bbox):
        # pred_bbox (32,4) score_map (32,1,16,16)  true_bbox (32,4)
        # gt_dict['search_anno']  (1,32,4)
        # gt gaussian map  search_anno是啥？？正确的bbox？就是正确的bbox
        gt_bbox = true_bbox  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        # 这个函数的第一个输入的维度是(1,32,4)
        gt_gaussian_maps = generate_heatmap(true_bbox.unsqueeze(0), self.__config.search_size,
                                            self.__config.backbone_stride)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # 生成高斯热图，用于位置损失的计算。

        # Get boxes
        pred_boxes = pred_bbox  # 预测框
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_xywh_to_xyxy(pred_boxes).view(-1,
                                                             4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2) 将(x,y,w,h)转变为(x1,y1,x2,y2) 左上角和右下角的坐标
        # gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
        #                                                                                                    max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4)
        # compute giou and iou  带有vec的都是(x1,y1,x2,y2)
        try:
            giou_loss, iou = self.giou_loss(pred_boxes_vec, gt_boxes_vec)
            # giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4) 计算iou_loss 和 iou
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        # l1_loss = L1Loss(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4) 计算l1_loss
        l1_loss = torch.abs(pred_boxes_vec - gt_boxes_vec).mean()
        # compute location loss
        focal_loss = FocalLoss(alpha=2, beta=4)
        location_loss = focal_loss(score_map, gt_gaussian_maps)  # 计算focal_loss
        # weighted sum 总Loss是三个loss的加权求和
        loss = self.__config.loss.giou * giou_loss + self.__config.loss.l1 * l1_loss + self.__config.loss.focal * location_loss
        return loss

    def track_maximum_probability_score(self, result):
        result = result.view(1, 1, -1)
        result = torch.nn.Softmax(dim=2)(result)
        result = result.squeeze(1)
        max_conf, _ = torch.max(result, dim=1)
        maximum_probability_score_loss = max_conf
        # return maximum_probability_score_loss * self.__config.track_mps_weight
        return maximum_probability_score_loss * self.__config.track_mps_weight

    # 从干净搜索图像得到最大的k个和最小的k个得分的位置，然后抑制对抗搜索图像的大的k个位置的得分，增加小的k个位置的得分
    def track_top_k_min_k_probability_score(self, result, top_k_pos, min_k_pos):
        """
        根据提供的最大和最小位置列表，计算对应得分的平均值，并返回它们的差值。

        参数:
            result (torch.Tensor): 原始的模型输出张量，形状为 [1, 1, 16, 16]。
            top_k_pos (list): 包含最大 k 个元素位置的列表，每个位置是一个元组 (b, c, h, w)。
            min_k_pos (list): 包含最小 k 个元素位置的列表，每个位置是一个元组 (b, c, h, w)。

        返回:
            torch.Tensor: 一个标量张量，表示 (最大位置得分的平均值) - (最小位置得分的平均值)。
        """
        flattened_result = result.view(1, 1, -1)

        # 2. 提取 top_k_pos 对应的得分并计算平均值（使用张量操作保持梯度）
        if len(top_k_pos) > 0:
            # 构建索引张量
            indices = []
            for pos in top_k_pos:
                b, c, h, w = pos
                idx = h * result.shape[3] + w
                indices.append(idx)

            # 转换为张量并提取对应位置的值
            indices_tensor = torch.tensor(indices, device=result.device, dtype=torch.long)
            top_k_scores_tensor = flattened_result[0, 0, indices_tensor]
            top_k_mean = top_k_scores_tensor.mean()
        else:
            top_k_mean = torch.tensor(0.0, device=result.device)

        # 3. 提取 min_k_pos 对应的得分并计算平均值（使用张量操作保持梯度）
        if len(min_k_pos) > 0:
            # 构建索引张量
            indices = []
            for pos in min_k_pos:
                b, c, h, w = pos
                idx = h * result.shape[3] + w
                indices.append(idx)

            # 转换为张量并提取对应位置的值
            indices_tensor = torch.tensor(indices, device=result.device, dtype=torch.long)
            min_k_scores_tensor = flattened_result[0, 0, indices_tensor]
            min_k_mean = min_k_scores_tensor.mean()
        else:
            min_k_mean = torch.tensor(0.0, device=result.device)

        # 4. 计算并返回差值 (大的平均值 - 小的平均值)
        # 添加平滑项防止log(0)导致的数值不稳定
        epsilon = 1e-8
        # 确保min_k_mean大于0，避免log(0)
        min_k_mean = torch.clamp(min_k_mean, min=epsilon)
        score_difference = top_k_mean * 10 - torch.log(min_k_mean)

        # 使用平滑的clamp替代硬性的max操作，确保梯度连续
        final_score = torch.clamp(score_difference, min=-10.0)

        return final_score

    import torch

    def calculate_ciou_loss(self, pred_bbox, target_bbox, img_hw=None):
        """
        计算两个边界框之间的CIoU (Complete Intersection over Union)损失。

        参数:
            pred_bbox (torch.Tensor): 预测边界框，形状为(4,)或(batch_size, 4)，格式为[x, y, w, h]。
                                      x, y: 左上角坐标；w, h: 宽度和高度。
            target_bbox (torch.Tensor): 目标边界框（如真实框或干净模型预测框），
                                        形状和格式需与pred_bbox完全一致。
            img_hw (torch.Tensor, optional): 图像尺寸，形状为(2,)或(batch_size, 2)，格式为[img_h, img_w]，
                                             用于裁剪越界的预测框。若为None则不裁剪。

        返回:
            torch.Tensor: CIoU损失值。若输入为批量数据，返回形状为(batch_size,)的张量；
                         若为单对框，返回标量张量。
        """
        # 确保输入为二维张量 (batch_size, 4)
        if pred_bbox.dim() == 1:
            pred_bbox = pred_bbox.unsqueeze(0)
        if target_bbox.dim() == 1:
            target_bbox = target_bbox.unsqueeze(0)

        # --- 新增：将 [x, y, w, h] 转换为 [x1, y1, x2, y2] ---
        # 预测框转换
        pred_x1 = pred_bbox[..., 0]
        pred_y1 = pred_bbox[..., 1]
        pred_x2 = pred_bbox[..., 0] + pred_bbox[..., 2]
        pred_y2 = pred_bbox[..., 1] + pred_bbox[..., 3]
        pred_bbox_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

        # 目标框转换
        target_x1 = target_bbox[..., 0]
        target_y1 = target_bbox[..., 1]
        target_x2 = target_bbox[..., 0] + target_bbox[..., 2]
        target_y2 = target_bbox[..., 1] + target_bbox[..., 3]
        target_bbox_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=-1)
        # --- 转换结束 ---

        # 裁剪预测框至图像范围内（可选）
        if img_hw is not None:
            if img_hw.dim() == 1:
                img_hw = img_hw.unsqueeze(0)
            img_w = img_hw[..., 1]
            img_h = img_hw[..., 0]
            # 逐元素裁剪，避免越界
            pred_bbox_xyxy[..., 0] = pred_bbox_xyxy[..., 0].clamp(0, img_w - 1)
            pred_bbox_xyxy[..., 1] = pred_bbox_xyxy[..., 1].clamp(0, img_h - 1)
            pred_bbox_xyxy[..., 2] = pred_bbox_xyxy[..., 2].clamp(0, img_w - 1)
            pred_bbox_xyxy[..., 3] = pred_bbox_xyxy[..., 3].clamp(0, img_h - 1)

        # 1. 计算交集区域
        inter_x1 = torch.max(pred_bbox_xyxy[..., 0], target_bbox_xyxy[..., 0])
        inter_y1 = torch.max(pred_bbox_xyxy[..., 1], target_bbox_xyxy[..., 1])
        inter_x2 = torch.min(pred_bbox_xyxy[..., 2], target_bbox_xyxy[..., 2])
        inter_y2 = torch.min(pred_bbox_xyxy[..., 3], target_bbox_xyxy[..., 3])

        inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1 + 1, min=0)

        # 2. 计算两个框的面积
        pred_area = (pred_bbox_xyxy[..., 2] - pred_bbox_xyxy[..., 0] + 1) * \
                    (pred_bbox_xyxy[..., 3] - pred_bbox_xyxy[..., 1] + 1)
        target_area = (target_bbox_xyxy[..., 2] - target_bbox_xyxy[..., 0] + 1) * \
                      (target_bbox_xyxy[..., 3] - target_bbox_xyxy[..., 1] + 1)

        # 3. 计算IoU (Intersection over Union)
        iou = inter_area / (pred_area + target_area - inter_area + 1e-6)

        # 4. 计算中心点距离的平方 (ρ²)
        pred_cx = (pred_bbox_xyxy[..., 0] + pred_bbox_xyxy[..., 2]) / 2
        pred_cy = (pred_bbox_xyxy[..., 1] + pred_bbox_xyxy[..., 3]) / 2
        target_cx = (target_bbox_xyxy[..., 0] + target_bbox_xyxy[..., 2]) / 2
        target_cy = (target_bbox_xyxy[..., 1] + target_bbox_xyxy[..., 3]) / 2

        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 5. 计算最小包围框对角线长度的平方 (c²)
        enclose_x1 = torch.min(pred_bbox_xyxy[..., 0], target_bbox_xyxy[..., 0])
        enclose_y1 = torch.min(pred_bbox_xyxy[..., 1], target_bbox_xyxy[..., 1])
        enclose_x2 = torch.max(pred_bbox_xyxy[..., 2], target_bbox_xyxy[..., 2])
        enclose_y2 = torch.max(pred_bbox_xyxy[..., 3], target_bbox_xyxy[..., 3])

        c2 = (enclose_x2 - enclose_x1 + 1) ** 2 + (enclose_y2 - enclose_y1 + 1) ** 2

        # 6. 计算长宽比一致性因子 (v)
        pred_w = pred_bbox_xyxy[..., 2] - pred_bbox_xyxy[..., 0] + 1
        pred_h = pred_bbox_xyxy[..., 3] - pred_bbox_xyxy[..., 1] + 1
        target_w = target_bbox_xyxy[..., 2] - target_bbox_xyxy[..., 0] + 1
        target_h = target_bbox_xyxy[..., 3] - target_bbox_xyxy[..., 1] + 1

        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + 1e-6)) - torch.atan(pred_w / (pred_h + 1e-6)), 2
        )

        # 7. 计算平衡因子 (α)
        alpha = v / ((1 - iou) + v + 1e-6)

        # 8. 计算CIoU并得到损失 (1 - CIoU)
        ciou = iou - (rho2 / (c2 + 1e-6)) - alpha * v
        loss = ciou * 10

        # 若输入为单样本，返回标量张量
        if loss.numel() == 1:
            loss = loss.squeeze()

        return loss

    def track_top_k_probability_score(self, result):
        result = result.view(1, 1, -1)
        result = torch.nn.Softmax(dim=2)(result)
        result = result.squeeze(1)
        # 获取前k个最大值
        topk_conf, _ = torch.topk(result, 30, dim=1)
        # 平均前k个值
        topk_probability_score_loss = topk_conf.mean(dim=1)
        return topk_probability_score_loss * self.__config.track_mps_weight

    def track_score_loss(self, score):
        # label = torch.full_like(score, -1)
        # max_pos = torch.argmax(score)
        # label.view(-1)[max_pos] = 1

        # 将前k个值作为正样本区域
        # 将score展平成1维张量，保留原始形状
        flattened_score = score.view(-1)

        # 创建与score形状相同的label，初始值为-1
        label = torch.full_like(flattened_score, -1)

        # 找到展平后的score中得分最高的前k个值的位置
        topk_pos_indices = torch.topk(flattened_score, 36).indices

        # 将得分最高的前k个值的位置对应的标签设为1
        label[topk_pos_indices] = 1

        # 重新调整label为与原score相同的形状 (1, 1, 16, 16)
        label = label.view_as(score)

        # 将得分最高的前k个值作为正样本区域
        label.view(-1)[topk_pos_indices] = 1

        # l_y_s = torch.log(1 + torch.exp(-label * score))
        # 改成了里面是score，把-label删了
        l_y_s = torch.log(1 + torch.exp(score))
        # 找到正标签区域的最小损失
        # min_loss_positive = torch.min(l_y_s[label == 1])
        # 改为正样本的最大
        min_loss_positive = torch.max(l_y_s[label == 1])

        # 找到负标签区域的最大损失
        max_loss_negative = torch.max(l_y_s[label == -1])
        # 改为负样本的最小
        # max_loss_negative = torch.min(l_y_s[label == -1])

        # 计算分数损失
        L_score = min_loss_positive - max_loss_negative
        return L_score

    def track_distance_loss(self, score, beta1=1.0, delta=1e-6, xi=0.1):
        # # score 是 (1, 1, 16, 16) 的张量，表示每一块的得分
        # score = score.squeeze()  # 去掉 batch 和 channel 维度，变为 (16, 16)
        #
        # # 获取中心区域最大得分及其索引
        # max_score_center = score.max()
        # index_max_center = (score == max_score_center).nonzero(as_tuple=True)
        #
        # # 获取非中心区域最大得分及其索引
        # non_center_area = score != max_score_center
        # max_score_non_center = (score * non_center_area).max()
        # # 改为非中心区域的最小，也就是所有区域的最小，所以不用找到非中心区域
        # # max_score_non_center = score.min()
        # index_max_non_center = (score == max_score_non_center).nonzero(as_tuple=True)
        #
        # # 将索引转换为二维坐标
        # center_coords = (index_max_center[0].item(), index_max_center[1].item())
        # non_center_coords = (index_max_non_center[0].item(), index_max_non_center[1].item())

        # 获取前k个值作为正样本区域
        # score 是 (1, 1, 16, 16) 的张量，表示每一块的得分
        score = score.squeeze()  # 去掉 batch 和 channel 维度，变为 (16, 16)

        # 将 score 展平为一维向量，方便取前k个值
        flattened_score = score.view(-1)

        # 获取前k个最大值及其索引 (正样本区域)
        topk_values, topk_indices = torch.topk(flattened_score, k=36)

        # 获取正样本区域的最大值及其索引（我们只取这一个正样本最大值的坐标）
        max_value_pos = topk_values[0]
        index_max_pos = topk_indices[0]

        # 将正样本最大值的索引转换为二维坐标
        center_coords = torch.tensor((index_max_pos // score.shape[0],index_max_pos % score.shape[1]), dtype=torch.float)

        # 获取非正样本区域（负样本区域）的最小值
        mask = torch.ones_like(flattened_score, dtype=torch.bool)
        mask[topk_indices] = False  # 将前k个最大值的索引设为False，剩下的为负样本区域

        # 获取负样本区域的最大值及其索引
        max_value_non_center = flattened_score[mask].max()
        index_max_non_center = (flattened_score == max_value_non_center).nonzero(as_tuple=True)[0].item()

        # 将负样本的最小值索引转换为二维坐标
        non_center_coords = torch.tensor((index_max_non_center // score.shape[0],index_max_non_center % score.shape[1]), dtype=torch.float)

        # 计算欧几里得距离
        distance = torch.norm(
            torch.tensor(center_coords, dtype=torch.float) - torch.tensor(non_center_coords, dtype=torch.float))

        # 计算距离损失
        L_dist = (beta1 / (delta + distance)) - xi

        return L_dist
    # 最大识别概率损失
    def maximum_probability_score(self, result):
        # result = torch.Size([1, 80055, 85])
        # 获取需要攻击的类别
        classes = self.__config.train_classes
        # 确保传入的目标识别网络的计算结果符合[x,y,w,h,存在物体的概率，每个类别的概率(coco数据集共有80种类别)]
        assert (result.size(-1) == (5 + len(self.__names)))
        # 将锚框中每个类别的概率，单独截断出来
        class_confs = result[:, :, 5:5 + len(self.__names)]

        if classes is not None:
            # 若制定了攻击类别，则使用softmax函数，对所有类别的概率进行归一化，并获取指定攻击类别在全部识别概率中的占比。
            class_confs = torch.nn.Softmax(dim=2)(class_confs)
            class_confs = class_confs[:, :, classes]
        else:
            # 若未指定攻击类别，则直接获取识别结果中，，进行攻击
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]

        # 获取识别结果的第四项，即存在物体概率
        objectiveness_score = result[:, :, 4]
        # 保证两者维度相同，（batch-size为1时，会导致两者维度不同，objectiveness_score的batch-size维度缺失）
        if class_confs.shape[-1] != objectiveness_score.shape[-1]:
            objectiveness_score = objectiveness_score.unsqueeze(-1)

        # 若仅攻击物体存在概率，则结果只返回物体存在概率，否则返回存在概率和分类概率的乘积
        if self.__config.only_objectness:
            confs_if_object = objectiveness_score
        else:
            confs_if_object = objectiveness_score * class_confs

        # 最后返回所有锚框中 最大的概率得分的锚框的概率得分
        max_conf, _ = torch.max(confs_if_object, dim=1)
        maximum_probability_score_loss = max_conf
        return maximum_probability_score_loss * self.__config.mps_weight

    # 平滑函数

    def total_variation(self, image, mask):
        # 计算patch第二维即宽上相邻两像素之间的差值，使用绝对值函数保证正数，并求和,得到一个一维张量
        # 每个值对应列上元素差值，并将每一列的差值求和得到宽上所有列差值的总和
        variation_w = torch.sum(torch.abs(image[:, 1:, :] - image[:, :-1, :] + 0.000001), 0)
        variation_w = torch.sum(torch.sum(variation_w, 0), 0)
        # 计算patch第一维即高上相邻两像素之间的差值，使用绝对值函数保证正数，并求和，得到一个一维张量
        # 每个值对应行上元素差值，并将每一行的差值求和得到高上所有行差值的总和
        variation_h = torch.sum(torch.abs(image[1:, :, :] - image[:-1, :, :] + 0.000001), 0)
        variation_h = torch.sum(torch.sum(variation_h, 0), 0)
        variation = variation_w + variation_h
        num = image.size(0) * image.size(1) - np.count_nonzero(mask)
        total_variation_loss = variation / num
        return torch.max(total_variation_loss * self.__config.tv_weight, torch.tensor(0.1, device=self.__config.device))
