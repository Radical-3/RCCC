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

    def giou_loss(boxes1, boxes2):
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
