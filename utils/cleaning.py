import torch


def is_garbage_sample(data, config, detector, rd, mesh):
    dist, elev, azim = data[4][0, :].float()
    background = data[1].to(config.device).to(torch.float32) / 255
    mask = data[2].to(config.device).to(torch.float32)

    rd.set_camera_position(dist, elev, azim)
    image_without_background = rd.render(mesh)
    image = image_without_background * mask + background * (1 - mask)

    result_background = detector.run(background, nms=True)
    result = detector.run(image, nms=True)

    if len(result) != 1 or len(result_background) != 1:
        return True
    if result[0][4] < config.clean_conf_threshold or result_background[0][4] < config.clean_conf_threshold:
        return True
    if len(data[3][0]) < 1:
        return True

    result = result[0]
    label = data[3][0].to(config.device).to(torch.float32)
    x1 = max(label[0], result[0])
    y1 = max(label[1], result[1])
    x2 = min(label[2], result[2])
    y2 = min(label[3], result[3])

    area_bbox1 = (label[2] - label[0] + 1) * (label[3] - label[1] + 1)
    area_bbox2 = (result[2] - result[0] + 1) * (result[3] - result[1] + 1)
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    union = area_bbox1 + area_bbox2 - intersection
    iou = intersection / union

    return iou < config.clean_iou_threshold
