import matplotlib.pyplot as plt
import os
import numpy as np
import csv

# 全局变量存储损失数据
# 数据结构: {dataset_name: {"epochs": [...], "total_losses": [...], "score_losses": [...], "ciou_losses": [...]}}
datasets_losses = {}


def init_metrics(save_dir):
    """
    初始化指标存储目录
    
    Args:
        save_dir: 保存目录路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def update_losses(dataset_name, epoch, total_loss, score_loss, iou_loss):
    """
    更新特定数据集的损失值
    
    Args:
        dataset_name: 数据集名称
        epoch: 当前epoch
        total_loss: 总损失
        score_loss: 得分损失
        ciou_loss: IoU损失
    """
    if dataset_name not in datasets_losses:
        datasets_losses[dataset_name] = {
            "epochs": [],
            "total_losses": [],
            "score_losses": [],
            "iou_losses": []
        }
    
    datasets_losses[dataset_name]["epochs"].append(epoch)
    datasets_losses[dataset_name]["total_losses"].append(total_loss)
    datasets_losses[dataset_name]["score_losses"].append(score_loss)
    datasets_losses[dataset_name]["iou_losses"].append(iou_loss)


def plot_losses(save_dir="./output"):
    """
    为三项损失分别绘制图表，每张图表包含所有数据集的损失曲线
    
    Args:
        save_dir: 图表保存目录
    """
    if len(datasets_losses) == 0:
        return
    
    # 获取所有唯一的epoch值
    all_epochs = set()
    for dataset_data in datasets_losses.values():
        all_epochs.update(dataset_data["epochs"])
    all_epochs = sorted(list(all_epochs))
    
    # 绘制总损失图表
    plt.figure(figsize=(10, 6))
    for dataset_name, dataset_data in datasets_losses.items():
        plt.plot(dataset_data["epochs"], dataset_data["total_losses"], marker='o', label=dataset_name)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss vs Epoch (All Datasets)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'total_losses.png'))
    plt.close()
    
    # 绘制得分损失图表
    plt.figure(figsize=(10, 6))
    for dataset_name, dataset_data in datasets_losses.items():
        plt.plot(dataset_data["epochs"], dataset_data["score_losses"], marker='s', label=dataset_name)
    plt.xlabel('Epoch')
    plt.ylabel('Score Loss')
    plt.title('Score Loss vs Epoch (All Datasets)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'score_losses.png'))
    plt.close()
    
    # 绘制CIoU损失图表
    plt.figure(figsize=(10, 6))
    for dataset_name, dataset_data in datasets_losses.items():
        plt.plot(dataset_data["epochs"], dataset_data["iou_losses"], marker='^', label=dataset_name)
    plt.xlabel('Epoch')
    plt.ylabel('IoU Loss')
    plt.title('IoU Loss vs Epoch (All Datasets)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_losses.png'))
    plt.close()


def save_losses_to_csv(save_dir="./output"):
    """
    将所有数据集的损失值保存为CSV文件
    
    Args:
        save_dir: CSV文件保存目录
    """
    if len(datasets_losses) == 0:
        return
    
    # 为每种损失创建一个CSV文件
    for loss_type in ["total_losses", "score_losses", "iou_losses"]:
        csv_path = os.path.join(save_dir, f'{loss_type}.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            header = ["Epoch"] + list(datasets_losses.keys())
            writer.writerow(header)
            
            # 获取所有唯一的epoch值
            all_epochs = set()
            for dataset_data in datasets_losses.values():
                all_epochs.update(dataset_data["epochs"])
            all_epochs = sorted(list(all_epochs))
            
            # 为每个epoch写入各数据集的损失值
            for epoch in all_epochs:
                row = [epoch]
                for dataset_name in datasets_losses.keys():
                    dataset_data = datasets_losses[dataset_name]
                    if epoch in dataset_data["epochs"]:
                        # 找到该epoch对应的损失值索引
                        epoch_index = dataset_data["epochs"].index(epoch)
                        row.append(dataset_data[loss_type][epoch_index])
                    else:
                        row.append("")  # 如果该数据集没有这个epoch的数据，则留空
                writer.writerow(row)