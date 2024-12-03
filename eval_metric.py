import os
from datetime import datetime
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from vit_pytorch.simple_vit import SimpleViT  # 假设你使用的是 SimpleViT
import numpy as np
from dataset_loader import SimpleVitDataset  # 你的自定义数据集
from tqdm import tqdm

"""
This script is used to evaluate the network using ViT.

Please note, class 1 is for wearing mask.
"""

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./dataset')
parser.add_argument('--ckpt-path', type=str, default='saved_ckpt')
parser.add_argument('--cuda', action='store_true', help='If use CUDA')

# 计算多分类的指标
def cal_metric_multi_class(conf_matrix, c=1):
    TP = conf_matrix.diag()[c]
    idx = torch.ones(2).bool()
    idx[1] = 0
    TN = conf_matrix[torch.nonzero(idx, as_tuple=False)[:, None], torch.nonzero(idx, as_tuple=False)].sum()
    FP = conf_matrix[idx, c].sum()
    FN = conf_matrix[c, idx].sum()
    return TP.item(), TN.item(), FP.item(), FN.item()

# 计算二分类的指标
def cal_metric(conf_matrix):
    TP = conf_matrix[1, 1].item()
    TN = conf_matrix[0, 0].item()
    FP = conf_matrix[0, 1].item()
    FN = conf_matrix[1, 0].item()
    return TP, TN, FP, FN

def cal_specificity(TN, FP):
    return TN / (TN + FP) if (TN + FP) > 0 else 0

def cal_sensitivity(TP, FN):
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def cal_accuracy(TP, TN, num_samples):
    return (TP + TN) / num_samples if num_samples > 0 else 0

def cal_cohen_kappa(TP, TN, FP, FN):
    num_samples = TP + TN + FP + FN
    p_o = (TP + TN) / num_samples if num_samples > 0 else 0
    p_neg = ((TN + FP) / num_samples) * ((TN + FN) / num_samples) if num_samples > 0 else 0
    p_pos = ((FN + TP) / num_samples) * ((FP + TP) / num_samples) if num_samples > 0 else 0
    p_e = p_neg + p_pos
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0

if __name__ == '__main__':
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # 创建并加载模型
    model = SimpleViT(
        image_size=256,
        patch_size=32,
        num_classes=2,  # 适配你的数据集类别数
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048
    )
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'model.pt'), map_location=device))  # 加载已保存的权重
    model.to(device)  # 将模型转移到设备上

    # 加载测试数据集
    test_set = SimpleVitDataset(args.data_path, os.path.join(args.data_path, 'test.txt'), image_size=(256, 256), mode='test')
    test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False, num_workers=4)

    # 开始评估
    model.eval()
    conf_matrix = torch.zeros(2, 2)  # 初始化混淆矩阵 (2x2 对于二分类)

    for batch_idx, (batch_data, batch_target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)

        with torch.no_grad():
            batch_pred = model(batch_data)
            batch_pred = torch.argmax(batch_pred, dim=1)  # 获取预测标签

            # 更新混淆矩阵
            for t, p in zip(batch_target, batch_pred):
                conf_matrix[t.long(), p.long()] += 1

    # 计算指标
    TP, TN, FP, FN = cal_metric(conf_matrix)
    specificity = cal_specificity(TN, FP)
    sensitivity = cal_sensitivity(TP, FN)
    accuracy = cal_accuracy(TP, TN, TP + TN + FP + FN)
    cohen_kappa = cal_cohen_kappa(TP, TN, FP, FN)

    # 打印计算指标
    print('[{}] Specificity: {:.3f}'.format(datetime.now(), specificity))
    print('[{}] Sensitivity: {:.3f}'.format(datetime.now(), sensitivity))
    print('[{}] Accuracy: {:.3f}'.format(datetime.now(), accuracy))
    print('[{}] Cohen\'s Kappa: {:.3f}'.format(datetime.now(), cohen_kappa))
