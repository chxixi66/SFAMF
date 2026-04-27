#!/usr/bin/env python3
"""
评估模型检测结果，对比检测输出和数据集标签
找出漏检和错检的情况
"""

import torch
import cv2
import numpy as np
import os
from pathlib import Path
from models.yolov5_object_detector import YOLOV5TorchObjectDetector
from utils.general import non_max_suppression


def load_gt_labels(label_path):
    """
    加载 Ground Truth 标签
    
    Args:
        label_path: 标签文件路径
    
    Returns:
        gt_boxes: 标签框列表 [(class, x1, y1, x2, y2), ...]
    """
    if not Path(label_path).exists():
        return []
    
    gt = np.loadtxt(label_path)
    if gt.size == 0:
        return []
    
    gt = gt.reshape(-1, 5)
    gt_boxes = []
    
    for row in gt:
        cls = int(row[0])
        x_center, y_center, w, h = row[1:5]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        gt_boxes.append((cls, x1, y1, x2, y2))
    
    return gt_boxes


def load_detection_results(pred, conf_thres=0.5):
    """
    从检测结果中提取检测框
    
    Args:
        pred: 检测结果
        conf_thres: 置信度阈值
    
    Returns:
        det_boxes: 检测框列表 [(class, x1, y1, x2, y2, conf), ...]
    """
    pred_nms = non_max_suppression(pred[0], conf_thres=conf_thres, iou_thres=0.5)
    det_boxes = []
    
    for detection in pred_nms[0]:
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        det_boxes.append((int(cls), float(x1), float(y1), float(x2), float(y2), float(conf)))
    
    return det_boxes


def calculate_iou(box1, box2):
    """
    计算两个框的 IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        iou: IoU 值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0
    
    return intersection / union


def match_detections(gt_boxes, det_boxes, iou_thres=0.5):
    """
    匹配检测结果和标签
    
    Args:
        gt_boxes: Ground Truth 标签框列表
        det_boxes: 检测框列表
        iou_thres: IoU 阈值
    
    Returns:
        matches: 匹配信息字典
    """
    matches = {
        'tp': [],  # True Positives: 正确检测
        'fp': [],  # False Positives: 误检
        'fn': []   # False Negatives: 漏检
    }
    
    gt_matched = [False] * len(gt_boxes)
    det_matched = [False] * len(det_boxes)
    
    # 计算所有框之间的 IoU
    iou_matrix = np.zeros((len(gt_boxes), len(det_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, det in enumerate(det_boxes):
            gt_box = [gt[1], gt[2], gt[3], gt[4]]  # x1, y1, x2, y2
            det_box = [det[1], det[2], det[3], det[4]]  # x1, y1, x2, y2
            iou_matrix[i, j] = calculate_iou(gt_box, det_box)
    
    # 匹配检测结果
    for i in range(len(gt_boxes)):
        for j in range(len(det_boxes)):
            if iou_matrix[i, j] >= iou_thres and not gt_matched[i] and not det_matched[j]:
                # 检查类别是否匹配
                if gt_boxes[i][0] == det_boxes[j][0]:
                    matches['tp'].append({
                        'gt_idx': i,
                        'det_idx': j,
                        'iou': iou_matrix[i, j],
                        'class': gt_boxes[i][0]
                    })
                    gt_matched[i] = True
                    det_matched[j] = True
                else:
                    # 类别不匹配，FP
                    matches['fp'].append({
                        'det_idx': j,
                        'iou': iou_matrix[i, j],
                        'gt_class': gt_boxes[i][0],
                        'det_class': det_boxes[j][0]
                    })
                    det_matched[j] = True
    
    # 剩余未匹配的检测是 FP
    for j in range(len(det_boxes)):
        if not det_matched[j]:
            matches['fp'].append({
                'det_idx': j,
                'class': det_boxes[j][0],
                'conf': det_boxes[j][5]
            })
    
    # 剩余未匹配的标签是 FN
    for i in range(len(gt_boxes)):
        if not gt_matched[i]:
            matches['fn'].append({
                'gt_idx': i,
                'class': gt_boxes[i][0]
            })
    
    return matches


def evaluate_dataset(model, rgb_dir, ir_dir, label_dir, image_list, conf_thres=0.5):
    """
    评估整个数据集
    
    Args:
        model: 模型
        rgb_dir: RGB 图像目录
        ir_dir: IR 图像目录
        label_dir: 标签目录
        image_list: 图像文件名列表
        conf_thres: 置信度阈值
    
    Returns:
        results: 评估结果
    """
    results = {
        'total_images': 0,
        'total_gt': 0,
        'total_det': 0,
        'total_tp': 0,
        'total_fp': 0,
        'total_fn': 0,
        'image_results': [],  # 每张图像的详细结果
        '漏检图像': [],  # 漏检图像索引
        '错检图像': []   # 错检图像索引
    }
    
    for idx, image_name in enumerate(image_list):
        rgb_path = os.path.join(rgb_dir, image_name)
        ir_path = os.path.join(ir_dir, image_name)
        label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_name)
        
        # 加载图像
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        
        if img_rgb is None:
            continue
        
        # 预处理
        torch_img_rgb = model.preprocessing(img_rgb[..., ::-1])
        torch_img_ir = model.preprocessing(img_ir[..., ::-1])
        
        # 前向传播
        with torch.no_grad():
            output = model.model(torch_img_rgb, torch_img_ir)
        
        # 加载 Ground Truth
        gt_boxes = load_gt_labels(label_path)
        
        # 加载检测结果
        det_boxes = load_detection_results(output, conf_thres)
        
        # 匹配检测结果
        matches = match_detections(gt_boxes, det_boxes, iou_thres=0.5)
        
        # 统计
        results['total_images'] += 1
        results['total_gt'] += len(gt_boxes)
        results['total_det'] += len(det_boxes)
        results['total_tp'] += len(matches['tp'])
        results['total_fp'] += len(matches['fp'])
        results['total_fn'] += len(matches['fn'])
        
        # 记录图像结果
        image_result = {
            'index': idx,
            'image_name': image_name,
            'gt_count': len(gt_boxes),
            'det_count': len(det_boxes),
            'tp_count': len(matches['tp']),
            'fp_count': len(matches['fp']),
            'fn_count': len(matches['fn']),
            'matches': matches
        }
        results['image_results'].append(image_result)
        
        # 记录漏检和错检图像
        if len(matches['fn']) > 0:
            results['漏检图像'].append(idx)
        
        if len(matches['fp']) > 0:
            results['错检图像'].append(idx)
    
    return results


def print_results(results):
    """
    打印评估结果
    """
    print("\n" + "="*60)
    print("检测结果评估报告")
    print("="*60)
    
    # 总体统计
    print("\n【总体统计】")
    print(f"总图像数: {results['total_images']}")
    print(f"总 Ground Truth 数: {results['total_gt']}")
    print(f"总检测数: {results['total_det']}")
    print(f"True Positives (TP): {results['total_tp']}")
    print(f"False Positives (FP): {results['total_fp']}")
    print(f"False Negatives (FN): {results['total_fn']}")
    
    # 计算指标
    precision = results['total_tp'] / (results['total_tp'] + results['total_fp'] + 1e-6)
    recall = results['total_tp'] / (results['total_tp'] + results['total_fn'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    print(f"\n【检测指标】")
    print(f"Precision (精确率): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 漏检和错检统计
    print(f"\n【漏检和错检统计】")
    print(f"漏检图像数量: {len(results['漏检图像'])}")
    print(f"错检图像数量: {len(results['错检图像'])}")
    
    # 漏检图像详情
    if results['漏检图像']:
        print(f"\n【漏检图像索引】")
        for idx in results['漏检图像']:
            result = results['image_results'][idx]
            print(f"  图像 {idx} ({result['image_name']}): "
                  f"GT={result['gt_count']}, Det={result['det_count']}, "
                  f"漏检={result['fn_count']}")
    
    # 错检图像详情
    if results['错检图像']:
        print(f"\n【错检图像索引】")
        for idx in results['错检图像']:
            result = results['image_results'][idx]
            print(f"  图像 {idx} ({result['image_name']}): "
                  f"GT={result['gt_count']}, Det={result['det_count']}, "
                  f"错检={result['fp_count']}")
    
    # 详细匹配信息（可选）
    print(f"\n【详细匹配信息】")
    for result in results['image_results']:
        if result['fn_count'] > 0 or result['fp_count'] > 0:
            print(f"\n  图像 {result['index']} ({result['image_name']}):")
            
            if result['fn_count'] > 0:
                print(f"    漏检 ({result['fn_count']}):")
                for fn in result['matches']['fn']:
                    print(f"      - 类别 {fn['class']} (GT index: {fn['gt_idx']})")
            
            if result['fp_count'] > 0:
                print(f"    错检 ({result['fp_count']}):")
                for fp in result['matches']['fp']:
                    if 'class' in fp:
                        print(f"      - 类别 {fp['class']} (conf: {fp.get('conf', 'N/A'):.2f})")
                    else:
                        print(f"      - GT 类别 {fp['gt_class']}, 检测类别 {fp['det_class']} (IoU: {fp['iou']:.2f})")


def main():
    # 模型路径
    model_path = "./runs/train/wo_SFMFB/weights/best.pt"
    
    # 数据集路径
    dataset_root = "./data/datasets/DroneVehicle"
    rgb_dir = os.path.join(dataset_root, "rgb", "images", "test")
    ir_dir = os.path.join(dataset_root, "ir", "images", "test")
    label_dir = os.path.join(dataset_root, "rgb", "labels", "test")
    
    # 加载图像列表
    image_list = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"[INFO] Loading model from {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOV5TorchObjectDetector(model_path, device, img_size=(640, 640), 
                                      names=['car', 'truck', 'bus', 'van', 'freight_car'])
    
    print(f"[INFO] Evaluating {len(image_list)} images...")
    
    # 评估数据集
    results = evaluate_dataset(model, rgb_dir, ir_dir, label_dir, image_list, conf_thres=0.5)
    
    # 打印结果
    print_results(results)
    
    # 保存结果到文件
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: evaluation_results.json")
    
    # 打印漏检和错检图像索引列表
    print(f"\n【漏检图像索引列表】")
    print(results['漏检图像'])
    
    print(f"\n【错检图像索引列表】")
    print(results['错检图像'])


if __name__ == "__main__":
    main()
