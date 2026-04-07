import os
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import numpy as np
import torch
import json

# NYU40 数据集类别字典
nyu40_dict = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels


def calculate_metrics(gt, pred, total_classes):
    gt = gt.cpu()
    pred = pred.cpu()
    pred[gt == 0] = 0

    ious = torch.zeros(total_classes)
    intersection = torch.zeros(total_classes)
    union = torch.zeros(total_classes)
    correct = torch.zeros(total_classes)
    total = torch.zeros(total_classes)

    for cls in range(1, total_classes):
        intersection[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        union[cls] = torch.sum((gt == cls) | (pred == cls)).item()
        correct[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        total[cls] = torch.sum(gt == cls).item()

    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    gt_classes = torch.unique(gt)
    valid_gt_classes = gt_classes[gt_classes != 0]

    # mIoU
    mean_iou = ious[valid_gt_classes].mean().item()

    # Accuracy
    valid_mask = gt != 0
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else 0.0

    # mAcc
    class_accuracy = torch.zeros(total_classes)
    mask = total > 0
    class_accuracy[mask] = correct[mask] / total[mask]
    mean_class_accuracy = class_accuracy[valid_gt_classes].mean().item()

    return ious, mean_iou, accuracy, mean_class_accuracy


if __name__ == "__main__":
    scene_list = ['scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00', 'scene0140_00',
                  'scene0200_00', 'scene0347_00', 'scene0400_00', 'scene0590_00', 'scene0645_00']

    iteration = 90000

    # --- 初始化结果存储字典 ---
    final_results = {
        "config": {"iteration": iteration, "task": "ScanNet Evaluation"},
        "scenes": {},
        "overall_average": {}
    }

    all_mious = []
    all_maccs = []

    for scan_name in scene_list:
        print(f"==> Processing {scan_name}...")

        # (1) 加载真值数据
        gt_file_path = f"E:\\opengaussians\\OpenGaussian-main\\data\\scannet\\{scan_name}/{scan_name}_vh_clean_2.labels.ply"
        if not os.path.exists(gt_file_path):
            print(f"Skipping {scan_name}, GT file not found.")
            continue
        points, labels = read_labels_from_ply(gt_file_path)

        # (2) 定义目标类别 (10类示例)
        target_id = [1, 2, 4, 5, 6, 7, 8, 9, 10, 33]
        target_dict = {key: nyu40_dict[key] for key in target_id}
        target_names = list(target_dict.values())

        # (3) 更新真值标签映射
        target_id_mapping = {value: index + 1 for index, value in enumerate(target_id)}
        updated_labels = np.zeros_like(labels)
        for original_value, new_value in target_id_mapping.items():
            updated_labels[labels == original_value] = new_value
        updated_gt_labels = torch.from_numpy(updated_labels.astype(np.int64)).cuda()

        # (4) 加载 Gaussian 点云并过滤低透明度点
        model_path = f"E:\\opengaussians\\OpenGaussian-main\\scripts\\output\\scannet/{scan_name}/"
        ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex'].data
        ignored_pts = sigmoid(vertex_data["opacity"]) < 0.1
        updated_gt_labels[ignored_pts] = 0

        # (5) 加载聚类语言特征
        mapping_file = os.path.join(model_path, "cluster_lang.npz")
        saved_data = np.load(mapping_file)
        leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()
        leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()
        leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()

        leaf_lang_feat[leaf_occu_count < 2] *= 0.0
        leaf_ind = leaf_ind.clamp(max=leaf_lang_feat.shape[0] - 1)

        # (6) 加载文本特征并匹配
        text_feat_path = 'E:/opengaussians/OpenGaussian-main/assets/text_features/text_features.json'
        with open(text_feat_path, 'r') as f:
            data_loaded = json.load(f)

        all_texts = list(data_loaded.keys())
        text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)

        query_text_feats = torch.zeros(len(target_names), 512).cuda()
        for i, text in enumerate(target_names):
            feat = text_features[all_texts.index(text)].unsqueeze(0)
            query_text_feats[i] = feat

        # (7) 计算余弦相似度获取预测结果
        query_text_feats = F.normalize(query_text_feats, dim=1, p=2)
        leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)
        cosine_similarity = torch.matmul(query_text_feats, leaf_lang_feat.transpose(0, 1))
        max_id = torch.argmax(cosine_similarity, dim=0)
        pred_pts_cls_id = max_id[leaf_ind] + 1

        # (8) 计算指标
        ious, mean_iou, accuracy, mean_acc = calculate_metrics(updated_gt_labels, pred_pts_cls_id,
                                                               total_classes=len(target_names) + 1)

        # --- 保存当前场景结果 ---
        final_results["scenes"][scan_name] = {
            "mIoU": round(float(mean_iou), 4),
            "mAcc": round(float(mean_acc), 4),
            "overall_accuracy": round(float(accuracy), 4),
            "per_class_iou": {name: round(float(ious[i + 1]), 4) for i, name in enumerate(target_names)}
        }

        all_mious.append(mean_iou)
        all_maccs.append(mean_acc)

        print(f"Scene: {scan_name}, mIoU: {mean_iou:.4f}, mAcc.: {mean_acc:.4f}")

        # --- 计算全场景平均值并保存 ---
    if all_mious:
        final_results["overall_average"] = {
            "mean_mIoU": round(float(np.mean(all_mious)), 4),
            "mean_mAcc": round(float(np.mean(all_maccs)), 4)
        }

    # (9) 最终写入 JSON 文件
    output_filename = f"eval_results_iter_{iteration}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\n[Finished] 评估完成，结果已保存至: {output_filename}")