"""
NTA-IoU (Novel Trajectory Agent IoU) 评测模块

使用 YOLO11 对生成图像和真值图像进行交通参与者检测，
比较两者检测结果的一致性来评估生成图像中车辆等目标的真实性。

评测逻辑（与 DriveDreamer4D 论文一致）：
1. 将图像 resize 到 480×320 后进行 YOLO11 目标检测（conf > 0.5）
2. 以gt图像的检测结果作为参考，按中心距离为每个gt框找最近的gen检测框
3. 中心距离 < 阈值时视为匹配，计算匹配框对的IoU，未匹配的gt框IoU记为0
4. NTA-IoU = 所有gt目标IoU的平均值

适用场景：
- 自车轨迹（有GT）：评估生成图与GT图中车辆检测一致性
- 异车轨迹（无GT图像标注）：用检测模型的预测结果互比，间接评估生成质量

参考: DriveDreamer4D (YOLO11)
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

from utils import load_config, get_image_pairs, load_image, load_image_pair, ensure_dir
from metrics import aggregate_metrics, format_metrics_table

# COCO类别中的交通参与者ID
# YOLO11 使用 COCO 类别
TRAFFIC_AGENT_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

# 所有交通参与者的类别ID集合
TRAFFIC_AGENT_IDS = set(TRAFFIC_AGENT_CLASSES.keys())


class YOLO11Detector:
    """
    YOLO11 目标检测器
    用于检测图像中的交通参与者
    """

    def __init__(self, model_size: str = "l", device: str = "cuda",
                 conf_threshold: float = 0.5, iou_threshold: float = 0.45,
                 detect_size: Tuple[int, int] = (480, 320)):
        """
        Args:
            model_size: 模型大小 (n/s/m/l/x)
            device: 推理设备
            conf_threshold: 置信度阈值（论文使用0.5）
            iou_threshold: NMS的IoU阈值
            detect_size: 检测前resize的目标尺寸 (width, height)，论文使用(480, 320)
        """
        self.model_size = model_size
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.detect_size = detect_size
        self._load_model()

    def _load_model(self):
        """加载 YOLO11 模型"""
        from ultralytics import YOLO

        model_name = f"yolo11{self.model_size}.pt"
        print(f"加载 YOLO11 模型: {model_name}")

        self.model = YOLO(model_name)
        self.model.to(self.device)
        print(f"成功加载 YOLO11 ({self.model_size})")

    def detect(self, image: np.ndarray,
               filter_classes: Optional[set] = None) -> List[Dict]:
        """
        检测图像中的目标

        与论文一致：先将图像resize到detect_size再进行检测。

        Args:
            image: RGB图像 (H, W, 3), uint8
            filter_classes: 只保留这些类别的检测结果，None则保留所有

        Returns:
            detections: 检测结果列表，每个检测包含:
                - bbox: [x1, y1, x2, y2] 边界框坐标（在resize后的图像上）
                - confidence: 置信度
                - class_id: 类别ID
                - class_name: 类别名称
        """
        import cv2
        # 与论文一致：resize到指定尺寸后检测
        resized = cv2.resize(image, self.detect_size)

        results = self.model(
            resized,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # 过滤类别
                if filter_classes is not None and class_id not in filter_classes:
                    continue

                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].item())
                class_name = result.names.get(class_id, str(class_id))

                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name,
                })

        return detections


def compute_bbox_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的IoU

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU值
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
        return 0.0
    return intersection / union



def find_closest_box(gt_box: List[float], gen_boxes: List[List[float]],
                     distance_threshold: float = 10.0) -> Optional[List[float]]:
    """
    为gt框找到中心距离最近的gen检测框（与论文一致）

    Args:
        gt_box: [x1, y1, x2, y2] GT边界框
        gen_boxes: gen检测框列表
        distance_threshold: 最大中心距离阈值（论文默认10像素）

    Returns:
        最近的gen检测框，若超出距离阈值则返回None
    """
    gt_center = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]
    min_distance = distance_threshold
    closest_box = None
    for gen_box in gen_boxes:
        gen_center = [(gen_box[0] + gen_box[2]) / 2, (gen_box[1] + gen_box[3]) / 2]
        distance = np.linalg.norm(np.array(gt_center) - np.array(gen_center))
        if distance < min_distance:
            min_distance = distance
            closest_box = gen_box
    return closest_box


def match_detections(dets_gen: List[Dict], dets_gt: List[Dict],
                     distance_threshold: float = 10.0) -> Dict[str, float]:
    """
    匹配gen和gt的检测结果，计算NTA-IoU（与论文一致）

    使用中心距离匹配：对每个gt框，找中心距离最近的gen框，
    距离 < 阈值时计算IoU，否则IoU=0。
    NTA-IoU = 所有gt框的平均IoU。

    Args:
        dets_gen: gen图像的检测结果
        dets_gt: gt图像的检测结果
        distance_threshold: 中心距离匹配阈值（像素，论文默认10）

    Returns:
        匹配指标字典:
            - nta_iou: 所有gt目标的平均IoU（核心指标，与论文一致）
            - nta_precision: 匹配上的gen检测 / 总gen检测
            - nta_recall: 匹配上的gt目标 / 总gt目标
            - nta_num_gen: gen图检测到的目标数
            - nta_num_gt: gt图检测到的目标数
            - nta_num_matched: 成功匹配的数量
    """
    n_gen = len(dets_gen)
    n_gt = len(dets_gt)

    # 边界情况：都没检测到
    if n_gen == 0 and n_gt == 0:
        return {
            'nta_iou': 100.0,  # 完美一致（都没有目标）
            'nta_precision': 100.0,
            'nta_recall': 100.0,
            'nta_num_gen': 0,
            'nta_num_gt': 0,
            'nta_num_matched': 0,
        }

    # 一方没检测到
    if n_gen == 0 or n_gt == 0:
        return {
            'nta_iou': 0.0,
            'nta_precision': 0.0 if n_gen > 0 else 100.0,
            'nta_recall': 0.0 if n_gt > 0 else 100.0,
            'nta_num_gen': n_gen,
            'nta_num_gt': n_gt,
            'nta_num_matched': 0,
        }

    # 与论文一致：对每个gt框，按中心距离找最近的gen框
    gen_boxes = [d['bbox'] for d in dets_gen]
    iou_values = []
    num_matched = 0

    for det_gt in dets_gt:
        gt_box = det_gt['bbox']
        closest = find_closest_box(gt_box, gen_boxes, distance_threshold)
        if closest is not None:
            iou = compute_bbox_iou(gt_box, closest)
            iou_values.append(iou)
            num_matched += 1
        else:
            iou_values.append(0)

    # NTA-IoU: 所有gt框的平均IoU（与论文一致）
    nta_iou = np.average(iou_values) * 100 if len(iou_values) > 0 else 0.0

    # Precision 和 Recall
    precision = (num_matched / n_gen) * 100 if n_gen > 0 else 0.0
    recall = (num_matched / n_gt) * 100 if n_gt > 0 else 0.0

    return {
        'nta_iou': nta_iou,
        'nta_precision': precision,
        'nta_recall': recall,
        'nta_num_gen': n_gen,
        'nta_num_gt': n_gt,
        'nta_num_matched': num_matched,
    }


def compute_nta_per_class(dets_gen: List[Dict], dets_gt: List[Dict],
                          distance_threshold: float = 10.0) -> Dict[str, float]:
    """
    按类别分别计算NTA-IoU

    Returns:
        per_class_nta: 每个交通参与者类别的NTA-IoU
    """
    per_class_results = {}

    for class_id, class_name in TRAFFIC_AGENT_CLASSES.items():
        gen_cls = [d for d in dets_gen if d['class_id'] == class_id]
        gt_cls = [d for d in dets_gt if d['class_id'] == class_id]

        if len(gen_cls) == 0 and len(gt_cls) == 0:
            continue  # 该类别不存在，跳过

        result = match_detections(gen_cls, gt_cls, distance_threshold)
        per_class_results[f'nta_iou_{class_name}'] = result['nta_iou']

    return per_class_results


def get_nta_detector(config: Dict) -> YOLO11Detector:
    """根据配置创建YOLO11检测器"""
    nta_config = config.get('nta', {})
    model_size = nta_config.get('model_size', 'l')
    device = nta_config.get('device', 'cuda')
    conf_threshold = nta_config.get('conf_threshold', 0.5)
    iou_threshold = nta_config.get('iou_threshold', 0.45)
    detect_w = nta_config.get('detect_width', 480)
    detect_h = nta_config.get('detect_height', 320)

    return YOLO11Detector(
        model_size=model_size,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        detect_size=(detect_w, detect_h),
    )


def evaluate_nta_consistency(config: Dict,
                              save_vis: bool = True) -> Dict[str, Dict]:
    """
    评测NTA-IoU（交通参与者检测一致性）

    对于每对(gen, gt)图像：
    1. 对两者分别进行YOLO11目标检测
    2. 过滤出交通参与者类别
    3. 匹配gen和gt的检测框，计算NTA-IoU

    Args:
        config: 配置字典
        save_vis: 是否保存可视化

    Returns:
        每个相机的评测结果
    """
    print("\n" + "=" * 60)
    print("NTA-IoU (交通参与者检测一致性) 评测")
    print("=" * 60)

    # 初始化检测器
    detector = get_nta_detector(config)

    # 获取图像对
    image_pairs = get_image_pairs(config)
    if not image_pairs:
        raise ValueError("没有找到有效的图像对")

    # 配置
    nta_config = config.get('nta', {})
    distance_threshold = nta_config.get('distance_threshold', 10.0)
    compute_per_class = nta_config.get('per_class', True)

    # 创建输出目录
    output_dir = os.path.join(config['output'].get('root', './results'), 'nta_maps')
    if save_vis:
        ensure_dir(output_dir)

    results = {}

    for camera, pairs in image_pairs.items():
        print(f"\n处理相机: {camera}")
        camera_metrics = []

        if save_vis:
            camera_out_dir = os.path.join(output_dir, camera)
            ensure_dir(camera_out_dir)

        for gen_path, gt_path in tqdm(pairs, desc=f"  {camera}"):
            filename = os.path.basename(gen_path).replace('.png', '')

            # 加载图像
            gen_img, gt_img = load_image_pair(gen_path, gt_path)

            # YOLO11 检测（只保留交通参与者）
            dets_gen = detector.detect(gen_img, filter_classes=TRAFFIC_AGENT_IDS)
            dets_gt = detector.detect(gt_img, filter_classes=TRAFFIC_AGENT_IDS)

            # 计算NTA-IoU（中心距离匹配，与论文一致）
            metrics = match_detections(dets_gen, dets_gt, distance_threshold)

            # 按类别计算
            if compute_per_class:
                per_class = compute_nta_per_class(dets_gen, dets_gt, distance_threshold)
                metrics.update(per_class)

            camera_metrics.append(metrics)

            # 保存检测可视化（GT vs Gen 对比图）
            if save_vis:
                _save_detection_vis_compare(
                    gen_img, gt_img, dets_gen, dets_gt, metrics,
                    os.path.join(camera_out_dir, f"{filename}_compare.png")
                )

        # 汇总该相机的指标
        results[camera] = aggregate_metrics(camera_metrics)

        print(f"\n  {camera} NTA-IoU 指标:")
        print(format_metrics_table(results[camera]))

    # 计算总体平均
    all_metrics = []
    for camera_results in results.values():
        all_metrics.append({k: v for k, v in camera_results.items()
                          if not k.endswith('_std') and not isinstance(v, list)})
    results['overall'] = aggregate_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("总体 NTA-IoU 评测结果:")
    print(format_metrics_table(results['overall']))

    return results


_DET_COLORS = {
    0: (220, 20, 60),    # person - 红
    1: (119, 11, 32),    # bicycle - 深红
    2: (0, 0, 142),      # car - 蓝
    3: (0, 0, 230),      # motorcycle - 亮蓝
    5: (0, 60, 100),     # bus - 深青
    7: (0, 0, 70),       # truck - 深蓝
}


def _draw_detections(draw, detections: List[Dict], font):
    """在ImageDraw上绘制检测框"""
    for det in detections:
        bbox = det['bbox']
        cls_id = det['class_id']
        color = _DET_COLORS.get(cls_id, (255, 255, 255))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        draw.rectangle(bbox, outline=color, width=2)
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 14), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1] - 14), label, fill=(255, 255, 255), font=font)


def _save_detection_vis(image: np.ndarray, detections: List[Dict],
                        save_path: str):
    """保存单张检测结果可视化（保留向后兼容）"""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    img = PILImage.fromarray(image)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    _draw_detections(draw, detections, font)
    img.save(save_path)


def _save_detection_vis_compare(gen_img: np.ndarray, gt_img: np.ndarray,
                                dets_gen: List[Dict], dets_gt: List[Dict],
                                metrics: Dict[str, float],
                                save_path: str):
    """
    保存GT与Gen的目标检测对比可视化（拼成一张图）

    左列：GT 原图 + 检测框
    右列：Gen 原图 + 检测框
    底部：检测统计和匹配指标
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    h, w = gt_img.shape[:2]

    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_title = font

    # 绘制GT检测图
    gt_pil = PILImage.fromarray(gt_img)
    gt_draw = ImageDraw.Draw(gt_pil)
    _draw_detections(gt_draw, dets_gt, font)

    # 绘制Gen检测图
    gen_pil = PILImage.fromarray(gen_img)
    gen_draw = ImageDraw.Draw(gen_pil)
    _draw_detections(gen_draw, dets_gen, font)

    # 拼接：左GT右Gen + 底部信息
    info_h = 80
    canvas_w = w * 2 + 4
    canvas_h = h + info_h
    canvas = PILImage.new('RGB', (canvas_w, canvas_h), (0, 0, 0))

    canvas.paste(gt_pil, (0, 0))
    canvas.paste(gen_pil, (w + 4, 0))

    draw = ImageDraw.Draw(canvas)

    # 中间分隔线
    draw.rectangle([w, 0, w + 3, h - 1], fill=(255, 255, 255))

    # 标题（左上角半透明标签）
    draw.text((10, 5), f"GT ({len(dets_gt)} objects)", fill=(255, 255, 0), font=font_title)
    draw.text((w + 14, 5), f"Gen ({len(dets_gen)} objects)", fill=(255, 255, 0), font=font_title)

    # 底部信息区（深灰背景）
    draw.rectangle([0, h, canvas_w, canvas_h], fill=(40, 40, 40))

    info_y = h + 6

    # 按类别统计检测数
    def count_by_class(dets):
        counts = {}
        for d in dets:
            name = d['class_name']
            counts[name] = counts.get(name, 0) + 1
        return counts

    gt_counts = count_by_class(dets_gt)
    gen_counts = count_by_class(dets_gen)
    all_classes = sorted(set(list(gt_counts.keys()) + list(gen_counts.keys())))

    class_str = "  ".join(
        f"{cls}: GT={gt_counts.get(cls, 0)} Gen={gen_counts.get(cls, 0)}"
        for cls in all_classes
    )
    if not class_str:
        class_str = "(no detections)"

    # 指标
    nta_iou = metrics.get('nta_iou', 0)
    nta_prec = metrics.get('nta_precision', 0)
    nta_recall = metrics.get('nta_recall', 0)
    n_matched = metrics.get('nta_num_matched', 0)
    n_gt = metrics.get('nta_num_gt', 0)
    n_gen = metrics.get('nta_num_gen', 0)

    line1 = f"GT: {n_gt} objects  |  Gen: {n_gen} objects  |  Matched: {n_matched}  |  {class_str}"
    line2 = f"NTA-IoU: {nta_iou:.1f}%  |  Precision: {nta_prec:.1f}%  |  Recall: {nta_recall:.1f}%"

    # 按类别IoU
    per_class_ious = []
    for cls in all_classes:
        key = f"nta_iou_{cls}"
        if key in metrics:
            per_class_ious.append(f"{cls}: {metrics[key]:.1f}%")
    if per_class_ious:
        line3 = "Per-class IoU:  " + "  |  ".join(per_class_ious)
    else:
        line3 = ""

    draw.text((10, info_y), line1, fill=(200, 200, 200), font=font)
    draw.text((10, info_y + 20), line2, fill=(0, 255, 150), font=font)
    if line3:
        draw.text((10, info_y + 40), line3, fill=(180, 180, 255), font=font)

    canvas.save(save_path)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="NTA-IoU 交通参与者检测一致性评测")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--no-vis", action="store_true",
                       help="不保存可视化结果")
    parser.add_argument("--output", type=str, default=None,
                       help="结果输出路径")
    args = parser.parse_args()

    config = load_config(args.config)
    results = evaluate_nta_consistency(config, save_vis=not args.no_vis)

    output_path = args.output or os.path.join(config['output']['metrics'], "nta_results.json")
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
