import json

import math
import re
from typing import Dict, List, Optional, Tuple


# Constants
TEXT_ANLS_THRESHOLD = 0.5
MAX_L2_DISTANCE = 140
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>")
THINK_ACTION_PATTERN = re.compile(r"<think>.*?</think>\s*<action>.*?</action>", re.DOTALL)
COORD_PATTERNS = {"start_box": r"start_box='<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>'", "end_box": r"end_box='\((\d+),(\d+)\)'"}
COORD_PATTERNS_ans = {"start_box": r"start_box='\((\d+),(\d+)\)'", "end_box": r"end_box='\((\d+),(\d+)\)'"}
COORD_PATTERNS_ans_loose = {"start_box": r"start_box='\((\d+), (\d+)\)'", "end_box": r"end_box='\((\d+),(\d+)\)'"}

BBOX_PATTERN = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]"
CONTENT_PATTERN = r"content='(.*?)'"
APP_NAME_PATTERN = r"app_name='(.*?)'"
DIRECTION_PATTERN = r"direction='(.*?)'"


def extract_coordinates(input_str: str, box_prefix: str = "start_box") -> Optional[Tuple[int, int]]:
    """Extract coordinates from input string using specified pattern."""
    match = re.search(COORD_PATTERNS[box_prefix], input_str)
    return (int(match.group(1)), int(match.group(2))) if match else None


def extract_coordinates_ans(input_str: str, box_prefix: str = "start_box") -> Optional[Tuple[int, int]]:
    """Extract coordinates from input string using specified pattern."""
    match = re.search(COORD_PATTERNS_ans[box_prefix], input_str)
    if not match:
        match = re.search(COORD_PATTERNS_ans_loose[box_prefix], input_str)
    return [int(match.group(1)), int(match.group(2))] if match else None


def gui_format_reward(predict_str: str) -> float:
    """Check if prediction follows required format."""
    return 1.0 if re.fullmatch(THINK_ACTION_PATTERN, predict_str) else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            distances_.append(distances[i1] if c1 == c2 else 1 + min(distances[i1], distances[i1 + 1], distances_[-1]))
        distances = distances_
    return distances[-1]


def text_matching(gt: str, pred: str) -> bool:
    """Check if two texts match based on threshold."""
    gt, pred = gt.strip(), pred.strip()
    if gt in pred or pred in gt:
        return True

    dist = levenshtein_distance(gt, pred)
    length = max(len(gt), len(pred))
    similarity = 1 - (float(dist) / length) if length > 0 else 0.0
    return similarity >= TEXT_ANLS_THRESHOLD


def get_action_content(action_str: str, pattern: str) -> str:
    """Helper to extract content from action string."""
    match = re.search(pattern, action_str)
    return match.group(1) if match else ""


def calculate_point_reward(predict_coord: Tuple[int, int], gt_coord: Tuple[int, int]) -> float:
    """Calculate reward based on L2 distance between points."""
    l2_distance = math.sqrt((predict_coord[0] - gt_coord[0]) ** 2 + (predict_coord[1] - gt_coord[1]) ** 2)

    if l2_distance > MAX_L2_DISTANCE:
        return 0.0

    
    return 1


def gui_accuracy_reward(predict_str: str, ground_truth: str, image_size) -> List[float]:
    """Calculate accuracy rewards based on action type matching."""
    # try:
    type_acc = 0
    action_acc = 0
    is_ground = False

    ground_truth = ground_truth.strip()
    action_match = re.search(ACTION_PATTERN, predict_str)
    given_answer = action_match.group(1).strip() if action_match else predict_str.strip()

    action_type = given_answer.split("(")[0].strip()
    gt_action_type = ground_truth.split("(")[0].strip()

    if action_type != gt_action_type:
        return type_acc, action_acc, is_ground, gt_action_type

    type_acc = 1  # Action type match reward

    # Handle different action types
    if gt_action_type in ["click", "long_press"]:
        gt_coords = extract_coordinates_ans(ground_truth)
        # print(ground_truth)
        # pred_coords = extract_coordinates(predict_str)

        pred_coords = extract_coordinates_ans(predict_str)
        # print(gt_coords, pred_coords)
        is_ground = True
        width, height = image_size[-1]
        if gt_coords and pred_coords:
            pred_coords[0] = int(pred_coords[0] / width * 1000)
            pred_coords[1] = int(pred_coords[1] / height * 1000)

            gt_coords[0] = int(gt_coords[0] / width * 1000)
            gt_coords[1] = int(gt_coords[1] / height * 1000)

            # print("pred_coords", pred_coords)
            # print("gt_coords", gt_coords)

            action_acc = calculate_point_reward(pred_coords, gt_coords)

    elif gt_action_type == "scroll":
        # 调整以符合AndroidControl格式
        gt_content: str = get_action_content(ground_truth, DIRECTION_PATTERN)
        pred_content = get_action_content(given_answer, DIRECTION_PATTERN)
        action_acc = 1.0 if (gt_content == pred_content) else 0


    elif gt_action_type == "type":
        gt_content: str = get_action_content(ground_truth, CONTENT_PATTERN)
        pred_content = get_action_content(given_answer, CONTENT_PATTERN)
        action_acc = 1.0 if text_matching(gt_content, pred_content) else 0.0

    elif gt_action_type == "open_app":
        gt_app = get_action_content(ground_truth, APP_NAME_PATTERN)
        pred_app = get_action_content(given_answer, APP_NAME_PATTERN)
        action_acc = 1.0 if text_matching(gt_app, pred_app) else 0.0

    # 剩下几种action：finish wait press_back press_home
    else:
        action_acc = 1

    return type_acc, action_acc, is_ground, gt_action_type

    # except Exception:
    #     print()
    #     return [0.0]


def gui_iou_reward(predict_str: str, ground_truth: str) -> float:
    """Calculate IOU reward for bounding box predictions."""

    def iou(box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        inter_x1, inter_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        inter_x2, inter_y2 = min(box1[2] - 1, box2[2] - 1), min(box1[3] - 1, box2[3] - 1)

        inter = (
            (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1) if inter_x1 < inter_x2 and inter_y1 < inter_y2 else 0
        )
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
        return float(inter) / union if union > 0 else 0.0

    try:
        gt_match = re.search(BBOX_PATTERN, ground_truth)
        pred_match = re.search(BBOX_PATTERN, predict_str)

        if gt_match and pred_match:
            gt_box = [int(gt_match.group(i)) for i in range(1, 5)]
            pred_box = [int(pred_match.group(i)) for i in range(1, 5)]
            return 1.0 if iou(pred_box, gt_box) > 0.5 else 0.0
    except Exception:
        pass
    return 0.0

def gui_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    accuracy = sum(gui_accuracy_reward(predict_str, ground_truth))
    format = gui_format_reward(predict_str)

    return {
        "overall": 0.5 * accuracy + 0.5 * format,
        "format": format,
        "accuracy": accuracy,
    }



import os
import argparse
all_data = []

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--file_path_dir",
    type=str,
    required=True,
    help="Path to the directory containing JSON files.",
)
file_path_dir = argparser.parse_args().file_path_dir

# file_path_dir = '/data/zhouxurui/Qwen2-5-VL-finetune-code/eval_results/kl0_9epoch_700samples_bs64_hint_v2_diff_0_01_click_1_2'
# file_path_dir = '/data/zhouxurui/Qwen2-5-VL-finetune-code/eval_results/kl0_9epoch_700samples_bs64_hint_v2_diff_0_01_click_1_43'
# file_path_dir = '/data/zhouxurui/Qwen2-5-VL-finetune-code/eval_results/kl0_real5epoch_700samples_bs64_hint_v2_diff_0_01_click_1_43'
# file_path_dir = '/data/zhouxurui/Qwen2-5-VL-finetune-code/eval_results/Baseline_kl0_V1_attempt_Qwen2-5VL-3B_9epoch_700samples'
for filename in os.listdir(file_path_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(file_path_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

# all_data 现在是一个包含所有 json 文件内容的 list
# print(f"总共读取了 {len(all_data)} 条数据")



# 计算三个指标：action_type acc，SSR，grounding acc
from collections import defaultdict

ssr_correct_num = 0
type_correct_num = 0
all_sample_nums = len(all_data)
all_grounding_num = 0
all_grounding_num_correct = 0

cnt = 0

# 用于统计每种gt_action_type的准确率
action_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for result in all_data:
    pred = result['response']
    answer = result['answer']
    image_size = all_data[cnt]['image_size']
    type_acc, action_acc, is_ground, gt_action_type = gui_accuracy_reward(pred, answer, image_size)
    
    # 累计总准确率
    type_correct_num += type_acc
    ssr_correct_num += action_acc
    all_grounding_num += int(is_ground)
    if is_ground:
        all_grounding_num_correct += action_acc
    
    # 更新该类别的统计
    action_type_stats[gt_action_type]["total"] += 1
    action_type_stats[gt_action_type]["correct"] += action_acc
    
    cnt += 1

# 打印整体结果
print(f"Action Type Accuracy: {type_correct_num / all_sample_nums:.4f}")
print(f"SSR: {ssr_correct_num / all_sample_nums:.4f}")
print(f"Grounding Accuracy: {all_grounding_num_correct / all_grounding_num:.4f}")

# 打印每种gt_action_type的准确率
all_accuracy = 0
print("\nPer Action Type Accuracy:")
for action_type, stats in action_type_stats.items():
    total = stats["total"]
    correct = stats["correct"]
    accuracy = correct / total if total > 0 else 0.0
    all_accuracy += accuracy
    print(f"{action_type}: {accuracy:.4f} ({correct}/{total})")

print("marco accL:", all_accuracy / 7)