import math
import re
from typing import Dict, List, Optional, Tuple


# Constants
TEXT_ANLS_THRESHOLD = 0.5
MAX_L2_DISTANCE = 140
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>")
THINK_ACTION_PATTERN = re.compile(r"<think>.*?</think>\s*<action>.*?</action>", re.DOTALL)
THINK_ACTION_PATTERN_without_thought = re.compile(r"<action>.*?</action>", re.DOTALL)
INTENTION_THINK_PATTERN = re.compile(
    r"<think>.*?</think>\s*"
    r"<intention>.*?</intention>\s*"
    r"<action>.*?</action>",
    re.DOTALL
)
PRED_THINK_PATTERN = re.compile(
    r"<think>.*?</think>\s*"
    r"<action>.*?</action>\s*"
    r"<think_pred>.*?</think_pred>\s*"
    r"<next_user_step1>.*?</next_user_step1>\s*"
    r"<next_user_step2>.*?</next_user_step2>",
    re.DOTALL
)
PRED_THINK_OPERATION_PATTERN = re.compile(
    r"<think>.*?</think>\s*"
    r"<action>.*?</action>\s*"
    r"<think_pred>.*?</think_pred>\s*"
    r"<future_action_1>.*?</future_action_1>\s*"
    r"<future_action_2>.*?</future_action_2>",
    re.DOTALL
)
PRED_THINK_OPERATION_ONE_PATTERN = re.compile(
    r"<think>.*?</think>\s*"
    r"<action>.*?</action>\s*"
    r"<think_pred>.*?</think_pred>\s*"
    r"<future_action>.*?</future_action>\s*",
    re.DOTALL
)
COORD_PATTERNS = {"start_box": r"start_box='\((\d+),(\d+)\)'", "end_box": r"end_box='\((\d+),(\d+)\)'"}
COORD_PATTERNS_loose = {"start_box": r"start_box='\((\d+), (\d+)\)'", "end_box": r"end_box='\((\d+), (\d+)\)'"}
BBOX_PATTERN = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]"
CONTENT_PATTERN = r"content='(.*?)'"
APP_NAME_PATTERN = r"app_name='(.*?)'"
DIRECTION_PATTERN = r"direction='(.*?)'"

# from sentence_transformers import SentenceTransformer, util
# SBERT_AVAILABLE = True
# SBERT_PATH = "/data/models/all-MiniLM-L6-v2"
# sentence_bert_model = SentenceTransformer(SBERT_PATH, device="cpu")


def extract_coordinates(input_str: str, box_prefix: str = "start_box") -> Optional[Tuple[int, int]]:
    """Extract coordinates from input string using specified pattern."""
    match = re.search(COORD_PATTERNS[box_prefix], input_str)
    if not match:
        match = re.search(COORD_PATTERNS_loose[box_prefix], input_str)
    return [int(match.group(1)), int(match.group(2))] if match else None


def gui_format_reward(predict_str: str) -> float:
    """Check if prediction follows required format."""
    return 1.0 if re.fullmatch(THINK_ACTION_PATTERN, predict_str) else 0.0

def gui_format_reward_without_thought(predict_str: str) -> float:
    """Check if prediction follows required format."""
    return 1.0 if re.fullmatch(THINK_ACTION_PATTERN_without_thought, predict_str) else 0.0

def gui_format_reward_intention(predict_str: str) -> float:
    """Check if prediction follows required format."""
    return 1.0 if re.fullmatch(INTENTION_THINK_PATTERN, predict_str) else 0.0

def gui_format_pred_reward(predict_str: str) -> float:
    # then check 
    """Check if prediction follows required format."""
    return 1.0 if re.search(PRED_THINK_PATTERN, predict_str) else 0.0

def gui_format_pred_operation_reward(predict_str: str) -> float:
    # then check 
    """Check if prediction follows required format."""
    return 1.0 if re.search(PRED_THINK_OPERATION_PATTERN, predict_str) else 0.0

def gui_format_pred_operation_reward_one(predict_str: str) -> float:
    # then check 
    """Check if prediction follows required format."""
    return 1.0 if re.search(PRED_THINK_OPERATION_ONE_PATTERN, predict_str) else 0.0

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

def calculate_point_reward_grounding(predict_coord: Tuple[int, int], gt_coord: Tuple[int, int], grounding_verison) -> float:
    """Calculate reward based on L2 distance between points."""
    l2_distance = math.sqrt((predict_coord[0] - gt_coord[0]) ** 2 + (predict_coord[1] - gt_coord[1]) ** 2)

    # 0.14内外都用1-x
    if grounding_verison == 1:
        return 1 - l2_distance / 1000
    # 0.14内是1，外是1-x
    elif grounding_verison == 2:
        if l2_distance > MAX_L2_DISTANCE:
            return 1 - l2_distance / 1000
        return 1
    # 0.14内是1-x，外是0
    elif grounding_verison == 3:
        if l2_distance > MAX_L2_DISTANCE:
            return 0
        return 1 - l2_distance / 1000
    # 0.14内是1，外是0
    elif grounding_verison == 4:
        if l2_distance > MAX_L2_DISTANCE:
            return 0
        return 1
    elif grounding_verison == 5:
        return 1 - (l2_distance / 1000)**4
    elif grounding_verison == 6:
        return 1 - (l2_distance / 1000)**0.25
    elif grounding_verison == 7:
        if l2_distance > 200:
            return 0
        g_value = 0.01
        if l2_distance > MAX_L2_DISTANCE:
            g_value += 1
        return g_value
    elif grounding_verison == 8:
        if l2_distance > 300:
            return 0
        g_value = 0.01
        if l2_distance > MAX_L2_DISTANCE:
            g_value += 1
        return g_value

def calculate_point_reward(predict_coord: Tuple[int, int], gt_coord: Tuple[int, int]) -> float:
    """Calculate reward based on L2 distance between points."""
    l2_distance = math.sqrt((predict_coord[0] - gt_coord[0]) ** 2 + (predict_coord[1] - gt_coord[1]) ** 2)

    if l2_distance > MAX_L2_DISTANCE:
        return 0.0

    decay_factor = math.log(2) / MAX_L2_DISTANCE
    return math.exp(-decay_factor * l2_distance)


def gui_accuracy_reward(predict_str: str, ground_truth: str, image_size = None) -> Dict[str, float | str]:
    """Calculate accuracy rewards based on action type matching."""
    # try:
    step_reward = {"action": 0.0, "value": 0.0, "type": ""}
    ground_truth = ground_truth.strip()
    action_match = re.search(ACTION_PATTERN, predict_str)
    given_answer = action_match.group(1).strip() if action_match else predict_str.strip()

    action_type = given_answer.split("(")[0].strip()
    gt_action_type = ground_truth.split("(")[0].strip()

    if action_type != gt_action_type:
        return step_reward

    step_reward["type"] = action_type
    step_reward["action"] = 1.0

    # Handle different action types
    if gt_action_type in ["click", "long_press"]:
        gt_coords = extract_coordinates(ground_truth)
        pred_coords = extract_coordinates(predict_str)
        # print("predict_str:", predict_str)
        # print(pred_coords, gt_coords)
        if gt_coords and pred_coords:
            if image_size is not None:
                # print(pred_coords, gt_coords)
                width, height = image_size
                pred_coords[0] = int(pred_coords[0] / width * 1000)
                pred_coords[1] = int(pred_coords[1] / height * 1000)

                gt_coords[0] = int(gt_coords[0] / width * 1000)
                gt_coords[1] = int(gt_coords[1] / height * 1000)

            step_reward["value"] = calculate_point_reward(pred_coords, gt_coords)
            # print(pred_coords, gt_coords, step_reward["value"])
            

    elif gt_action_type == "scroll":
        # 调整以符合AndroidControl格式
        gt_content: str = get_action_content(ground_truth, DIRECTION_PATTERN)
        pred_content = get_action_content(given_answer, DIRECTION_PATTERN)
        step_reward["value"] = 1.0 if (gt_content == pred_content) else 0.0

    elif gt_action_type == "type":
        gt_content: str = get_action_content(ground_truth, CONTENT_PATTERN)
        pred_content = get_action_content(given_answer, CONTENT_PATTERN)
        step_reward["value"] = 1.0 if text_matching(gt_content, pred_content) else 0.0

    elif gt_action_type == "open_app":
        gt_app = get_action_content(ground_truth, APP_NAME_PATTERN)
        pred_app = get_action_content(given_answer, APP_NAME_PATTERN)
        step_reward["value"] = 1.0 if text_matching(gt_app, pred_app) else 0.0

    # 剩下几种action：finish wait press_back press_home
    else:
        step_reward["value"] = 1.0

    return step_reward

    # except Exception:
    #     return step_reward



def gui_accuracy_reward_grounding(predict_str: str, ground_truth: str, image_size = None, grounding_version = None) -> Dict[str, float | str]:
    """Calculate accuracy rewards based on action type matching."""
    # try:
    step_reward = {"action": 0.0, "value": 0.0, "type": ""}
    ground_truth = ground_truth.strip()
    action_match = re.search(ACTION_PATTERN, predict_str)
    given_answer = action_match.group(1).strip() if action_match else predict_str.strip()

    action_type = given_answer.split("(")[0].strip()
    gt_action_type = ground_truth.split("(")[0].strip()

    if action_type != gt_action_type:
        return step_reward

    step_reward["type"] = action_type
    step_reward["action"] = 1.0

    # Handle different action types
    if gt_action_type in ["click", "long_press"]:
        gt_coords = extract_coordinates(ground_truth)
        pred_coords = extract_coordinates(predict_str)
        # print("predict_str:", predict_str)
        # print(pred_coords, gt_coords)
        if gt_coords and pred_coords:
            if image_size is not None:
                # print(pred_coords, gt_coords)
                width, height = image_size
                pred_coords[0] = int(pred_coords[0] / width * 1000)
                pred_coords[1] = int(pred_coords[1] / height * 1000)

                gt_coords[0] = int(gt_coords[0] / width * 1000)
                gt_coords[1] = int(gt_coords[1] / height * 1000)
            step_reward["value"] = calculate_point_reward_grounding(pred_coords, gt_coords, grounding_version)
            

    elif gt_action_type == "scroll":
        # 调整以符合AndroidControl格式
        gt_content: str = get_action_content(ground_truth, DIRECTION_PATTERN)
        pred_content = get_action_content(given_answer, DIRECTION_PATTERN)
        step_reward["value"] = 1.0 if (gt_content == pred_content) else 0.0

    elif gt_action_type == "type":
        gt_content: str = get_action_content(ground_truth, CONTENT_PATTERN)
        pred_content = get_action_content(given_answer, CONTENT_PATTERN)
        step_reward["value"] = 1.0 if text_matching(gt_content, pred_content) else 0.0

    elif gt_action_type == "open_app":
        gt_app = get_action_content(ground_truth, APP_NAME_PATTERN)
        pred_app = get_action_content(given_answer, APP_NAME_PATTERN)
        step_reward["value"] = 1.0 if text_matching(gt_app, pred_app) else 0.0

    # 剩下几种action：finish wait press_back press_home
    else:
        step_reward["value"] = 1.0

    return step_reward

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
    step_score = gui_accuracy_reward(predict_str, ground_truth)
    format = gui_format_reward(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0

    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format,
        "format_score": format,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }


def gui_compute_score_without_thought(predict_str: str, ground_truth: str, image_size) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version=1)
    format = gui_format_reward_without_thought(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.86:
            grounding_acc = 1
        else:
            grounding_acc = 0

    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format,
        "format_score": format,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1.86 else 0,
    }



def gui_compute_score_determ(predict_str: str, ground_truth: str, image_size) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward(predict_str, ground_truth, image_size)
    format_score = gui_format_reward(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0

    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }


def gui_compute_score_determ_format(predict_str: str, ground_truth: str, image_size) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward(predict_str, ground_truth, image_size)
    format_score = gui_format_pred_reward(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0

    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }


# from sentence_transformers import SentenceTransformer, util

def calc_similarity_score(model_text, gold_text):
    global sentence_bert_model
    
    if len(gold_text) == 0:
        return 0

    # 首先匹配pred_text中的
    model_text = re.findall(r"<next_user_step[12]>(.*?)</next_user_step[12]>", model_text, re.DOTALL)
    # print(model_text)
    if len(model_text) == 0:
        return 0
    
    # print("model_text", model_text)
    # print("gold_text:", gold_text)
    
    pred_score = 0
    length = min(len(gold_text), len(model_text))
    for i in range(length):
        # Calculate embeddings
        pred_embedding = sentence_bert_model.encode(model_text[i], convert_to_tensor=True, show_progress_bar=False)
        gold_embedding = sentence_bert_model.encode(gold_text[i], convert_to_tensor=True, show_progress_bar=False)
        pred_score += util.pytorch_cos_sim(pred_embedding, gold_embedding).item()
    if length == 0:
        return 0
    pred_score /= length
    return pred_score
    # calc_similarity_score(pred_action['answer'], gold_action['answer'])


def calc_similarity_score_intention(model_text, gold_text):
    global sentence_bert_model

    # 首先匹配pred_text中的
    model_text = re.findall(r"<intention>(.*?)</intention>", model_text, re.DOTALL)
    # print(model_text)
    if len(model_text) == 0 or len(model_text) > 1:
        return 0
        
    pred_embedding = sentence_bert_model.encode(model_text, convert_to_tensor=True, show_progress_bar=False)
    gold_embedding = sentence_bert_model.encode(gold_text, convert_to_tensor=True, show_progress_bar=False)
    return util.pytorch_cos_sim(pred_embedding, gold_embedding).item()

def calc_similarity_score_intention_syntax(model_text, gold_text):
    global sentence_bert_model

    # 首先匹配pred_text中的
    model_text = re.findall(r"<intention>(.*?)</intention>", model_text, re.DOTALL)
    # print(model_text)
    if len(model_text) == 0 or len(model_text) > 1:
        return 0
        
    pred_embedding = sentence_bert_model.encode(model_text, convert_to_tensor=True, show_progress_bar=False)
    gold_embedding = sentence_bert_model.encode(gold_text, convert_to_tensor=True, show_progress_bar=False)
    return util.pytorch_cos_sim(pred_embedding, gold_embedding).item()


def gui_compute_score_format(predict_str: str, ground_truth: str) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward(predict_str, ground_truth)
    format_score = gui_format_pred_reward(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0

    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }



def gui_compute_score_determ_grounding(predict_str: str, ground_truth: str, image_size, grounding_version) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version)
    format_score = gui_format_reward(predict_str)
    grounding_score = None
    type_score, action_type = step_score["action"], step_score["type"]

    Step_SR = 0
    if grounding_version == 4 or grounding_version == 3:
        Step_SR = 1 if step_score["action"] + step_score["value"] > 1 else 0
    elif grounding_version == 1 or grounding_version == 2:
        Step_SR = 1 if step_score["action"] + step_score["value"] > 1.86 else 0
    elif grounding_version == 5:
        Step_SR = 1 if step_score["action"] + step_score["value"] > 1.9804 else 0
    elif grounding_version == 6:
        Step_SR = 1 if step_score["action"] + step_score["value"] > 1.6258 else 0
    elif grounding_version == 7 or grounding_version == 8:
        Step_SR = 1 if step_score["action"] + step_score["value"] > 1.1 else 0
        if step_score["value"] > 0:
            step_score["value"] = 1
        else:
            step_score["value"] = 0


    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]


    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "SSR": Step_SR,
    }





def gui_compute_score_v2(predict_str: str, ground_truth: str, pred_steps, iamge_size=None) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward(predict_str, ground_truth, iamge_size)
    format_score = gui_format_pred_reward(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]
        # print(predict_str, ground_truth, grounding_score)

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0
    
    # 对pred_steps进行判断
    pred_score = calc_similarity_score(predict_str, pred_steps)

    all_record = {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score + 0.5 * pred_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "pred_score": pred_score,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }
    # print(all_record)

    return all_record


def operation_pred(model_text, pred_steps, image_size):
    # 首先匹配pred_text中的
    model_text = re.findall(r"<future_action_[12]>(.*?)</future_action_[12]>", model_text, re.DOTALL)
    # print(model_text)


    # print(model_text, pred_steps, image_size)
    if len(model_text) == 0:
        return 0
    length = min(len(pred_steps), len(model_text))
    if length == 0:
        return 0
        
    pred_score = 0
    for i in range(length):
        # Calculate embeddings
        step_score = gui_accuracy_reward(model_text[i], pred_steps[i], image_size)
        pred_score += 0.5 * (step_score["action"] + step_score["value"])
    pred_score /= length
    return pred_score

def operation_pred_one(model_text, pred_steps, image_size, grounding_version):
    # 首先匹配pred_text中的
    model_text = re.findall(r"<future_action>(.*?)</future_action>", model_text, re.DOTALL)
    # print(model_text)
    # print(model_text, pred_steps, image_size)
    length = min(len(pred_steps), len(model_text))
    if length == 0:
        return 0
    pred_steps = pred_steps[0]
    model_text = model_text[0]
    step_score = gui_accuracy_reward_grounding(model_text, pred_steps, image_size, grounding_version)
    pred_score = 0.5 * (step_score["action"] + step_score["value"])
    return pred_score


def gui_compute_score_v2_operation(predict_str: str, ground_truth: str, pred_steps, image_size=None) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward(predict_str, ground_truth, image_size)
    format_score = gui_format_pred_operation_reward(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]
        # print(predict_str, ground_truth, grounding_score)

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0
    
    # 对pred_steps进行判断
    pred_score = operation_pred(predict_str, pred_steps, image_size)

    all_record = {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score + 0.5 * pred_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "pred_score": pred_score,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }

    return all_record


def gui_compute_score_v2_operation_one_score(predict_str: str, ground_truth: str, pred_steps, image_size=None, grounding_version=1) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version=1)
    format_score = gui_format_pred_operation_reward_one(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]
        # print(predict_str, ground_truth, grounding_score)

    if grounding_score is not None:
        if grounding_score > 0.86:
            grounding_acc = 1
        else:
            grounding_acc = 0
    
    # 对pred_steps进行判断
    pred_score = operation_pred_one(predict_str, pred_steps, image_size, grounding_version=1)

    all_record = {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "pred_score": pred_score,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1.86 else 0,
    }

    return all_record


def gui_compute_score_v2_operation_one_select(predict_str: str, ground_truth: str, pred_steps, image_size=None, grounding_version=1) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version=1)
    format_score = gui_format_pred_operation_reward_one(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]
        # print(predict_str, ground_truth, grounding_score)

    if grounding_score is not None:
        if grounding_score > 0.86:
            grounding_acc = 1
        else:
            grounding_acc = 0
    
    # 对pred_steps进行判断
    pred_score = operation_pred_one(predict_str, pred_steps, image_size, grounding_version=1)
    overall_score = 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score
    if step_score["action"] + step_score["value"] >= 1.86:
        overall_score += pred_score

    all_record = {
        "overall":overall_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "pred_score": pred_score,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1.86 else 0,
    }

    return all_record



def gui_compute_score_v2_operation_select(predict_str: str, ground_truth: str, pred_steps, image_size=None, grounding_version=1) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version)
    format_score = gui_format_pred_operation_reward(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]
        # print(predict_str, ground_truth, grounding_score)

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0
    
    # 对pred_steps进行判断
    pred_score = operation_pred(predict_str, pred_steps, image_size)
    overall_score = 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score
    if step_score["action"] + step_score["value"] >= 1.86:
        overall_score += pred_score

    all_record = {
        "overall": overall_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "pred_score": pred_score,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }

    return all_record




def gui_compute_score_operation_only_format(predict_str: str, ground_truth: str, image_size) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version=1)
    format = gui_format_pred_operation_reward_one(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.86:
            grounding_acc = 1
        else:
            grounding_acc = 0

    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format,
        "format_score": format,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1.86 else 0,
    }





## intention format只表示有这个格式，而不涉及计算score
def gui_compute_score_determ_intention_score(predict_str: str, ground_truth: str, image_size, gt_intention, grounding_version=1) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size,  grounding_version)
    format_score = gui_format_reward_intention(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.86:
            grounding_acc = 1
        else:
            grounding_acc = 0
    
    intention_score = calc_similarity_score_intention(predict_str, gt_intention)

    return {
        "overall": 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score + 0.2 * intention_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "intention_score": intention_score,
        "grounding_acc": grounding_acc,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1.86 else 0,
    }

def gui_compute_score_determ_intention(predict_str: str, ground_truth: str, image_size, gt_intention, grounding_version=1) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version)
    format_score = gui_format_reward_intention(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.86:
            grounding_acc = 1
        else:
            grounding_acc = 0
    # 对intention_steps进行判断
    intention_score = calc_similarity_score_intention(predict_str, gt_intention)

    if intention_score > 0.5 and (step_score["action"] + step_score["value"]) > 1.86:
        overall_score = (step_score["action"] + step_score["value"]) + 0.5 * format_score
    else:
        overall_score = 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score

    return {
        "overall": overall_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "intention_score": intention_score,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1.86 else 0,
    }





def gui_compute_score_determ_intention_syntax(predict_str: str, ground_truth: str, image_size, gt_intention, grounding_version, only_format, score_version) -> Dict[str, float]:
    """Compute overall score with format and accuracy components."""
    step_score = gui_accuracy_reward_grounding(predict_str, ground_truth, image_size, grounding_version)
    format_score = gui_format_reward_intention(predict_str)
    grounding_score = None
    grounding_acc = None
    type_score, action_type = step_score["action"], step_score["type"]
    if action_type in ["click", "long_press"]:
        grounding_score: float = step_score["value"]

    if grounding_score is not None:
        if grounding_score > 0.0:
            grounding_acc = 1
        else:
            grounding_acc = 0
    # 相似度阈值
    threshold = 0.5
    # 对intention_steps进行判断
    intention_score = calc_similarity_score_intention_syntax(predict_str, gt_intention)

    # only_format version
    if only_format:
        overall_score = 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score

    if score_version:
        overall_score = 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score + 0.5 * intention_score

    ## select version
    if not only_format and not score_version:
        if intention_score > threshold and (step_score["action"] + step_score["value"]) > 1:
            overall_score = (step_score["action"] + step_score["value"]) + 0.5 * format_score
        else:
            overall_score = 0.5 * (step_score["action"] + step_score["value"]) + 0.5 * format_score

    return {
        "overall": overall_score,
        "format_score": format_score,
        "step_score": step_score["action"] + step_score["value"],
        "grounding_score": grounding_score,
        "type_score_acc": type_score,
        "grounding_acc": grounding_acc,
        "intention_score": intention_score,
        "SSR": 1 if step_score["action"] + step_score["value"] > 1 else 0,
    }












if __name__ == "__main__":
    # predict_str = "<think>Since the goal is to find men's Nike hiking shoes in size 8, I need to use the filter option to narrow down the search results to only show relevant items.</think>\n<action>click(start_box='(47, 226)')</action>\n<think_pred>After clicking the filter button, I will see options to refine my search by selecting specific categories like 'Shoes' and 'Men's'. I'll need to select 'Hiking Shoes' and then choose 'Nike' to focus on the desired brand.</think_pred>\n<next_user_step1>Click on the 'Filter' button to open the filter options.</next_user_step1>\n<next_user_step2>Click on the 'Shoes' category to view available shoe options.</next_user_step2>"
    predict_str = """output: <think>(To achieve the goal, I need to open the 'Category' filter to select the 'Double End Open Jaw Spanner' category. This will allow me to narrow down the search results to the desired product type.)</think>
<action>click(start_box=\'(100,285)\')</action>
<think_pred>(The 'Category' filter is currently open, and I need to select the appropriate category to proceed with the search.)</think_pred>
<next_user_step1>(Click on the 'Category' filter to expand it and view available options.)</next_user_step1>
<next_user_step2>(Select the 'Double End Open Jaw Spanner' category from the list.)</next_user_step2>
"""
    # predict_str = '<think>\nThe current screen shows the Google Photos app with the "Open" button highlighted, indicating that the app is ready to be opened. To proceed with the task of opening the device folders in Google Photos, the next logical step is to tap on the "Open" button to launch the app. Tap on the "Open" button located on the right side of the "Google Photos" app details screen.\n</think><action>click(start_box=\'(696,224)\')</action>'
    ground_truth = "click(start_box='(100,285)')"
    pred_steps = [
            "Select on Double End Open Jaw spanner",
            "Click on Apply "
        ]
    # print(gui_compute_score_v2(predict_str, ground_truth))
    print(gui_compute_score_v2(predict_str, ground_truth, pred_steps))
    # print(gui_accuracy_reward(predict_str, ground_truth))


    # predict_str = '<think>\nThe current screen shows the Google Photos app with the "Open" button highlighted, indicating that the app is ready to be opened. To proceed with the task of opening the device folders in Google Photos, the next logical step is to tap on the "Open" button to launch the app. Tap on the "Open" button located on the right side of the "Google Photos" app details screen.\n</think><action>click(start_box=\'(696,224)\')</action>'
    # ground_truth = "click(start_box='(624,255)')"
    # print(gui_accuracy_reward(predict_str, ground_truth))

    # predict_str = "<action>open_app(app_name='Martinoz Pizza')</action>"
    # ground_truth = "open_app(app_name='Martinoz Pizza')"
    # print(gui_accuracy_reward(predict_str, ground_truth))

    # predict_str = "<action>type(content='Vancouver')</action>"
    # ground_truth = "type(content='Vancouver')"
    # print(gui_accuracy_reward(predict_str, ground_truth))

    # predict_str = "<action>scroll(direction='down')</action>"
    # ground_truth = "scroll(direction='up')"
    # print(gui_accuracy_reward(predict_str, ground_truth))
    # predict_str = "<action>scroll(direction='down')</action>"
    # ground_truth = "scroll(direction='down')"
    # print(gui_accuracy_reward(predict_str, ground_truth))

    # predict_str = "<action>finished()</action>"
    # ground_truth = "finished()"
    # print(gui_accuracy_reward(predict_str, ground_truth))

    # predict_str = "<action>wait()</action>"
    # ground_truth = "wait()"
    # print(gui_accuracy_reward(predict_str, ground_truth))

