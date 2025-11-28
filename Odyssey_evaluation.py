import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
import numpy as np
from qwen_vl_utils import process_vision_info
import json
import os
from PIL import Image
from tqdm import tqdm

IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--model_path', type=str, default="/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2his_2branchforward_3KL_drop_6_KL10_AMEX/global_step_414/actor/huggingface",
                    help='The path where the model is loaded from')
parser.add_argument('--part', type=int, default=0,
                    help='The path where the model is loaded from')
parser.add_argument('--drop_k', type=int, default=3,
                    help='The path where the model is loaded from')
parser.add_argument('--data_path', type=str, default="/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data.json",
                    help='The path where the model is loaded from')

parser.add_argument('--split', type=str, default="random",
                    help='The path where the model is loaded from')

# 解析参数
args = parser.parse_args()

outputs = []
# '/data5/GUIAgent/dataset/mcts_data_set/data_for_check/android_control_reward_model_input_validation_language_realcase.json'
# data_path = '/data/user/yxu916/zhouxurui/GUI-dataset/Autogen/2_data_for_check/android_control_reward_model_input_validation_language_realcase.json'
## base GRPO prompt
SYSTEM_PROMPT="You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. You FIRST need to think based on the current image, task, and historical actions. The reasoning process MUST BE enclosed within <think> </think> tags. Then output the action, which MUST BE put in <action> </action> and MUST BE in Action Space. \n## Output Format\n<think>...</think><action>...</action>\n## Action Space\nclick(start_box='(x,y)')\nlong_press(start_box='(x,y)')\ntype(content='')\nscroll(direction='down or up or right or left')\nimpossible()\npress_back()\npress_home()\npress_recent()\nfinished()## Example:\n<think>(The user wants to search for shoes. The current screen has a search bar at the top.)</think>\n<action>click(start_box=\'(x,y)\')</action>##User Instruction\n"

data_path = args.data_path
processor = AutoProcessor.from_pretrained("/data/user/yxu916/zhouxurui/Qwen2_5VL-modify/Qwen2_5VL")

if args.split == 'random':
    part_num = 3662
else:
    part_num = 2500
    
cur_part_idx = args.part
with open(data_path, 'r') as file:
    all_data = json.load(file)[cur_part_idx*part_num:(cur_part_idx + 1) * part_num]


from transformers import AutoProcessor
from mirage import XYQForConditionalGeneration
# model_path = "/data7/Users/zxr/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/V2_format_attempt_2his_compression/global_step_45/actor/huggingface"
# model_path = "/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/V2_format_attempt_2his_2branch_forward/global_step_45/actor/huggingface"
# model_path = "/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2his_2branchforward_3KL_drop_6_KL10_AMEX/global_step_414/actor/huggingface"

device = 'cuda'
model = XYQForConditionalGeneration.from_pretrained(
    # "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
     args.model_path, torch_dtype=torch.bfloat16, device_map=device
    #"/home/wentao/project/gui_ads/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
)

# from mirage import Qwen2_5_VLForConditionalGeneration
# model_path = "/data7/Users/zxr/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/V2_format_attempt_2his_compression/global_step_45/actor/huggingface"
# device = 'cuda'
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     # "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
#      model_path, torch_dtype=torch.bfloat16, device_map=device
#     #"/home/wentao/project/gui_ads/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
# )

processor = AutoProcessor.from_pretrained("/data/user/yxu916/zhouxurui/Qwen2_5VL-modify/Qwen2_5VL")


def get_image_info(image_path, min_pixel=0, max_pixel=512 * 28 * 28):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixels": min_pixel,
                "max_pixels": max_pixel,
            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]


def generate_grounding(image_path, query):
    # TODO
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": query},
                #  "text": f"{query}\n请告诉我怎么操作同时输出坐标。"},
            ],
        }
    ]

    images = []
    for image_file in image_path:
        images.append(get_image_info(image_file))

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text = text.replace(LLAVA_IMAGE_TOKEN, VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN)
    # print(text)
    # print(images)

    # print(messages)
    # print(image_inputs)
#    resized_w, resized_h = image_inputs[0].size
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False, do_compression=1)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]

    return output_text

answer_labels = []
pred_labels = []

all_results = []
from tqdm import tqdm
for i in tqdm(range(len(all_data))):
    cur_step = all_data[i]
    cur_all_imgs = cur_step['images']
    prompt = cur_step['problem']
    response = generate_grounding(cur_all_imgs, SYSTEM_PROMPT + prompt)
    cur_step['response'] = response
    all_results.append(cur_step)

import os
import json

output_dir = os.path.join(
    '/data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_eval_results',
    args.model_path.split('/')[-4],
    args.split
)
os.makedirs(output_dir, exist_ok=True)  # 创建目录

output_file = os.path.join(output_dir, f'{args.part}.json')
with open(output_file, 'w') as file:
    json.dump(all_results, file, indent=4)
