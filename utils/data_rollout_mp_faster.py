import argparse
import json
import os
from multiprocessing import Process
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

# —— 以下常量保持不变 —— 
IGNORE_INDEX = -100
MAX_PIXELS = 512*28*28
MIN_PIXELS = 0

# GRPO_SYSTEM_PROMPT = "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. You FIRST need to think based on the current image, task, and historical actions. The reasoning process MUST BE enclosed within <think> </think> tags. Then output the action, which MUST BE put in <action> </action> and MUST BE in Action Space. \n## Output Format\n<think>...</think><action>...</action>\n## Action Space\nclick(start_box='(x,y)')\nlong_press(start_box='(x,y)')\ntype(content='')\nscroll(direction='down or up or right or left')\nopen_app(app_name='')\npress_back()\npress_home()\nwait()\nfinished()## Example:\n<think>The user wants to search for shoes. The current screen has a search bar at the top.</think>\n<action>click(start_box=\'(x,y)\')</action>##User Instruction\n"

GRPO_SYSTEM_PROMPT = "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. You FIRST need to think based on the current image, task, and historical actions. The reasoning process MUST BE enclosed within <think> </think> tags. Then output the action, which MUST BE put in <action> </action> and MUST BE in Action Space. \n## Output Format\n<think>...</think><action>...</action>\n## Action Space\nclick(start_box='(x,y)')\nlong_press(start_box='(x,y)')\ntype(content='')\nscroll(direction='down or up or right or left')\nopen_app(app_name='')\npress_back()\npress_home()\nwait()\nfinished()## Example:\n<think>(The user wants to search for shoes. The current screen has a search bar at the top.)</think>\n<action>click(start_box=\'(x,y)\')</action>##User Instruction\n"
class GropoDataset(Dataset):
    def __init__(self, data_list, img_dir, model_path, system_prompt, min_pixels, max_pixels):
        self.data = data_list
        self.img_dir = img_dir
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.system_prompt = system_prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        step = self.data[idx]
        # 1) 读图路径
        img_name = os.path.basename(step["images"][0])
        img_path = os.path.join(self.img_dir, img_name)
        # 2) 构建 prompt 文本
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img_path, 
                 "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                {"type": "text",  "text": self.system_prompt + step["problem"]},
            ]},
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 3) 视觉预处理
        image_inputs, _ = process_vision_info(messages)
        # 4) 返回模型输入和原始 step，用于后续拼结果
        llm_input = {"prompt": prompt_text, "multi_modal_data": {"image": image_inputs}}
        return step, llm_input

# —— Collate 函数 —— 
def collate_fn(batch):
    # batch: List of (step, llm_input)
    steps, inputs = zip(*batch)
    return steps, list(inputs)

# 处理并发推理的函数
def worker(proc_id, num_procs, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(proc_id)
    from vllm import LLM, SamplingParams
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f"results_{proc_id}.json")

    # 1) 加载数据，分割子集
    with open(args.input_json, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    sub_data = [step for idx, step in enumerate(all_data) if idx % num_procs == proc_id]

    # 2) 恢复断点
    if os.path.exists(out_path):
        done = json.load(open(out_path, 'r', encoding='utf-8'))
        start_idx = len(done)
    else:
        done, start_idx = [], 0

    # 3) 构建 Dataset + DataLoader
    dataset = GropoDataset(
        sub_data[start_idx:], args.img_dir, args.model_path,
        GRPO_SYSTEM_PROMPT, MIN_PIXELS, MAX_PIXELS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,            # I/O 并行进程数，根据 CPU 核心调
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True   # 持久 worker，避免每个 epoch 重启
    )

    # 4) 初始化模型
    llm = LLM(model=args.model_path,
              limit_mm_per_prompt={"image":10, "video":10}, dtype="bfloat16")
    sampling_params = SamplingParams(
        temperature=0.0,         # 设置为0，完全确定性
        top_p=1.0,               # 保证保留全部概率质量
        top_k=-1,                # 不截断前k个token，默认就是-1
        repetition_penalty=1.0,  # 无惩罚
        max_tokens=512,
        stop_token_ids=[]
    )

    # 5) 批量推理
    results = done[:]  # 先填入已完成部分
    count = start_idx
    for steps, llm_inputs in tqdm(dataloader):
        # vLLM 接受多个输入列表
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
        for step, out in zip(steps, outputs):
            results.append({**step, "response": out.outputs[0].text})
        count += len(steps)

        # 定期保存
        if count % args.save_interval == 0:
            with open(out_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, indent=4, ensure_ascii=False)

    # 完成后再存一次
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)
    print(f"[GPU{proc_id}] Done, total {len(results)} items → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True,
                        help="原始样本 JSON 文件")
    # /data/GUIAgent/zhouxurui/GUI-dataset/AndroidControl/Qwen2-5VL/RL_sequence_low_level_4A_with_finish_test_all.json
    parser.add_argument("--img_dir",    type=str, required=True,
                        help="图片文件夹路径")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Qwen2-5-VL 模型路径")
    parser.add_argument("--num_gpu",    type=int, default=8,
                        help="要并行的 GPU/进程 数量")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="每次 generate 调用的 batch 大小")
    parser.add_argument("--save_dir",   type=str, default="./eval_results",
                        help="输出结果存放目录")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="每 N 条保存一次")
    args = parser.parse_args()
    
    # 启动多个子进程
    procs = []
    for pid in range(args.num_gpu):
        p = Process(target=worker, args=(pid, args.num_gpu, args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    
    print("All workers finished.")