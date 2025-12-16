# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

import os
from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ..base import BaseRollout
from ..config import RolloutConfig

from transformers import (
    AutoProcessor
)
MODEL_PATH="/data/user/yxu916/zhouxurui/Qwen2_5VL-modify/Qwen2_5VL"

from ....models.transformers.qwen2_vl import get_rope_index


import re
import random
import math
def weighted_discrete_sample(max_hist, k):
    # 计算权重
    weights = [math.exp(k * i) for i in range(max_hist + 1)]
    s = sum(weights)
    probs = [w / s for w in weights]

    # 离散采样
    r = random.random()
    cum_prob = 0.0
    for i, p in enumerate(probs):
        cum_prob += p
        if r < cum_prob:
            return i
    return max_hist  # 理论上不会到这里，防止浮点误差

def truncate_recent_history(text, global_step=None, keep_probs=None):
    """
    Truncate GUI action history, keeping only the last N (random), and rename Image/Step indices.
    Preserve the tail content (e.g., assistant response) after the last Step.
    """
    # 匹配所有 Image 和对应的编号
    image_pattern = r"(Image_(\d+):<\|vision_start\|>.*?<\|vision_end\|>\n?)"
    step_pattern = r"Step_(\d+):.*?\.\n?"

    image_matches = list(re.finditer(image_pattern, text, re.DOTALL))
    step_matches = {int(m.group(1)): m.group(0) for m in re.finditer(step_pattern, text)}

    # 分离出 history blocks（Image + Step）和最后一个 Image（无 Step）
    history_blocks = []
    final_image_block = ""
    final_image_index = -1
    for img_match in image_matches:
        img_idx = int(img_match.group(2))
        img_block = img_match.group(0)
        if img_idx in step_matches:
            step_block = step_matches[img_idx]
            history_blocks.append((img_idx, img_block, step_block))
        else:
            final_image_block = img_block
            final_image_index = img_match.start()

    # 决定保留多少个最近的历史
    max_hist = len(history_blocks)
    if keep_probs is None:
        steepness=2
        total_step=414
        w = min(1.0, 3 * (global_step - 1) / (total_step - 1))

        if w == 0:
            # 完全均匀
            keep_n = random.randint(0, max_hist)
        if w == 1:
            keep_n =  max_hist

        k = steepness * w  # steepness控制陡峭度，可以调节
        keep_n = weighted_discrete_sample(max_hist, k)
    else:
        # options = list(range(max_hist + 1))
        # weights = [keep_probs.get(k, 0) for k in options]
        # s = sum(weights)
        # weights = [w / s for w in weights]
        # keep_n = random.choices(options, weights=weights)[0]
        keep_n = max_hist
        # print(max_hist, keep_n)

    # 只保留最近 keep_n 条历史 + 当前帧
    if keep_n == 0:
        kept = []
    else:
        kept = history_blocks[-keep_n:]  # 从最后面开始选

    # kept = history_blocks[-keep_n:]
    all_blocks = kept + [(None, final_image_block, None)]  # 当前帧无 Step

    # 重命名 Image/Step index 并拼接
    new_blocks = []
    for new_idx, (_, img_block, step_block) in enumerate(all_blocks):
        img_block = re.sub(r"Image_\d+", f"Image_{new_idx}", img_block)
        new_blocks.append(img_block)
        if step_block:
            step_block = re.sub(r"Step_\d+", f"Step_{new_idx}", step_block)
            new_blocks.append(step_block)

    # prefix: 第一张图片之前的 prompt 说明部分
    prefix = text[:image_matches[0].start()]
    # suffix: 当前帧之后的所有内容（包括 assistant 输出等）
    suffix = text[final_image_index + len(final_image_block):]

    return prefix + "".join(new_blocks) + suffix, keep_n

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, List[Any]], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)
import copy
def repeat_dict_list(value, repeats: int):
    """
    对 list[dict] 的每个元素重复 repeats 次，返回新 list，保证每个 dict 是独立副本。
    
    示例：value = [dict0, dict1], repeats = 3
          => [dict0, dict0, dict0, dict1, dict1, dict1]
    """
    return [copy.deepcopy(v) for v in value for _ in range(repeats)]

class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=config.prompt_length + config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **vllm_init_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)
        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

        sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)
        # print("batch_size:", batch_size)
        cur_step_num = prompts.meta_info['global_step']
        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        ## construct inputs for calculating logits
        if cur_step_num <= 46:
            # cur_sampling_n = self.sampling_params.n
            cur_sampling_n = self.sampling_params.n
        else:
            cur_sampling_n = self.sampling_params.n

        if cur_step_num <= 46:
            if self.sampling_params.n > 1:
                batch_size = batch_size * cur_sampling_n
                unchanged_input_ids = _repeat_interleave(input_ids, cur_sampling_n)
                input_ids = _repeat_interleave(input_ids, cur_sampling_n)
                unchanged_attention_mask = _repeat_interleave(attention_mask, cur_sampling_n)
                attention_mask = _repeat_interleave(attention_mask, cur_sampling_n)
                unchanged_position_ids = _repeat_interleave(position_ids, cur_sampling_n)
                position_ids = _repeat_interleave(position_ids, cur_sampling_n)

                # print("non_tensor_batch[multi_modal_inputs]",non_tensor_batch["multi_modal_inputs"])

                if "multi_modal_inputs" in non_tensor_batch.keys():
                    raw_multi_modal_inputs = repeat_dict_list(
                        non_tensor_batch["multi_modal_inputs"], cur_sampling_n
                    )

                padding_id = 151643
                # print("input_ids[0][input_ids[0]!=padding_id].tolist()", input_ids[0][input_ids[0]!=padding_id].tolist())
                all_input_ids = []
                all_attention_mask = []
                all_position_ids = []
                all_raw_input_ids = []
                all_history_n= []
                for i, cur_input_id in enumerate(input_ids):
                    # 先把左填充的padding去除
                    cur_input_id = cur_input_id[cur_input_id!=padding_id]
                    cur_str = self.tokenizer.decode(cur_input_id, skip_special_tokens=False)

                    # here we add the logits of  sampling ratio
                    # print(self.sampling_params.n)
                    # if (i % self.sampling_params.n) < (self.sampling_params.n // 2):
                    #     refined_str, recent_n = truncate_recent_history(cur_str, 1)
                    #     # print("i and recent_n", i , recent_n)
                    # else:
                    #     refined_str, recent_n = truncate_recent_history(cur_str)

                    # all random case
                    refined_str, recent_n = truncate_recent_history(cur_str, cur_step_num)
                    # refined_str, recent_n = truncate_recent_history(cur_str, cur_step_num, 1)

                    all_history_n.append(recent_n)
                    ## TODO: 把string改成raw prompt
                    refined_str_raw = re.sub(r"<\|vision_start\|>.*?<\|vision_end\|>", "<|vision_start|><|image_pad|><|vision_end|>", refined_str)
                    refined_id_raw = self.tokenizer.encode(refined_str_raw, add_special_tokens=False)
                    refined_id_raw = torch.tensor(refined_id_raw).to(input_ids.device)
                    all_raw_input_ids.append(refined_id_raw)

                    refined_id = self.tokenizer.encode(refined_str, add_special_tokens=False)
                    refined_id = torch.tensor(refined_id).to(input_ids.device)
                    # print(non_tensor_batch["multi_modal_inputs"][i]["image_grid_thw"])
                    # print("recent_n", recent_n)
                    cur_image_thw = raw_multi_modal_inputs[i]["image_grid_thw"][-(recent_n + 1):]
                    # print("cur_image_thw:", cur_image_thw)
                    cur_attention_mask = torch.ones_like(refined_id).to(input_ids.device)
                    cur_position_ids = get_rope_index(
                        self.processor,
                        input_ids=refined_id,
                        image_grid_thw=cur_image_thw,
                    )

                    cur_input_ids, cur_attention_mask, cur_position_ids = VF.postprocess_data(
                        input_ids=refined_id,
                        attention_mask=cur_attention_mask,
                        position_ids=cur_position_ids,
                        max_length=2048,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,
                        truncation="right",
                    )

                    all_input_ids.append(cur_input_ids)
                    all_attention_mask.append(cur_attention_mask)
                    all_position_ids.append(cur_position_ids)

                input_ids = torch.stack(all_input_ids).to(input_ids.device)
                attention_mask = torch.stack(all_attention_mask).to(input_ids.device)
                position_ids = torch.stack(all_position_ids).to(input_ids.device)
                raw_prompt_ids = all_raw_input_ids


            ## TODO 把其中的元素全按照all_history_n处理一遍
            ## multi_modal_inputs是image feature，multi_modal_data是image
            if "multi_modal_inputs" in non_tensor_batch.keys():            
                non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                    non_tensor_batch["multi_modal_inputs"], cur_sampling_n
                )

            ## construct vllm inputs for generation 
            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                all_image_data = non_tensor_batch.pop("multi_modal_data")
                # print(all_image_data)
                # print(len(raw_prompt_ids))
                for i in range(len(raw_prompt_ids)):
                    # print("all_image_data", len(all_image_data[i%self.sampling_params.n]['image']))
                    # print("all_history_n[i]", all_history_n[i])
                    cur_mm_data = {'image':all_image_data[i//cur_sampling_n]['image'][-(all_history_n[i] + 1):]}
                    vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids[i]), "multi_modal_data": cur_mm_data})
            else:
                raise RuntimeError


            # print("Start generation")
            with self.update_sampling_params(n=1, **prompts.meta_info):
                completions = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=(self.rank == 0),
                )
                response_ids = [output.token_ids for completion in completions for output in completion.outputs]
                response_ids = VF.pad_2d_list_to_length(
                    response_ids, self.pad_token_id, max_length=self.config.response_length
                ).to(input_ids.device)

            sequence_ids = torch.cat([unchanged_input_ids, response_ids], dim=-1)
            response_length = response_ids.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
            if unchanged_position_ids.dim() == 3:  # qwen2vl mrope
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
            response_position_ids = unchanged_position_ids[..., -1:] + delta_position_id
            unchanged_position_ids = torch.cat([unchanged_position_ids, response_position_ids], dim=-1)
            response_mask = VF.get_eos_mask(
                response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
            )
            unchanged_attention_mask = torch.cat((unchanged_attention_mask, response_mask), dim=-1)

            # all the tp ranks should contain the same data here. data in all ranks are valid
            batch = TensorDict(
                {
                    "prompts": unchanged_input_ids,
                    "responses": response_ids,
                    "input_ids": sequence_ids,  # here input_ids become the whole sentences
                    "attention_mask": unchanged_attention_mask,
                    "response_mask": response_mask,
                    "position_ids": unchanged_position_ids,
                    "his_lens": torch.tensor(all_history_n).reshape(-1, 1)
                },
                batch_size=batch_size,
            )
            non_tensor_batch.pop("raw_prompt_ids")
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        else:    
            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                for raw_prompt_ids, multi_modal_data in zip(
                    non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
                ):
                    vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
            else:
                vllm_inputs = [
                    {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
                ]

            # users can customize different sampling_params at different run
            with self.update_sampling_params(**prompts.meta_info):
                completions: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
                )
                response_ids = [output.token_ids for completion in completions for output in completion.outputs]
                response_ids = VF.pad_2d_list_to_length(
                    response_ids, self.pad_token_id, max_length=self.config.response_length
                ).to(input_ids.device)

                if self.sampling_params.n > 1:
                    batch_size = batch_size * self.sampling_params.n
                    input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                    attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                    position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                    if "multi_modal_inputs" in non_tensor_batch.keys():
                        non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                            non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                        )

            sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
            response_length = response_ids.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
            if position_ids.dim() == 3:  # qwen2vl mrope
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            response_mask = VF.get_eos_mask(
                response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

            # all the tp ranks should contain the same data here. data in all ranks are valid
            batch = TensorDict(
                {
                    "prompts": input_ids,
                    "responses": response_ids,
                    "input_ids": sequence_ids,  # here input_ids become the whole sentences
                    "attention_mask": attention_mask,
                    "response_mask": response_mask,
                    "position_ids": position_ids,
                    "his_lens": torch.ones((input_ids.shape[0], 1)) * 2
                },
                batch_size=batch_size,
            )
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
