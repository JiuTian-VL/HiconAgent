from transformers import PreTrainedTokenizer
from verl.utils.reward_score import math_compute_score, r1v_compute_score
from verl.workers.reward import CustomRewardManager

from mirage.reward.score import gui_compute_score, gui_compute_score_v2, gui_compute_score_format, gui_compute_score_determ, gui_compute_score_determ_format, gui_compute_score_v2_operation, gui_compute_score_determ_grounding, gui_compute_score_determ_intention, gui_compute_score_determ_intention_score
from mirage.reward.score import gui_compute_score_determ_intention_syntax, gui_compute_score_v2_operation_select, gui_compute_score_v2_operation_one_select, gui_compute_score_v2_operation_one_score, gui_compute_score_without_thought, gui_compute_score_operation_only_format
from typing import Any, Callable, Dict, Tuple, TypedDict
from verl.protocol import DataProto
from collections import defaultdict
import torch 
from types import MethodType

class GUICustomRewardManager(CustomRewardManager):
    def __init__(self, tokenizer: PreTrainedTokenizer, compute_score: str):
        self.tokenizer = tokenizer
        self.version = 1
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "gui":
            self.compute_score = gui_compute_score
        elif compute_score == "gui-format":
            self.compute_score = gui_compute_score_format
        elif compute_score == 'gui-determ':
            self.version = 2
            self.compute_score = gui_compute_score_determ
        elif compute_score == 'gui-determ-format':
            self.version = 2
            self.compute_score = gui_compute_score_determ_format
        elif compute_score == "gui-v2":
            self.version = 3
            self.compute_score = gui_compute_score_v2
        elif compute_score == "gui-v2-operation":
            self.version = 3
            self.compute_score = gui_compute_score_v2_operation


        ## grounding系列实验
        elif compute_score == 'gui-determ-grounding-1':
            self.version = 4
            self.grounding_version = 1
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-1-record':
            self.version = 11
            self.grounding_version = 1
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-1-record-v2':
            self.version = 12
            self.grounding_version = 1
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-1-record-v3':
            self.version = 13
            self.grounding_version = 1
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-1-record-v4':
            self.version = 14
            self.grounding_version = 1
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-2':
            self.version = 4
            self.grounding_version = 2
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-3':
            self.version = 4
            self.grounding_version = 3
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-4':
            self.version = 4
            self.grounding_version = 4
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-5':
            self.version = 4
            self.grounding_version = 5
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-6':
            self.version = 4
            self.grounding_version = 6
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-7':
            self.version = 4
            self.grounding_version = 7
            self.compute_score = gui_compute_score_determ_grounding
        elif compute_score == 'gui-determ-grounding-8':
            self.version = 4
            self.grounding_version = 8
            self.compute_score = gui_compute_score_determ_grounding


        ## 基于grounding_version1的后续实验
        elif compute_score == 'gui-determ-intention-syntax-format':
            self.version = 6
            self.grounding_version = 1
            self.only_format = True
            self.score_version = False
            self.compute_score = gui_compute_score_determ_intention_syntax
        elif compute_score == 'gui-determ-intention-syntax-score':
            self.version = 6
            self.grounding_version = 1
            self.only_format = False
            self.score_version = True
            self.compute_score = gui_compute_score_determ_intention_syntax
        elif compute_score == 'gui-determ-intention-syntax-select':
            self.version = 6
            self.grounding_version = 1
            self.only_format = False
            self.score_version = False
            self.compute_score = gui_compute_score_determ_intention_syntax
        elif compute_score == "gui-v2-operation-select":
            self.version = 7
            self.grounding_version = 1
            self.compute_score = gui_compute_score_v2_operation_select
        elif compute_score == "gui-v2-operation-select-one":
            self.version = 7
            self.grounding_version = 1
            self.compute_score = gui_compute_score_v2_operation_one_select
        elif compute_score == "gui-v2-operation-score-one":
            self.version = 7
            self.grounding_version = 1
            self.compute_score = gui_compute_score_v2_operation_one_score
        elif compute_score == 'gui-determ-intention':
            self.version = 5
            self.compute_score = gui_compute_score_determ_intention
        elif compute_score == 'gui-determ-intention-score':
            self.version = 5
            self.compute_score = gui_compute_score_determ_intention_score
        elif compute_score == "gui-without-thought":
            self.version = 2
            self.compute_score = gui_compute_score_without_thought
        elif compute_score == "gui-v2-operation-only-format":
            self.version = 2
            self.compute_score = gui_compute_score_operation_only_format



        else:
            raise NotImplementedError(f"Unknown compute_score type: {compute_score}")

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if self.version == 1:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]

                score = self.compute_score(response_str, ground_truth)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

            return reward_tensor, reward_metrics
        
        elif self.version == 2:
            ## 多传一个image_size参数
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]

                score = self.compute_score(response_str, ground_truth, image_size)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

            return reward_tensor, reward_metrics

        elif self.version==3:
            ## 多传image_size和pred_steps
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)

            for i in range(len(data)):
                data_item = data[i]
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                pred_steps = data_item.non_tensor_batch["pred_steps"]
                image_size = data_item.non_tensor_batch["image_size"]

                score = self.compute_score(response_str, ground_truth, pred_steps, image_size)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
            return reward_tensor, reward_metrics

        elif self.version == 4:
            ## 多传一个image_size参数，和self.grounding
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]

                score = self.compute_score(response_str, ground_truth, image_size, self.grounding_version)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

            return reward_tensor, reward_metrics

        elif self.version == 5:
            ## 多传一个image_size参数，以及intention
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]
                intention = data_item.non_tensor_batch["intention"]

                score = self.compute_score(response_str, ground_truth, image_size, intention)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

            return reward_tensor, reward_metrics

        elif self.version == 6:
            ## 多传一个image_size参数，和self.grounding, intention
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]
                intention = data_item.non_tensor_batch["intention"]

                score = self.compute_score(response_str, ground_truth, image_size, intention, self.grounding_version, self.only_format, self.score_version)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

            return reward_tensor, reward_metrics

        elif self.version == 7:
            ## 多传一个image_size参数，和self.grounding, pred_steps
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]
                pred_steps = data_item.non_tensor_batch["pred_steps"]

                score = self.compute_score(response_str, ground_truth, pred_steps, image_size, self.grounding_version)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

            return reward_tensor, reward_metrics

        elif self.version == 11:
            ## 多传一个image_size参数，和self.grounding
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            his_info = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]                
                # here we calculate the history len in each batch
                if i % 16 == 0:
                    cur_his_info = {}
                    cur_his_info['his_len'] = []
                    cur_his_info['reward'] = []
                cur_his_info['his_len'].append((data_item.batch["input_ids"] == 151652).sum().cpu().item() - 1)
                

                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]

                score = self.compute_score(response_str, ground_truth, image_size, self.grounding_version)
                cur_his_info['reward'].append(score['overall'])

                ## here we add the logic of dynamic rewards
                if (cur_his_info['his_len'][-1] < max(cur_his_info['his_len'])) and (score["overall"] >= 1.43):
                    # print("cur_his_info['his_len'][-1]", cur_his_info['his_len'][-1])
                    # print("max(cur_his_info['his_len'])", max(cur_his_info['his_len']))
                    # print(score["overall"])
                    reward_tensor[i, valid_response_length - 1] = score["overall"] * 1.2
                else:
                    reward_tensor[i, valid_response_length - 1] = score["overall"]

                # reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
                
                if i % 8 == 0 and i != 0:
                    cur_his_info['action'] = response_str
                    his_info.append(cur_his_info)
                # 最后一个也要补上
            his_info.append(cur_his_info)
                
            return reward_tensor, reward_metrics, his_info


        elif self.version == 12:
            ## 多传一个image_size参数，和self.grounding
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            his_info = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]                
                # here we calculate the history len in each batch
                if i % 8 == 0:
                    cur_his_info = {}
                    cur_his_info['his_len'] = []
                    cur_his_info['reward'] = []
                # print(data_item.batch["his_lens"])
                cur_his_info['his_len'].append(data_item.batch["his_lens"][0].item())
                
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]

                score = self.compute_score(response_str, ground_truth, image_size, self.grounding_version)
                cur_his_info['reward'].append(score['overall'])

                ## here we add the logic of dynamic rewards
                if (cur_his_info['his_len'][-1] < max(cur_his_info['his_len'])) and (score["overall"] >= 1.43):
                    # print("cur_his_info['his_len'][-1]", cur_his_info['his_len'][-1])
                    # print("max(cur_his_info['his_len'])", max(cur_his_info['his_len']))
                    # print(score["overall"])
                    reward_tensor[i, valid_response_length - 1] = score["overall"] * 1.2
                else:
                    reward_tensor[i, valid_response_length - 1] = score["overall"]

                # reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
                
                if i % 8 == 0 and i != 0:
                    cur_his_info['action'] = ground_truth
                    his_info.append(cur_his_info)
                # 最后一个也要补上
            his_info.append(cur_his_info)
                
            return reward_tensor, reward_metrics, his_info

        elif self.version == 13:
            ## 多传一个image_size参数，和self.grounding
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            his_info = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]                
                # here we calculate the history len in each batch
                if i % 8 == 0:
                    cur_his_info = {}
                    cur_his_info['his_len'] = []
                    cur_his_info['reward'] = []
                # print(data_item.batch["his_lens"])
                cur_his_info['his_len'].append(data_item.batch["his_lens"][0].item())
                
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["ground_truth"]
                image_size = data_item.non_tensor_batch["image_size"]

                score = self.compute_score(response_str, ground_truth, image_size, self.grounding_version)
                cur_his_info['reward'].append(score['overall'])

                ## here we add the logic of dynamic rewards
                if (cur_his_info['his_len'][-1] < max(cur_his_info['his_len'])) and (score["overall"] >= 1.43):
                    # print("cur_his_info['his_len'][-1]", cur_his_info['his_len'][-1])
                    # print("max(cur_his_info['his_len'])", max(cur_his_info['his_len']))
                    # print(score["overall"])
                    reward_tensor[i, valid_response_length - 1] = score["overall"] * 1.2
                else:
                    reward_tensor[i, valid_response_length - 1] = score["overall"]

                # reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
                
                if i % 8 == 0 and i != 0:
                    cur_his_info['action'] = ground_truth
                    his_info.append(cur_his_info)
                # 最后一个也要补上
            his_info.append(cur_his_info)
                
            return reward_tensor, reward_metrics, his_info
        

        elif self.version == 14:
            ## 多传一个image_size参数，和self.grounding
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_metrics = defaultdict(list)
            his_info = []
            cur_step = data.meta_info['global_step']
            if cur_step <= 46:
                for i in range(len(data)):
                    data_item = data[i]  # DataProtoItem
                    response_ids = data_item.batch["responses"]
                    response_mask = data_item.batch["response_mask"]                
                    # here we calculate the history len in each batch
                    if i % 16 == 0:
                        cur_his_info = {}
                        cur_his_info['his_len'] = []
                        cur_his_info['reward'] = []
                    # print(data_item.batch["his_lens"])
                    cur_his_info['his_len'].append(data_item.batch["his_lens"][0].item())

                    valid_response_length = response_mask.sum()
                    valid_response_ids = response_ids[:valid_response_length]

                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    ground_truth = data_item.non_tensor_batch["ground_truth"]
                    image_size = data_item.non_tensor_batch["image_size"]

                    score = self.compute_score(response_str, ground_truth, image_size, self.grounding_version)
                    cur_his_info['reward'].append(score['overall'])

                    ## here we add the logic of dynamic rewards
                    if (cur_his_info['his_len'][-1] < max(cur_his_info['his_len'])) and (score["overall"] >= 1.43):
                        # print("cur_his_info['his_len'][-1]", cur_his_info['his_len'][-1])
                        # print("max(cur_his_info['his_len'])", max(cur_his_info['his_len']))
                        # print(score["overall"])
                        reward_tensor[i, valid_response_length - 1] = score["overall"]
                    else:
                        reward_tensor[i, valid_response_length - 1] = score["overall"]

                    # reward_tensor[i, valid_response_length - 1] = score["overall"]
                    for key, value in score.items():
                        reward_metrics[key].append(value)

                    if i % 16 == 0 and i != 0:
                        cur_his_info['action'] = ground_truth
                        his_info.append(cur_his_info)
                    # 最后一个也要补上
                his_info.append(cur_his_info)
            else:
                for i in range(len(data)):
                    data_item = data[i]  # DataProtoItem
                    response_ids = data_item.batch["responses"]
                    response_mask = data_item.batch["response_mask"]
                    valid_response_length = response_mask.sum()
                    valid_response_ids = response_ids[:valid_response_length]
    
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    ground_truth = data_item.non_tensor_batch["ground_truth"]
                    image_size = data_item.non_tensor_batch["image_size"]
    
                    score = self.compute_score(response_str, ground_truth, image_size, self.grounding_version)
                    reward_tensor[i, valid_response_length - 1] = score["overall"]
                    for key, value in score.items():
                        reward_metrics[key].append(value)

            return reward_tensor, reward_metrics, his_info
