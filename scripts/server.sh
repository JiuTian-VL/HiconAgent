#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 vllm serve checkpoints/easy_r1/qwen2_5_vl_3b_geo/global_step_55/actor/huggingface --port 9555 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5

# CUDA_VISIBLE_DEVICES=6 vllm serve /data7/Users/xyq/developer/RL4GUI/mirage/checkpoints/MirageR1/ui_tars2b_test/global_step_32/actor/huggingface --port 9555 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5
CUDA_VISIBLE_DEVICES=6 vllm serve models/UI-TARS-2B-SFT --port 9555 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5