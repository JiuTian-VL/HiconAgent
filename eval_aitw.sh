# sleep 5h
# MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2brachforward_3KL_AMEX_ratio_1_KL10_drop_k_6_detach/global_step_414/actor/huggingface

# CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 0 & 
# CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 1 &
# CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 2 &
# CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 3 &

# MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_baseline_AMEX/global_step_414/actor/huggingface

# CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 0 & 
# CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 1 &
# CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 2 &
# CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 3 &
# python
MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/AMEX_dynamic_reward_16_fix_inputs_2AI_decrease_ratio_only_first46steps/global_step_414/actor/huggingface

CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 0 & 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 1 &
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 2 &
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 3 &