# # sleep 1h
# # sleep 11h
# # sleep 9h
# python /data/user/yxu916/zhouxurui/HiconAgent-R1/scripts/model_merger.py --local_dir  /data/user/yxu916/zhouxurui/HiconAgent-R1-goat/checkpoints/MirageR1/AMEX_dynamic_reward_n_8_3his_sampling_group_supply/global_step_368/actor
# MODEL_PATH=/data/user/yxu916/zhouxurui/HiconAgent-R1-goat/checkpoints/MirageR1/AMEX_dynamic_reward_n_8_3his_sampling_group_supply/global_step_368/actor/huggingface
# STEP_NUM=368
# CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} & 
# CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM}




# python /data/user/yxu916/zhouxurui/HiconAgent-R1/scripts/model_merger.py --local_dir  /data/user/yxu916/zhouxurui/HiconAgent-R1-goat/checkpoints/MirageR1/AMEX_dynamic_reward_n_8_3his_sampling_group_supply/global_step_322/actor
# MODEL_PATH=/data/user/yxu916/zhouxurui/HiconAgent-R1-goat/checkpoints/MirageR1/AMEX_dynamic_reward_n_8_3his_sampling_group_supply/global_step_322/actor/huggingface
# STEP_NUM=322
# CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} & 
# CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM}

# python /data/user/yxu916/zhouxurui/HiconAgent-R1/scripts/model_merger.py --local_dir  /data/user/yxu916/zhouxurui/HiconAgent-R1-goat/checkpoints/MirageR1/AMEX_dynamic_reward_n_8_3his_sampling_group_supply/global_step_276/actor
# MODEL_PATH=/data/user/yxu916/zhouxurui/HiconAgent-R1-goat/checkpoints/MirageR1/AMEX_dynamic_reward_n_8_3his_sampling_group_supply/global_step_276/actor/huggingface
# STEP_NUM=276
# CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} & 
# CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM}

# sleep 6h
python /data/user/yxu916/zhouxurui/HiconAgent-R1/scripts/model_merger.py --local_dir /data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/AMEX_dynamic_reward_8_fix_inputs_2AI_decrease_ratio_only_first46steps_advantage_positive_KL15/global_step_414/actor

MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/AMEX_dynamic_reward_8_fix_inputs_2AI_decrease_ratio_only_first46steps_advantage_positive_KL15/global_step_414/actor/huggingface
STEP_NUM=414
CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} & 
CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM}


# python /data/user/yxu916/zhouxurui/HiconAgent-R1/scripts/model_merger.py --local_dir  /data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2brachforward_3KL_AMEX_ratio_1_KL10_drop_k_6_detach/global_step_322/actor
# MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2brachforward_3KL_AMEX_ratio_1_KL10_drop_k_6_detach/global_step_322/actor/huggingface
# STEP_NUM=322
# CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} & 
# CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM}


# python /data/user/yxu916/zhouxurui/HiconAgent-R1/scripts/model_merger.py --local_dir  /data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2brachforward_3KL_AMEX_ratio_1_KL10_drop_k_6_detach/global_step_276/actor
# MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2brachforward_3KL_AMEX_ratio_1_KL10_drop_k_6_detach/global_step_276/actor/huggingface
# STEP_NUM=276
# CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} & 
# CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
# CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM}



# CUDA_VISIBLE_DEVICES=4 python AndroidControl_evaluation.py & 
# CUDA_VISIBLE_DEVICES=5 python AndroidControl_evaluation_batch.py &
# CUDA_VISIBLE_DEVICES=6 python AndroidControl_evaluation_batch1.py &
# CUDA_VISIBLE_DEVICES=7 python AndroidControl_evaluation_batch2.py


# sleep 7h
# CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py & 
# CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py &
# CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py &
# CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py

# CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py
