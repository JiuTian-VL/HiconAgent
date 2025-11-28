python /data/user/yxu916/zhouxurui/HiconAgent-R1/scripts/model_merger.py --local_dir /data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/AMEX_dynamic_reward_8_fix_inputs_2AI_decrease_ratio_only_first46steps_advantage_positive_KL15/global_step_414/actor

MODEL_PATH=yourpath/global_step_414/actor/huggingface
STEP_NUM=414
CUDA_VISIBLE_DEVICES=0 python AndroidControl_evaluation.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} & 
CUDA_VISIBLE_DEVICES=1 python AndroidControl_evaluation_batch.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
CUDA_VISIBLE_DEVICES=2 python AndroidControl_evaluation_batch1.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM} &
CUDA_VISIBLE_DEVICES=3 python AndroidControl_evaluation_batch2.py --model_path ${MODEL_PATH} --step_num ${STEP_NUM}
