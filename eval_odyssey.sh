# sleep 5h
# MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2brachforward_3KL_AMEX_ratio_1_KL10_drop_k_6_detach_positive_fix/global_step_414/actor/huggingface

# MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/h100_2brachforward_3KL_AMEX_ratio_1_KL10_drop_k_6_detach_student_ratio/global_step_414/actor/huggingface
# DATA_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data.json
# CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 0 --split random --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 1 --split random --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 2 --split random --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 3 --split random --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=4 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 4 --split random --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=5 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 5 --split random --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=6 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 6 --split random --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=7 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 7 --split random --data_path ${DATA_PATH}

MODEL_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/checkpoints/MirageR1/AMEX_dynamic_reward_8_fix_inputs_2AI_decrease_ratio_only_first46steps_run2/global_step_414/actor/huggingface
DATA_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data_app.json
CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 0 --split app --data_path ${DATA_PATH}& 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 1 --split app --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 2 --split app --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 3 --split app --data_path ${DATA_PATH}


CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 4 --split app --data_path ${DATA_PATH}& 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 5 --split app --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 6 --split app --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 7 --split app --data_path ${DATA_PATH}

DATA_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data_device.json
CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 0 --split device --data_path ${DATA_PATH}& 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 1 --split device --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 2 --split device --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 3 --split device --data_path ${DATA_PATH}


CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 4 --split device --data_path ${DATA_PATH}& 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 5 --split device --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 6 --split device --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 7 --split device --data_path ${DATA_PATH}

DATA_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data_task.json
CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 0 --split task --data_path ${DATA_PATH}& 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 1 --split task --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 2 --split task --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 3 --split task --data_path ${DATA_PATH}


CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 4 --split task --data_path ${DATA_PATH}& 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 5 --split task --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 6 --split task --data_path ${DATA_PATH}&
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 7 --split task --data_path ${DATA_PATH}

# DATA_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data_app.json
# CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 0 --split app --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 1 --split app --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 2 --split app --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 3 --split app --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=4 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 4 --split app --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=5 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 5 --split app --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=6 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 6 --split app --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=7 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 7 --split app --data_path ${DATA_PATH}


# DATA_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data_device.json
# CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 0 --split device --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 1 --split device --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 2 --split device --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 3 --split device --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=4 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 4 --split device --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=5 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 5 --split device --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=6 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 6 --split device --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=7 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 7 --split device --data_path ${DATA_PATH}


# DATA_PATH=/data/user/yxu916/zhouxurui/GUIAgent-R1/data_process/Odyssey/test_data_task.json
# CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 0 --split task --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 1 --split task --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 2 --split task --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 3 --split task --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=4 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 4 --split task --data_path ${DATA_PATH}& 
# CUDA_VISIBLE_DEVICES=5 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 5 --split task --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=6 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 6 --split task --data_path ${DATA_PATH}&
# CUDA_VISIBLE_DEVICES=7 python /data/user/yxu916/zhouxurui/GUIAgent-R1/Odyssey_evaluation.py --model_path ${MODEL_PATH} --part 7 --split task --data_path ${DATA_PATH}
