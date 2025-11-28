MODEL_PATH=yourpath/global_step_414/actor/huggingface
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