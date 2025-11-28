MODEL_PATH=yourpath/global_step_414/actor/huggingface
CUDA_VISIBLE_DEVICES=0 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 0 & 
CUDA_VISIBLE_DEVICES=1 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 1 &
CUDA_VISIBLE_DEVICES=2 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 2 &
CUDA_VISIBLE_DEVICES=3 python /data/user/yxu916/zhouxurui/GUIAgent-R1/AITW_evaluation.py --model_path ${MODEL_PATH} --part 3 &