set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export SWANLAB_LOG_DIR=swanlog

SYSTEM_PROMPT="You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. You FIRST need to think based on the current image, task, and historical actions. The reasoning process MUST BE enclosed within <think> </think> tags. Then output the action, which MUST BE put in <action> </action> and MUST BE in Action Space. \n## Output Format\n<think>...</think><action>...</action>\n## Action Space\nclick(start_box='(x,y)')\ntype(content='')\nscroll(direction='down or up or right or left')\npress_back()\npress_home()\npress_enter()\nfinished()## Example:\n<think>The user wants to search for shoes. The current screen has a search bar at the top.</think>\n<action>click(start_box=\'(x,y)\')</action>##User Instruction\n"
MODEL_PATH="Qwen2_5VL-modify/"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m mirage.main \
    config=configs/rl/ui_tars_2b_grpo.yaml \
    data.train_files=Minuskid/HiconAgent-AMEX \
    data.val_files=Minuskid/HiconAgent-AMEX \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.padding_free=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.compute_score=gui-determ-grounding-1-record-v2 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=AMEX_HiconAgent \
    trainer.n_gpus_per_node=4 \
    trainer.val_freq=-1 \
    trainer.total_episodes=9 \
    trainer.save_freq=100 \
    trainer.save_limit=5 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.val_before_train=false \
    algorithm.use_compression=true \
    worker.rollout.n=8 \
    worker.rollout.limit_images=3 \