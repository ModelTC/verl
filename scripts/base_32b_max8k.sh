#!/bin/bash
set -x

export WANDB_API_KEY=local-f7b45c463750d1ed36dca538a4e57c11d8ef1efc
export WANDB_HOST=http://10.119.16.245:8080
export WANDB_DIR=./
wandb login ${WANDB_API_KEY} --host ${WANDB_HOST}

export PYTHONUNBUFFERED=1

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/root/Qwen2.5-32B"
fi

project_name=$1
experiment_name=$2
train_files=$3
val_files=$4
n_nodes=$5

echo "project_name: $project_name"
echo "experiment_name: $experiment_name"
echo "train_files: $train_files"
echo "val_files: $val_files"

result_dir=./results/$project_name/$experiment_name
mkdir -p $result_dir
mkdir -p $result_dir/checkpoints

T=$(date +%y%m%d%H%M%S)

# verl pythonpath
# example /mnt/afs/yaoyongqiang/open_r1_new/verl_dev
ROOT=verl
# vllm pythonpath
VLLM=/mnt/afs/yaoyongqiang/open_r1_new/python_env
# vllm >=0.7.3 flash-attn is faster
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONPATH=$VLLM:$ROOT:$PYTHONPATH
 
# example /mnt/afs/yaoyongqiang/open_r1_new/deepscaler/deepscaler/rewards/math_reward.py
reward_fn_path=/your/path/math_reward.py
reward_fn_name=deepscaler_reward_fn
reward_manager=deepscaler

# input template
INPUT_TEMPLATE="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {} Please reason step by step, and put your final answer within \\boxed{{}}. Assistant: <think>\n"
# INPUT_TEMPLATE="<｜begin▁of▁sentence｜><｜User｜>{} Let's think step by step and output the final answer within \\boxed{{}}.<｜Assistant｜><think>\n"

# partial example
# trainer.partial_rollout.enable=True \
# trainer.partial_rollout.max_response_length=2048 \
# trainer.partial_rollout.train_num_threshold=0.6 \

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.input_template="\"$INPUT_TEMPLATE\"" \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=$reward_manager \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=$reward_fn_name \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$result_dir/checkpoints \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=30 | tee $result_dir/train-$T.log