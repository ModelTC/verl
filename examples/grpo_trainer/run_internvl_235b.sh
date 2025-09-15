#!/bin/bash
set -ex

cp $HOME/aoss.conf ~/aoss.conf

# Set default model path if not provided
MODEL_PATH=$HOME/models/internvl_1+235b_tool
train_files=$HOME/data/general_tool_rl_train_v1.0.0.json
val_files=$HOME/data/test.json

CODE_DIR=$HOME/verl

master_ip=$(grep 'master-0' /etc/hosts | head -n 1 | awk '{print $1}')
if [ -z "$master_ip" ]; then
  master_ip="127.0.0.1"
fi
echo "master_ip: ${master_ip}"

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONPATH=$CODE_DIR:$PYTHONPATH
export RAY_DEBUG=legacy
export VERL_VLLM_DISTRIBUTED_BACKEND=ray

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0

PP=8
TP=4
EP=8
ETP=1
CP=2
INFER_TP=8
T=$(date +%y%m%d%H%M%S)

project_name='internvl_1+235b'
experiment_name='baseline'
num_nodes=16

train_batch_size=128
val_before_train=True
max_token_len_per_gpu=$((1024 * 8))

result_dir=$HOME/results/$project_name/$experiment_name
ckpt_dir=$result_dir/checkpoints
rollout_dir=$result_dir/samples
mkdir -p $ckpt_dir
mkdir -p $rollout_dir

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.custom_cls.path=$CODE_DIR/verl/utils/dataset/rl_dataset_json_v2.py \
    data.custom_cls.name=RLHFJSONDataset \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=$((1024 * 14)) \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.image_key=image \
    +data.internvl_config_file=$MODEL_PATH/config.json \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.001 \
    actor_rollout_ref.actor.clip_ratio_high=0.0014 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.cuda_graph_sizes="[128]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization="fp8" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((1024 * 16)) \
    actor_rollout_ref.rollout.response_length=$((1024 * 4)) \
    actor_rollout_ref.rollout.max_model_len=$((1024 * 32)) \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.invalid_token_ids="[151669]" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((1024 * 20)) \
    reward_model.reward_manager=omni \
    reward_model.launch_reward_fn_async=True \
    reward_model.reward_fn_parallel_size=$((${num_nodes} * 4)) \
    +reward_model.reward_kwargs.log_num_round=True \
    +reward_model.reward_kwargs.service_url=http://$master_ip:80 \
    +reward_model.reward_fn_timeout=300 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=$val_before_train \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$num_nodes \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.rollout_data_dir=$rollout_dir/train \
    trainer.validation_data_dir=$rollout_dir/validation \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler_num=$((${num_nodes} * 1)) \
    actor_rollout_ref.rollout.chat_scheduler=verl.workers.rollout.tool_chat_scheduler.ToolChatCompletionScheduler_InternVL \
    +actor_rollout_ref.rollout.chat_scheduler_kwargs="{sandbox_address: $master_ip:80, max_turns: 10, use_data_config: True}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=3 \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=15 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_expert_capacity_factor=1.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo_token \
    trainer.total_epochs=15 2>&1 | tee $result_dir/train-$T.log
