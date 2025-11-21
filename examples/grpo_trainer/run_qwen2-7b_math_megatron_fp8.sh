set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

rollout_mode="sync"
return_raw_chat="False"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

HOME=/mnt/miaohua/hujiahao1/FP8_training

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path']" # , '$math_train_path']"
test_files="['$gsm8k_test_path']" # , '$math_test_path']"

CODE_DIR=/mnt/afs_rl/yinzixin/workspace/verl_new
# VLLM=/mnt/afs_rl/yinzixin/workspace/vllm-0.9.0-copy-copy

MEGATRON_LM=/mnt/afs_rl/yinzixin/workspace/Megatron-LM

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONPATH=$CODE_DIR:$MEGATRON_LM:$PYTHONPATH
export VERL_VLLM_DISTRIBUTED_BACKEND=ray
export VLLM_LOGGING_LEVEL=DEBUG
export RAY_DEDUP_LOGS=0

vllm_path=$CODE_DIR/vllm_patch/0.10.2
vllm=/opt/conda/lib/python3.11/site-packages/vllm
cp -rv ${vllm_path}/* ${vllm} 

T=$(date +%y%m%d%H%M%S)

project_name=yzx_fp8_train
experiment_name=qwen2_7b_megatron_fp8_train

result_dir=./results/$project_name/$experiment_name
ckpt_dir=$result_dir/checkpoints
rollout_dir=$result_dir/samples
mkdir -p $ckpt_dir
mkdir -p $rollout_dir


python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.rollout_data_dir=$rollout_dir/train \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=tensorwise \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_amax_history_len=1024 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_amax_compute_algo=max \
    +actor_rollout_ref.actor.megatron.override_transformer_config.first_last_layers_bf16=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_at_start_in_bf16=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_at_end_in_bf16=1 \
    trainer.total_epochs=15 2>&1 | tee $result_dir/train-$T.log
