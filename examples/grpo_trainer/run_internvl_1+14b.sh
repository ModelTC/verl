#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

cp ./data/aoss.conf ~/aoss.conf

# ========== 环境变量保持不变 ==========
N_GPUS=8
N_NODES=1
ACTOR_SP_SIZE=2
ROLLOUT_TP_SIZE=2
VLLM_USE_V1=1
VLLM_ATTENTION_BACKEND=FLASH_ATTN
GPU_MEM_UTIL=0.8
DATA_DIR=./data_json
BASE_MODEL=qwen3_14B_vit_1b
RESULT_DIR=results
export EXPERIMENT_NAME=qwen2d5_toolrl_debug
mkdir -p $RESULT_DIR/$EXPERIMENT_NAME

ROOT=verl
export PYTHONUNBUFFERED=16
export PYTHONPATH=$ROOT:$PYTHONPATH
export RAY_DEBUG=legacy

export IMG_CONTEXT_TOKEN_ID=151669

train_files=./train.json
val_files=./test.json

# data.train_files=$DATA_DIR/train_sample_1000.parquet \
# data.val_files=$DATA_DIR/test.parquet \
# ========== 启动训练（无行尾 \） ==========
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.custom_cls.path=$ROOT/verl/utils/dataset/rl_dataset_json_v2.py \
    data.custom_cls.name=RLHFJSONDataset \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=16 \
    data.max_prompt_length=15360 \
    data.max_response_length=1024 \
    data.truncation=left \
    data.filter_overlong_prompts=True \
    data.dataloader_num_workers=0 \
    +data.apply_internvl_preprocess=True \
    +data.internvl_config_file=$BASE_MODEL/config.json \
    +data.max_dynamic_patch=12 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ACTOR_SP_SIZE \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.custom_kwargs.img_context_token_id=${IMG_CONTEXT_TOKEN_ID} \
    +actor_rollout_ref.model.internvl=True \
    +actor_rollout_ref.model.freeze_vit=True \
    +actor_rollout_ref.model.freeze_mlp=True \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["InternVisionModel","InternVisionEncoderLayer","Qwen3ForCausalLM","Qwen3Model","Embedding","Qwen3DecoderLayer"] \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=deepscaler \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=ToolRL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$RESULT_DIR/$EXPERIMENT_NAME/checkpoints \
    trainer.rollout_data_dir=$RESULT_DIR/$EXPERIMENT_NAME/samples \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    2>&1 | tee $RESULT_DIR/$EXPERIMENT_NAME/train.log