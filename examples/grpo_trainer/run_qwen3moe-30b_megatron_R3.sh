set -x

HF_MODEL_PATH=Qwen/Qwen3-30B-A3B
DIST_CKPT_PATH=${DIST_CKPT_PATH}

python scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.rollout.chat_scheduler=verl.workers.rollout.llm_chat_scheduler.NaiveChatCompletionScheduler \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_rl' \
    trainer.experiment_name='onpolicy_gsm8k_qwen3_30b_moe_megatron' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=2000 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_return_routed_experts=True \
    trainer.use_refactor_rollout=True \
    trainer.total_epochs=15 2>&1 | tee log.qwen3moe_30b_a3b_gsm8k

# actor_rollout_ref.rollout.chat_scheduler=verl.workers.rollout.tool_chat_scheduler.ToolChatCompletionScheduler_InternVL
