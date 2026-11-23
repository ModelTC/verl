# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate time-sharing of the actor GPU pool before starting Ray workers."""

from omegaconf import OmegaConf, open_dict

from verl.utils.config import omega_conf_to_dataclass


def is_shared_opd_enabled(config):
    return bool(OmegaConf.select(config, "shared_opd.enabled", default=False))


def prepare_config(config):
    if not is_shared_opd_enabled(config):
        return
    enabled = OmegaConf.select(config, "distillation.enabled", default=False)
    if not enabled:
        raise ValueError("Shared OPD requires distillation.enabled=True")
    if not config.trainer.get("use_v1", True):
        raise ValueError("Shared OPD requires trainer.use_v1=True")
    strategy = config.actor_rollout_ref.actor.strategy
    if strategy not in ("fsdp", "fsdp2", "megatron"):
        raise ValueError("Shared OPD supports the FSDP/FSDP2 and Megatron engines")
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    if config.trainer.v1.trainer_mode not in ("sync", "colocate_async"):
        raise ValueError("Shared OPD requires trainer.v1.trainer_mode=sync or colocate_async")
    if OmegaConf.select(config, "shared_opd.teacher_execution", default="separate_engine") != "separate_engine":
        raise ValueError("Shared OPD requires separate_engine teachers")
    if not actor.engine.param_offload or not actor.engine.optimizer_offload:
        raise ValueError("Shared OPD requires actor parameter offload and full optimizer offload")
    rollout = config.actor_rollout_ref.rollout
    if not rollout.free_cache_engine or not rollout.get("enable_sleep_mode", True):
        raise ValueError("Shared OPD requires student free_cache_engine and enable_sleep_mode")
    if rollout.checkpoint_engine.backend != "naive":
        raise ValueError("Shared OPD requires the naive checkpoint engine")
    reward = config.reward.reward_model
    if reward.enable and not reward.enable_resource_pool:
        raise ValueError("Shared OPD requires a separate pool for GPU reward models")
    if config.critic.enable or config.algorithm.adv_estimator == "gae":
        raise ValueError("Shared OPD currently requires an advantage estimator without a critic, e.g. grpo")
    if OmegaConf.select(config, "shared_opd.max_concurrent_requests", default=32) <= 0:
        raise ValueError("shared_opd.max_concurrent_requests must be positive")
    with open_dict(config):
        config.distillation.n_gpus_per_node = config.trainer.n_gpus_per_node
        config.distillation.nnodes = config.trainer.nnodes
    distillation = omega_conf_to_dataclass(config.distillation)
    per_node = config.trainer.n_gpus_per_node
    for teacher in distillation.teacher_models.values():
        width = teacher.per_replica_world_size
        if teacher.num_replicas <= 0 or width <= 0 or (per_node % width and width % per_node):
            raise ValueError(f"Teacher {teacher.key!r} replicas must align to the shared GPU nodes")
        if not teacher.inference.free_cache_engine or not teacher.inference.enable_sleep_mode:
            raise ValueError(f"Shared teacher {teacher.key!r} requires free_cache_engine and enable_sleep_mode")
