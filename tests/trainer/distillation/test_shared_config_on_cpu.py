# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Validate shared OPD configuration before allocating GPU workers."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.experimental.teacher_loop.shared_config import prepare_config
from verl.utils.config import omega_conf_to_dataclass, validate_config


def _config(*overrides):
    directory = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "shared_opd.enabled=true",
                "distillation.enabled=true",
                "distillation.distillation_loss.loss_mode=forward_kl_topk",
                "distillation.distillation_loss.use_task_rewards=false",
                "algorithm.adv_estimator=grpo",
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "actor_rollout_ref.actor.fsdp_config.param_offload=true",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.model.path=/student",
                "distillation.teacher_models.teacher_model.model_path=/teacher",
                *overrides,
            ],
        )


@pytest.mark.parametrize("trainer_mode", ["sync", "colocate_async"])
def test_shared_opd_prepares_teacher_pool(trainer_mode):
    config = _config(f"trainer.v1.trainer_mode={trainer_mode}")
    validate_config(config, use_reference_policy=False, use_critic=False)
    distillation = omega_conf_to_dataclass(config.distillation)
    teacher = distillation.teacher_models["default"]
    assert teacher.world_size == config.trainer.n_gpus_per_node * config.trainer.nnodes
    assert distillation.distillation_loss.loss_settings.use_topk


@pytest.mark.parametrize("trainer_mode", ["sync", "colocate_async"])
def test_shared_rejects_engine_reuse(trainer_mode):
    config = _config(f"trainer.v1.trainer_mode={trainer_mode}", "shared_opd.teacher_execution=reuse_student_engine")
    with pytest.raises(ValueError, match="separate_engine"):
        prepare_config(config)


def test_shared_rejects_resident_optimizer():
    config = _config("actor_rollout_ref.actor.fsdp_config.optimizer_offload=false")
    with pytest.raises(ValueError, match="full optimizer offload"):
        prepare_config(config)


def test_shared_pool_reuses_actor_reservations(monkeypatch):
    from verl.trainer.ppo.utils import Role
    from verl.trainer.ppo.v1 import trainer_base
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

    config = _config()
    prepare_config(config)
    trainer = object.__new__(PPOTrainerSync)
    trainer.config = config
    trainer.shared_opd_enabled = True
    monkeypatch.setattr(trainer_base.ray, "remote", lambda worker: worker)
    trainer._init_resource_pool_mgr()
    manager = trainer.resource_pool_manager
    assert manager.mapping[Role.TeacherModel] == manager.mapping[Role.ActorRollout]
    assert "teacher_pool" not in manager.resource_pool_spec
    assert manager.max_colocate_count == 4


def test_prepare_config_preserves_teacher_context_on_repeated_validation():
    config = _config()
    prepare_config(config)
    first = omega_conf_to_dataclass(config.distillation).teacher_models["default"]
    prepare_config(config)
    second = omega_conf_to_dataclass(config.distillation).teacher_models["default"]
    assert first.inference.prompt_length == second.inference.prompt_length
    assert first.inference.response_length == second.inference.response_length == 1
