import torch
from verl.experimental.rollout.rollout_manager import Sample

__all__ = ["filter_reward_nonzero_std"]


def filter_reward_nonzero_std(samples: list[Sample]) -> bool:
    rewards = [sample.reward_score for sample in samples]
    is_filter = not (torch.tensor(rewards, dtype=torch.float).std() > 0.0)
    return is_filter
