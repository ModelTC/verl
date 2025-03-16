import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from .ray_trainer import (
    RayPPOTrainer, 
    Role, 
    WorkerType, 
    ResourcePoolManager, 
    _timer, 
    _compute_response_info, 
    compute_advantage, 
    reduce_metrics,
    compute_data_metrics,
    compute_timing_metrics
)

import torch

def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )


def expand_idx_to_group(seq_idxs, group_size):
    group_seq_idxs = set()
    for idx in seq_idxs:
        group_idx = idx // group_size
        for i in range(group_idx*group_size, (group_idx+1)*group_size):
            group_seq_idxs.add(i)
    return sorted(list(group_seq_idxs))


def compute_generate_data_metrics(gen_batch):
    response_info = _compute_response_info(gen_batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']
    metrics = {
        # prompt length
        'gen_batch/prompt_length/sum':
            torch.sum(prompt_length).detach().item(),
        'gen_batch/prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'gen_batch/prompt_length/mean':
            torch.mean(prompt_length).detach().item(),

        # response length
        'gen_batch/response_length/sum':
            torch.sum(response_length).detach().item(),
        'gen_batch/response_length/max':
            torch.max(response_length).detach().item(),
        'gen_batch/response_length/mean':
            torch.mean(response_length).detach().item(),
    }
    return metrics

class RayPPOPartialRolloutTrainer(RayPPOTrainer):
    def __init__(self,
             config,
             tokenizer,
             role_worker_mapping: dict[Role, WorkerType],
             resource_pool_manager: ResourcePoolManager,
             ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
             processor=None,
             reward_fn=None,
             val_reward_fn=None):
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn
        )

    def _create_dataloader(self):
        if self.config.trainer.partial_rollout.enable:
            self.max_prompt_length_in_gen = (
                self.config.data.max_prompt_length + 
                self.config.data.max_response_length - 
                self.config.trainer.partial_rollout.max_response_length
            )
            self.max_response_length_in_gen =self.config.trainer.partial_rollout.max_response_length
        else:
            self.max_prompt_length_in_gen = self.config.data.max_prompt_length
            self.max_response_length_in_gen = self.config.data.max_response_length

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', True),
                                         template_key=self.config.data.get('template_key', None),
                                         padding_size=self.max_prompt_length_in_gen)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   # iter manually
                                                   batch_size=1,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', True),
                                       template_key=self.config.data.get('template_key', None))
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _balance_gen_batch(self, batch: DataProto, metrics, logging_prefix='gen_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

        idx_map = {}
        for i, idx in enumerate(global_idx.tolist()):
            idx_map[idx] = i
        
        reorder_idx = []
        for i in range(len(batch)):
            reorder_idx.append(idx_map[i])
        return torch.tensor(reorder_idx)

    def _get_seq_idx_for_partial_rollout(self, batch):
        ## unfinish
        unfinish_mask = (
            (batch.batch['responses'][:, -1] != self.tokenizer.eos_token_id) & 
            (batch.batch['responses'][:, -1] != self.tokenizer.pad_token_id)
        )
        ## unexceed
        response_lengths = batch.batch['attention_mask'].sum(-1) - torch.tensor(batch.non_tensor_batch['prompt_length'].astype(int))
        unexceed_mask = response_lengths < self.config.data.max_response_length
        #TODO: add repeat detection
        # pass
        mask = unfinish_mask & unexceed_mask
        return torch.nonzero(mask, as_tuple=True)[0].tolist()

    def _recompute_batch(self, batch, old_log_probs):
        from torch.nn.utils.rnn import pad_sequence
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length

        prompt_length = torch.tensor(batch.non_tensor_batch['prompt_length'].astype(int))
        prompt_start_idx = (batch.batch['input_ids'] != self.tokenizer.pad_token_id).int().argmax(dim=1)
        prompt_end_idx = prompt_start_idx + prompt_length
        prompts = [batch.batch['input_ids'][i, prompt_start_idx[i] : prompt_end_idx[i]] for i in range(len(batch))]
        prompts = torch.stack(
            [pad_sequence_to_length(prompt, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True) for prompt in prompts]
        )

        resp_length = batch.batch['attention_mask'].sum(-1) - prompt_length
        resp_start_idx = prompt_end_idx
        resp_end_idx = resp_start_idx + resp_length
        # responses = [batch.batch['input_ids'][i, resp_start_idx[i]:] for i in range(len(batch))]
        # responses = pad_sequence(responses, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        responses = [batch.batch['input_ids'][i, resp_start_idx[i] : resp_end_idx[i]] for i in range(len(batch))]
        responses = pad_sequence(responses, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        old_log_probs = pad_sequence(old_log_probs, batch_first=True, padding_value=0.)
        assert responses.shape == old_log_probs.shape, f"get responses.shape:{responses.shape}, old_log_probs.shape:{old_log_probs.shape}"

        prompt_attention_mask = (prompts != self.tokenizer.pad_token_id).long()
        response_attention_mask = get_eos_mask(
            response_id=responses,
            eos_token=self.tokenizer.eos_token_id,
            dtype=torch.int64
        )
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        batch.batch['prompts'] = prompts
        batch.batch['responses'] = responses
        batch.batch['input_ids'] = torch.cat([prompts, responses], dim=-1)
        batch.batch['attention_mask'] = attention_mask
        batch.batch['position_ids'] = compute_position_id_with_mask(batch.batch['attention_mask'])
        batch.batch['old_log_probs'] = old_log_probs
        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        n_samples = self.config.actor_rollout_ref.rollout.n
        self.partial_batch = DataProto()
        self.partial_old_log_probs = []
        for _ in range(self.config.trainer.total_epochs):
            data_iter = iter(self.train_dataloader)
            data_exhausted = False  # Flag to indicate if the iterator is exhausted
            
            while not data_exhausted:
                metrics = {}
                timing_raw = {}

                new_batch = []
                for _ in range(self.config.data.train_batch_size - len(self.partial_batch)//n_samples):
                    try:
                        batch_dict = next(data_iter)
                        del batch_dict['raw_prompt_ids']
                        new_batch.append(DataProto.from_single_dict(batch_dict))
                    except StopIteration:
                        data_exhausted = True

                # If the iterator is exhausted, break the outer while loop as well
                if data_exhausted:
                    print("Data iterator exhausted, breaking the loop.")
                    break

                if len(new_batch) > 0:
                    new_batch = DataProto.concat(new_batch)
                    new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    new_batch.non_tensor_batch['continue_generate'] = np.array([False for _ in range(len(new_batch.batch))], dtype=object)
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                gen_batch = []
                # add data from new batch
                if len(new_batch) > 0:
                    gen_batch.append(new_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids']))

                # add data from partial batch
                idx_in_partial_batch = []
                if len(self.partial_batch) > 0:
                    idx_in_partial_batch = (np.where(self.partial_batch.non_tensor_batch['continue_generate']==True)[0]).tolist()
                    partial_gen_batch = dataprotoitem_to_dataproto(self.partial_batch[idx_in_partial_batch])
                    partial_gen_batch = partial_gen_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    for key in partial_gen_batch.batch.keys():
                        partial_gen_batch.batch[key] = partial_gen_batch.batch[key][:, self.max_response_length_in_gen:]
                    gen_batch.append(partial_gen_batch)
                gen_batch = DataProto.concat(gen_batch)
                # pad to be divisible by dp_size
                gen_batch, padding_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                        
                metrics['batch/partial_rollout_num'] = len(self.partial_batch)
                metrics['batch/continue_generate_num'] = len(idx_in_partial_batch)
                print(f"step: {self.global_steps}, len(new_batch): {len(new_batch)}, len(partial_batch):{len(self.partial_batch)}, ",
                      f"len(continue_gen):{len(idx_in_partial_batch)}, len(gen_batch): {len(gen_batch)}, padding_size:{padding_size}")

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch.meta_info['n'] = 1
                        gen_batch.meta_info['max_tokens'] = self.max_response_length_in_gen
                        reorder_idx = self._balance_gen_batch(gen_batch, metrics)
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        gen_batch_output.reorder(reorder_idx)
                        metrics.update(compute_generate_data_metrics(gen_batch_output))
                    batch = []
                    if len(new_batch) > 0:
                        new_batch_output = gen_batch_output[:len(new_batch)]
                        new_batch = new_batch.union(new_batch_output)
                        batch.append(new_batch)

                    if len(self.partial_batch) > 0 :
                        partial_batch_output = gen_batch_output[len(new_batch): len(new_batch)+len(idx_in_partial_batch)]
                        self.partial_batch[idx_in_partial_batch] = partial_batch_output
                        self.partial_batch.non_tensor_batch['continue_generate'][:] = False
                        batch.append(self.partial_batch)
                    batch = DataProto.concat(batch)

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob_proto = self.actor_rollout_wg.compute_log_prob(batch)

                        if self.config.trainer.partial_rollout.enable:
                            from verl.protocol import union_numpy_dict
                            from verl.utils.py_functional import union_two_dict
                            batch.non_tensor_batch = union_numpy_dict(batch.non_tensor_batch, old_log_prob_proto.non_tensor_batch)
                            batch.meta_info = union_two_dict(batch.meta_info, old_log_prob_proto.meta_info)

                            response_lengths = _compute_response_info(batch)['response_length'].int()
                            old_log_probs = old_log_prob_proto.batch['old_log_probs']
                            old_log_probs = [old_log_probs[i, :response_lengths[i]] for i in range(old_log_probs.shape[0])]

                            if len(self.partial_batch) > 0:
                                for i in range(len(self.partial_batch)):
                                    idx_b = i + len(new_batch)
                                    if i in idx_in_partial_batch:
                                        old_log_probs[idx_b] = torch.cat(
                                            (self.partial_old_log_probs[i], old_log_probs[idx_b])
                                        )
                                    else:
                                        old_log_probs[idx_b] = self.partial_old_log_probs[i]
                        else:
                            batch = batch.union(old_log_prob_proto)

                    # get partial rollout
                    if self.config.trainer.partial_rollout.enable:
                        partial_idxs = self._get_seq_idx_for_partial_rollout(batch)
                        if len(partial_idxs) > 0:
                            batch.non_tensor_batch['continue_generate'][partial_idxs] = True
                            if self.config.algorithm.adv_estimator == "grpo":
                                partial_idxs = expand_idx_to_group(partial_idxs, n_samples)

                        remain_idxs = [i for i in range(len(batch)) if i not in partial_idxs]
                        
                        print(f"step:{self.global_steps}, len(remain_idxs):{len(remain_idxs)}, len(partial_idxs):{len(partial_idxs)}")
                        if len(remain_idxs) < len(batch) * self.config.trainer.partial_rollout.train_num_threshold:
                            partial_idxs = list(range(len(batch)))
                            self.partial_batch = dataprotoitem_to_dataproto(batch[partial_idxs])
                            self.partial_old_log_probs = [old_log_probs[idx] for idx in partial_idxs]
                            continue
                        else:
                            if len(partial_idxs) > 0:
                                self.partial_batch = dataprotoitem_to_dataproto(batch[partial_idxs])
                                self.partial_old_log_probs = [old_log_probs[idx] for idx in partial_idxs]
                            else:
                                self.partial_batch = DataProto()
                                self.partial_old_log_probs = []

                        metrics['batch/train_seq_num'] = len(remain_idxs)
                        metrics['batch/train_new_num'] = len([i for i in remain_idxs if i < len(new_batch)])

                        batch = dataprotoitem_to_dataproto(batch[remain_idxs])
                        old_log_probs = [old_log_probs[idx] for idx in remain_idxs]
                        # reset prompt and response for training
                        batch = self._recompute_batch(batch, old_log_probs)
                        batch, _ = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        with _timer('reward_fn', timing_raw):
                            # reward_tensor, reward_info = self.reward_fn(batch)
                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                        
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_all'] = solve_all

                        # for key in reward_info:
                        #     metrics[f'critic/{key}_reward/mean'] = reward_info[key].mean().item()

                        # if self.config.trainer.rejection_sample:
                        #     # If no valid samples remain, skip this batch and get a new one
                        #     if not valid_mask.any():
                        #         continue

                        #     # Filter batch to keep only valid samples
                        #     batch = batch[valid_mask]
                        #     batch = dataprotoitem_to_dataproto(batch)
                        #     # Round down to the nearest multiple of world size
                        #     num_trainer_replicas = self.actor_rollout_wg.world_size 
                        #     max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                        #     if not max_batch_size:
                        #         # give up, you got everything either all wrong or right.
                        #         continue

                        #     size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
                        #     size_mask[:max_batch_size] = True
                        #     batch = batch[size_mask]
                        #     batch = dataprotoitem_to_dataproto(batch)

                        # # recompute old_log_probs
                        # with _timer('old_log_prob', timing_raw):
                        #     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        #     batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed
                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss
                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
        
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            pprint(f'validation metrics: {val_metrics}')
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
