import torch
import numpy as np
import uuid
from verl import DataProto
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from torch.nn.utils.rnn import pad_sequence
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import pad_dataproto_to_divisor, DataProtoItem


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

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _balance_gen_batch(batch: DataProto, metrics, logging_prefix='gen_seqlen', world_size=1):
    """Reorder the data on single controller such that each dp rank gets similar total tokens"""
    attention_mask = batch.batch['attention_mask']
    batch_size = attention_mask.shape[0]
    global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
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


def _get_seq_idx_for_partial_rollout(batch, config, tokenizer):
    ## unfinish
    unfinish_mask = (
        (batch.batch['responses'][:, -1] != tokenizer.eos_token_id) & 
        (batch.batch['responses'][:, -1] != tokenizer.pad_token_id)
    )
    ## unexceed
    response_lengths = batch.batch['attention_mask'].sum(-1) - torch.tensor(batch.non_tensor_batch['prompt_length'].astype(int))
    unexceed_mask = response_lengths < config.data.max_response_length
    #TODO: add repeat detection
    # pass
    mask = unfinish_mask & unexceed_mask
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


def _recompute_batch(batch, old_log_probs, config, tokenizer):
    prompt_length = torch.tensor(batch.non_tensor_batch['prompt_length'].astype(int))
    prompt_start_idx = (batch.batch['input_ids'] != tokenizer.pad_token_id).int().argmax(dim=1)
    prompt_end_idx = prompt_start_idx + prompt_length
    prompts = [batch.batch['input_ids'][i, prompt_start_idx[i] : prompt_end_idx[i]] for i in range(len(batch))]
    prompts = torch.stack(
        [pad_sequence_to_length(prompt, config.data.max_prompt_length, tokenizer.pad_token_id, left_pad=True) for prompt in prompts]
    )

    resp_length = batch.batch['attention_mask'].sum(-1) - prompt_length
    resp_start_idx = prompt_end_idx
    resp_end_idx = resp_start_idx + resp_length
    # responses = [batch.batch['input_ids'][i, resp_start_idx[i]:] for i in range(len(batch))]
    # responses = pad_sequence(responses, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    responses = [batch.batch['input_ids'][i, resp_start_idx[i] : resp_end_idx[i]] for i in range(len(batch))]
    responses = pad_sequence(responses, batch_first=True, padding_value=tokenizer.pad_token_id)

    old_log_probs = pad_sequence(old_log_probs, batch_first=True, padding_value=0.)
    assert responses.shape == old_log_probs.shape, f"get responses.shape:{responses.shape}, old_log_probs.shape:{old_log_probs.shape}"

    prompt_attention_mask = (prompts != tokenizer.pad_token_id).long()
    response_attention_mask = get_eos_mask(
        response_id=responses,
        eos_token=tokenizer.eos_token_id,
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


def get_new_batch(data_iter, train_dataloader, config, partial_batch, max_response_length_in_gen, world_size):
    new_batch = []
    gen_batch = []
    for _ in range(config.data.train_batch_size - len(partial_batch) // config.actor_rollout_ref.rollout.n):
        try:
            batch_dict = next(data_iter)
            new_batch.append(DataProto.from_single_dict(batch_dict))
        except StopIteration:
            print("Data iterator exhausted, begin new epoch")
            data_iter = iter(train_dataloader)
            batch_dict = next(data_iter)
            new_batch.append(DataProto.from_single_dict(batch_dict))
    if len(new_batch) > 0:
        new_batch = DataProto.concat(new_batch)
        new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
        new_batch.non_tensor_batch['continue_generate'] = np.array([False for _ in range(len(new_batch.batch))], dtype=object)
        new_batch = new_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
        if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
            gen_batch.append(new_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                            non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs']))
        else:
            gen_batch.append(new_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids']))
            
    idx_in_partial_batch = []
    if len(partial_batch) > 0:
        idx_in_partial_batch = (np.where(partial_batch.non_tensor_batch['continue_generate']==True)[0]).tolist()
        partial_gen_batch = dataprotoitem_to_dataproto(partial_batch[idx_in_partial_batch])
        if 'multi_modal_inputs' in partial_gen_batch.non_tensor_batch.keys():
            partial_gen_batch = partial_gen_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'])
        else:
            partial_gen_batch = partial_gen_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        for key in partial_gen_batch.batch.keys():
            partial_gen_batch.batch[key] = partial_gen_batch.batch[key][:, max_response_length_in_gen:]
        gen_batch.append(partial_gen_batch)
    gen_batch = DataProto.concat(gen_batch)
    # pad to be divisible by dp_size
    gen_batch, padding_size = pad_dataproto_to_divisor(gen_batch, world_size)
    
    return data_iter, gen_batch, new_batch, padding_size, idx_in_partial_batch

def get_partial_old_log_prob(batch, partial_batch, partial_old_log_probs, idx_in_partial_batch, old_log_prob_proto, new_batch):
    from verl.protocol import union_numpy_dict
    from verl.utils.py_functional import union_two_dict
    batch.non_tensor_batch = union_numpy_dict(batch.non_tensor_batch, old_log_prob_proto.non_tensor_batch)
    batch.meta_info = union_two_dict(batch.meta_info, old_log_prob_proto.meta_info)

    response_lengths = _compute_response_info(batch)['response_length'].int()
    old_log_probs = old_log_prob_proto.batch['old_log_probs']
    old_log_probs = [old_log_probs[i, :response_lengths[i]] for i in range(old_log_probs.shape[0])]

    if len(partial_batch) > 0:
        for i in range(len(partial_batch)):
            idx_b = i + len(new_batch)
            if i in idx_in_partial_batch:
                old_log_probs[idx_b] = torch.cat(
                    (partial_old_log_probs[i], old_log_probs[idx_b])
                )
            else:
                old_log_probs[idx_b] = partial_old_log_probs[i]
    return batch, old_log_probs

def get_partial_batch(batch, config, tokenizer, old_log_probs, world_size, global_steps, metrics, new_batch):
    partial_idxs = _get_seq_idx_for_partial_rollout(batch, config, tokenizer)
    if len(partial_idxs) > 0:
        batch.non_tensor_batch['continue_generate'][partial_idxs] = True
        if config.algorithm.adv_estimator == "grpo":
            partial_idxs = expand_idx_to_group(partial_idxs, config.actor_rollout_ref.rollout.n)

    remain_idxs = [i for i in range(len(batch)) if i not in partial_idxs]
    
    print(f"step:{global_steps}, len(remain_idxs):{len(remain_idxs)}, len(partial_idxs):{len(partial_idxs)}")
    if len(remain_idxs) < len(batch) * config.data.train_num_threshold:
        partial_idxs = list(range(len(batch)))
        partial_batch = dataprotoitem_to_dataproto(batch[partial_idxs])
        partial_old_log_probs = [old_log_probs[idx] for idx in partial_idxs]
        return batch, partial_batch, partial_old_log_probs, True
    else:
        if len(partial_idxs) > 0:
            partial_batch = dataprotoitem_to_dataproto(batch[partial_idxs])
            partial_old_log_probs = [old_log_probs[idx] for idx in partial_idxs]
        else:
            partial_batch = DataProto()
            partial_old_log_probs = []

    metrics['batch/train_seq_num'] = len(remain_idxs)
    metrics['batch/train_new_num'] = len([i for i in remain_idxs if i < len(new_batch)])

    batch = dataprotoitem_to_dataproto(batch[remain_idxs])
    old_log_probs = [old_log_probs[idx] for idx in remain_idxs]

    batch = _recompute_batch(batch, old_log_probs, config, tokenizer)
    batch, _ = pad_dataproto_to_divisor(batch, world_size)
    return batch, partial_batch,  partial_old_log_probs, False

def get_solve_all_none_metrics(batch, reward_tensor, reward_info, metrics):
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

    for key in reward_info:
        metrics[f'critic/{key}_reward/mean'] = reward_info[key].mean().item()