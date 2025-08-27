import torch
from megatron.core import parallel_state as mpu
from datetime import timedelta

# 设置默认值，避免在 worker 初始化前访问属性时出错
setattr(mpu, '_LLM_TP_SIZE', None)
setattr(mpu, '_LLM_TP_RANK', None)
setattr(mpu, '_LLM_TP_GROUP', None)
setattr(mpu, '_VISION_TP_SIZE', None)
setattr(mpu, '_VISION_DATA_PARALLEL_GROUP', None)
setattr(mpu, '_VISION_DATA_PARALLEL_RANK', None)
setattr(mpu, '_VISION_DATA_PARALLEL_SIZE', None)

def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def stride_list(input_list, step):
    all_sublists = []
    for start in range(step):
        sublist = input_list[start::step]
        all_sublists.append(sublist)
    return all_sublists

def init_vision_tp(config):
    world_size, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
    tp_size = config.actor.megatron.tensor_model_parallel_size
    pp_size = config.actor.megatron.pipeline_model_parallel_size
    cp_size = config.actor.megatron.context_parallel_size
    vision_tp_size = config.actor.megatron.vision_tensor_model_parallel_size

    assert tp_size % vision_tp_size == 0, \
        "tp_size must be divisible by vision_tp_size, but got tp_size: %d, vision_tp_size: %d" % (tp_size, vision_tp_size)

    dp_size = world_size // (tp_size * pp_size * cp_size)
    decoder_rank_generator = mpu.RankGenerator(
        tp=tp_size,
        ep=1,
        dp=dp_size,
        pp=pp_size,
        cp=cp_size,
        order="tp-cp-ep-dp-pp",
        rank_offset=0,
    )

    pp_first_stage_ranks = set()
    for ranks in decoder_rank_generator.get_ranks("pp"):
        pp_first_stage_ranks.add(ranks[0])

    first_stage_tp_ranks = []
    for ranks in decoder_rank_generator.get_ranks("tp"):
        if set(ranks).issubset(pp_first_stage_ranks):
            first_stage_tp_ranks.append(ranks)

    # build new group for vision tensor parallel
    for ranks in first_stage_tp_ranks:
        for vision_tp_ranks in chunk_list(ranks, vision_tp_size):
            # print(f"vision_tp_ranks:{vision_tp_ranks}")
            group = mpu.create_group(
                vision_tp_ranks,
                timeout=timedelta(minutes=30),
                pg_options=None,
                group_desc='VISION_TENSOR_MODEL_PARALLEL_GROUP',
            )

            # replace the original tensor model parallel group
            if rank in vision_tp_ranks:
                mpu._LLM_TP_SIZE = mpu.get_tensor_model_parallel_world_size()
                mpu._LLM_TP_RANK = mpu.get_tensor_model_parallel_rank()
                mpu._LLM_TP_GROUP = mpu.get_tensor_model_parallel_group()
                mpu._VISION_TP_SIZE = vision_tp_size

                mpu._TENSOR_MODEL_PARALLEL_GROUP = group
                mpu._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = vision_tp_ranks

    # build new group for vision data parallel
    for ranks in first_stage_tp_ranks:
        for vision_dp_ranks in stride_list(ranks, vision_tp_size):
            # print(f"vision_dp_ranks:{vision_dp_ranks}")
            group = mpu.create_group(
                vision_dp_ranks,
                timeout=timedelta(minutes=30),
                pg_options=None,
                group_desc='VISION_DATA_PARALLEL_GROUP',
            )
            if rank in vision_dp_ranks:
                mpu._VISION_DATA_PARALLEL_GROUP = group
                mpu._VISION_DATA_PARALLEL_RANK = torch.distributed.get_rank(group=group)
                mpu._VISION_DATA_PARALLEL_SIZE = len(vision_dp_ranks)

    # for backup
    if rank not in pp_first_stage_ranks:
        mpu._LLM_TP_SIZE = mpu.get_tensor_model_parallel_world_size()
        mpu._LLM_TP_RANK = mpu.get_tensor_model_parallel_rank()
        mpu._LLM_TP_GROUP = mpu.get_tensor_model_parallel_group()
        mpu._VISION_TP_SIZE = vision_tp_size
