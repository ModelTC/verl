import os
import torch
import numpy as np

import json
import blosc


ROUTING_REPLAY = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)
        
    def record(self, top_indices):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def record_bytes(self, bytes):
        self.top_indices_list.append(bytes)
        
    def unpack_bytes(self, bytes):
        top_indices = torch.from_numpy(_unpack_single_sample(bytes)).to(dtype=torch.uint8)
        print("**** **** single route_top_indices unpacking done !!! **** ****")
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        top_indices = buf.to(torch.cuda.current_device())
        return top_indices

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return self.unpack_bytes(top_indices)

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return self.unpack_bytes(top_indices)

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()
    
    @staticmethod
    def clear_all_routing_replays():
        RoutingReplay.all_routing_replays = []


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                print(f"---- replay_forward ----")
                top_indices = ROUTING_REPLAY.pop_forward()
                top_indices = top_indices.to(dtype=torch.int64)
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                top_indices = top_indices.to(dtype=torch.int64)
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)
            
        # def backward_hook(*args, **kwargs):
        #     set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
        # module.register_full_backward_pre_hook(backward_hook)
    

def fill_routing_replay(micro_batches, non_tensor_batch, n_micro_batch, indices, actor_module):
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
    from verl.models.mcore.util import preprocess_packed_seqs_for_r3
    
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    if "route_top_indices" not in non_tensor_batch:
        return
    rollout_routed_experts = non_tensor_batch["route_top_indices"]
    for _ in range(n_micro_batch):
        batch_idxs = indices[_]
        batch = micro_batches[_]
        tmp_rollout_routed_experts = []
        for idx in batch_idxs:
            unpack_r3_idxs = _unpack_single_sample(rollout_routed_experts[idx])
            tmp_rollout_routed_experts.append(torch.from_numpy(unpack_r3_idxs).to(dtype=torch.uint8).cuda())
            # tmp_rollout_routed_experts.append(torch.from_numpy(rollout_routed_experts[idx]).to(dtype=torch.uint8).unsqueeze(0))
        print(f"**** **** route_top_indices {_} unpacking done !!! **** ****")
        # tmp_rollout_routed_experts = torch.cat(tmp_rollout_routed_experts).cuda()
        tokens = batch["input_ids"]
        assert len(tmp_rollout_routed_experts) == len(tokens)
        # for a, b in zip(tmp_rollout_routed_experts, tokens):
        #     assert a.shape[0] == b.shape[0], f"{a.shape}, {b.shape}"
            
        attention_mask = batch['attention_mask']
        route_top_indices_rmpad, _ = preprocess_packed_seqs_for_r3(tmp_rollout_routed_experts, attention_mask, pre_process=True)
        for idx in range(len(tmp_rollout_routed_experts)):
            tmp_rollout_routed_experts[idx] = tmp_rollout_routed_experts[idx].cpu()
        route_top_indices_rmpad = route_top_indices_rmpad.contiguous().squeeze().cpu()
        
        if tp_size > 1:
            seqlen = route_top_indices_rmpad.size(0)
            assert seqlen % tp_size == 0
            start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
            route_top_indices_rmpad = route_top_indices_rmpad[start:end]
            
        if len(RoutingReplay.all_routing_replays) > 0:
            routing_replay_offset = 0
            for _, model in enumerate(actor_module):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config) # , vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config) # , vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    layer_routed_experts = route_top_indices_rmpad[:, layer_id].numpy()
                    comp = _pack_single_sample(layer_routed_experts)
                    RoutingReplay.all_routing_replays[routing_replay_offset].record_bytes(comp)
                    # RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)
            print(f"**** **** route_top_indices {_} packing done !!! **** ****")
        else:
            print("RoutingReplay.all_routing_replays is empty!!!")

def _pack_single_sample(arr: np.ndarray) -> bytes:
    """压缩单个样本（无batch维度）"""
    meta = json.dumps({
        "shape": arr.shape,  # 单个样本的shape
        "dtype": str(arr.dtype)
    }).encode()
    raw = arr.tobytes()
    full_data = meta + b"\xFD\xFC\xFB\xFA" + raw  # 安全分隔符
    
    # 小于2GB直接压缩
    if len(full_data) < 2_100_000_000:
        return blosc.compress(full_data)
    
    # 超大样本分块压缩（1GB/块）
    chunk_size = 1_000_000_000
    chunks = []
    offset = 0
    while offset < len(full_data):
        chunk = full_data[offset:offset + chunk_size]
        chunks.append(blosc.compress(chunk))
        offset += chunk_size
    
    # 长度前缀打包
    packed = b""
    for chunk in chunks:
        packed += len(chunk).to_bytes(4, 'big') + chunk
    return packed

def _unpack_single_sample(payload: bytes) -> np.ndarray:
    """解压单个样本"""
    try:
        # 尝试直接解压（小样本）
        decompressed = blosc.decompress(payload)
        meta_raw, raw = decompressed.split(b"\xFD\xFC\xFB\xFA", 1)
    except:
        # 处理分块压缩的大样本
        chunks = []
        pos = 0
        while pos < len(payload):
            if pos + 4 > len(payload):
                raise ValueError("Incomplete chunk header")
            chunk_len = int.from_bytes(payload[pos:pos+4], 'big')
            pos += 4
            if pos + chunk_len > len(payload):
                raise ValueError("Incomplete chunk data")
            chunks.append(blosc.decompress(payload[pos:pos+chunk_len]))
            pos += chunk_len
        full_data = b"".join(chunks)
        meta_raw, raw = full_data.split(b"\xFD\xFC\xFB\xFA", 1)
    
    meta = json.loads(meta_raw.decode())
    return np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["shape"])

def pack_blosc_batch(arr: np.ndarray) -> list[bytes]:
    """
    将batch数组按第一维度拆分为独立样本分别压缩
    
    参数:
        arr: shape为[batch_size, ...]的numpy数组
        
    返回:
        长度为batch_size的bytes列表，每个元素对应一个样本的压缩数据
    """
    if arr.ndim < 1:
        raise ValueError("Input array must have at least 1 dimension (batch dimension)")
    
    batch_size = arr.shape[0]
    # 按batch维度分割成单个样本
    samples = [arr[i] for i in range(batch_size)]
    
    # 分别压缩每个样本
    compressed_list = []
    for sample in samples:
        compressed_list.append(_pack_single_sample(sample))
    
    return compressed_list

def unpack_blosc_batch(payload_list: list[bytes]) -> np.ndarray:
    """
    将压缩样本列表解压并合并为batch数组
    
    参数:
        payload_list: 长度为batch_size的压缩数据列表
        
    返回:
        shape为[batch_size, ...]的numpy数组
    """
    if not isinstance(payload_list, list):
        raise TypeError("Input must be a list of compressed samples")
    
    batch_size = len(payload_list)
    if batch_size == 0:
        return np.array([])
    
    # 解压所有样本
    samples = []
    sample_shapes = []
    sample_dtypes = []
    
    for payload in payload_list:
        sample = _unpack_single_sample(payload)
        samples.append(sample)
        sample_shapes.append(sample.shape)
        sample_dtypes.append(sample.dtype)
    
    # 验证所有样本shape和dtype一致
    first_shape = sample_shapes[0]
    first_dtype = sample_dtypes[0]
    
    for i, (shape, dtype) in enumerate(zip(sample_shapes, sample_dtypes), 1):
        if shape != first_shape:
            raise ValueError(
                f"Inconsistent sample shapes: sample0={first_shape}, sample{i}={shape}"
            )
        if dtype != first_dtype:
            raise ValueError(
                f"Inconsistent dtypes: sample0={first_dtype}, sample{i}={dtype}"
            )
    
    # 合并为batch数组
    return np.stack(samples, axis=0)