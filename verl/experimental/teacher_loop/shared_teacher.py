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
"""Teachers run concurrently in disjoint sub-pools shared with the student."""

import asyncio
import logging
import time
from collections import defaultdict

import transfer_queue as tq
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.experimental.teacher_loop.teacher_model import MultiTeacherModelManager, _run_all
from verl.utils import tensordict_utils as tu
from verl.utils.profiler import marked_timer

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


async def _gather_and_drain(awaitables):
    """Finish every operation before propagating an error or cancellation."""
    pending = asyncio.gather(*awaitables, return_exceptions=True)
    try:
        results = await asyncio.shield(pending)
    except asyncio.CancelledError:
        # HTTP/Ray work may outlive local cancellation. Keep the GPU phase alive
        # until all operations (including teacher cleanup) have completed.
        await pending
        raise
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


async def run_teacher_groups(
    groups,
    *,
    wake,
    compute,
    sleep,
    max_concurrency,
    on_complete=None,
    phase_metrics=None,
):
    """Run teachers concurrently, limiting requests per teacher and draining before sleep.

    When provided, on_complete consumes each group's outputs after it sleeps,
    so completed groups need not retain their tensors until all teachers finish.
    """

    async def run_group(key, requests):
        try:
            await wake(key)
            requests_started = time.perf_counter()
            semaphore = asyncio.Semaphore(max_concurrency)

            async def run_one(request):
                async with semaphore:
                    return await compute(key, request)

            outputs = await _gather_and_drain([run_one(r) for r in requests])
            if phase_metrics is not None:
                phase_metrics[f"shared_opd/{key}/requests_seconds"] = time.perf_counter() - requests_started
        finally:
            await sleep(key)
        if on_complete is not None:
            on_complete(key, outputs)
            return None
        return outputs

    results = await _gather_and_drain([run_group(key, requests) for key, requests in groups.items()])
    return dict(zip(groups, results, strict=True))


class SharedTeacherModelManager(MultiTeacherModelManager):
    def _initialize_teacher_model_managers(self):
        self.active_teachers = set()
        self._phase_metrics = {}
        # The parent partitions the pool and launches all teachers concurrently.
        super()._initialize_teacher_model_managers(parallel_launch=True)
        # Release every teacher's GPU allocations before student rollout starts.
        _run_all([_gather_and_drain([self.sleep(key) for key in self.teacher_model_managers])])
        self._scoring_client = AsyncTeacherLLMServerManager(config=self.config, teacher_client=self.get_client())

    async def wake(self, key):
        if key in self.active_teachers:
            raise RuntimeError(f"Teacher {key!r} is still active")
        self.active_teachers.add(key)
        logger.info("shared_opd wake teacher=%s", key)
        with marked_timer(f"shared_opd/{key}/wake_seconds", self._phase_metrics):
            await _gather_and_drain([r.wake_up() for r in self.teacher_model_managers[key].rollout_replicas])

    async def sleep(self, key):
        with marked_timer(f"shared_opd/{key}/sleep_seconds", self._phase_metrics):
            await _gather_and_drain([r.sleep() for r in self.teacher_model_managers[key].rollout_replicas])
        self.active_teachers.discard(key)
        logger.info("shared_opd sleeping teacher=%s", key)

    def compute_batch(self, batch):
        self._phase_metrics = {}
        read_started = time.perf_counter()
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["input_ids", "extra_fields"]
        )
        self._phase_metrics["shared_opd/tq_read_seconds"] = time.perf_counter() - read_started
        metadata = tu.get(data, "extra_fields")
        groups = defaultdict(list)
        for index, sample_key in enumerate(batch.keys):
            request = metadata[index].get("teacher_request")
            if request is None:
                raise ValueError(f"Missing deferred teacher request for sample {sample_key!r}")
            key = self._scoring_client._resolve_teacher_key(request.get("routing_key"))
            groups[key].append((index, request))

        async def compute(key, item):
            index, request = item
            started = time.perf_counter()
            ids, logprobs = await self._scoring_client.compute_teacher_logprobs_single(
                sequence_ids=data["input_ids"][index].tolist(),
                routing_key=key,
                multi_modal_data=request.get("multi_modal_data"),
                mm_processor_kwargs=request.get("mm_processor_kwargs"),
                mm_processor_output=request.get("mm_processor_output"),
            )
            return batch.keys[index], ids, logprobs, time.perf_counter() - started

        started = time.perf_counter()

        def write(key, outputs):
            keys, ids, logprobs, durations = zip(*outputs, strict=True)
            prefix = f"shared_opd/{key}"
            with marked_timer(f"{prefix}/tq_pack_seconds", self._phase_metrics):
                fields = TensorDict(
                    {
                        "teacher_ids": tu.nested_tensor_from_tensor_list(ids, ragged_idx=1),
                        "teacher_logprobs": tu.nested_tensor_from_tensor_list(logprobs, ragged_idx=1),
                    },
                    batch_size=[len(keys)],
                )
            with marked_timer(f"{prefix}/tq_write_seconds", self._phase_metrics):
                tq.kv_batch_put(keys=list(keys), partition_id=batch.partition_id, fields=fields)
            self._phase_metrics[f"{prefix}/samples"] = len(keys)
            self._phase_metrics[f"{prefix}/seconds"] = time.perf_counter() - started
            self._phase_metrics[f"{prefix}/tq_chunks"] = 1
            self._phase_metrics[f"{prefix}/request/request_seconds/count"] = len(keys)
            self._phase_metrics[f"{prefix}/request/request_seconds/sum"] = sum(durations)
            self._phase_metrics[f"{prefix}/request/request_seconds/mean"] = sum(durations) / len(keys)
            self._phase_metrics[f"{prefix}/request/request_seconds/max"] = max(durations)

        asyncio.run(
            run_teacher_groups(
                groups,
                wake=self.wake,
                compute=compute,
                sleep=self.sleep,
                max_concurrency=self.config.shared_opd.max_concurrent_requests,
                on_complete=write,
                phase_metrics=self._phase_metrics,
            )
        )
        return self._phase_metrics
