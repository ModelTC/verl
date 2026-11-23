# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared GPU phases may not finish while teacher RPCs are still running."""

import asyncio

import pytest

from verl.experimental.teacher_loop.shared_teacher import run_teacher_groups


def test_failed_request_drains_siblings_before_sleep():
    async def run():
        events = []
        started = asyncio.Event()

        async def wake(key):
            events.append(("wake", key))

        async def sleep(key):
            events.append(("sleep", key))

        async def compute(key, item):
            if item == 0:
                await started.wait()
                raise RuntimeError("teacher request failed")
            started.set()
            await asyncio.sleep(0)
            events.append(("completed", key))

        with pytest.raises(RuntimeError, match="teacher request failed"):
            await run_teacher_groups(
                {"a": [0, 1]},
                wake=wake,
                sleep=sleep,
                compute=compute,
                max_concurrency=2,
            )
        assert events == [("wake", "a"), ("completed", "a"), ("sleep", "a")]

    asyncio.run(run())


def test_scoring_writes_logprobs_to_original_keys(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import torch

    from verl.experimental.teacher_loop import shared_teacher
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
    from verl.utils import tensordict_utils as tu

    routes = ["math", "code", "math"]
    rows = [
        {
            "input_ids": torch.arange(i + 2),
            "extra_fields": {"teacher_request": {"routing_key": route, "mm_processor_output": [{"sample": i}]}},
        }
        for i, route in enumerate(routes)
    ]
    keys = ["first", "second", "third"]
    batch = SimpleNamespace(keys=keys, partition_id="train")
    monkeypatch.setattr(shared_teacher.tq, "kv_batch_get", lambda **kwargs: tu.list_of_dict_to_tensordict(rows))
    written = {}

    def put(keys, partition_id, fields):
        assert partition_id == "train"
        for index, key in enumerate(keys):
            written[key] = fields["teacher_ids"][index], fields["teacher_logprobs"][index]

    monkeypatch.setattr(shared_teacher.tq, "kv_batch_put", put)
    manager = object.__new__(shared_teacher.SharedTeacherModelManager)
    manager.config = SimpleNamespace(shared_opd=SimpleNamespace(max_concurrent_requests=1))
    manager.wake, manager.sleep = AsyncMock(), AsyncMock()
    client = object.__new__(AsyncTeacherLLMServerManager)
    client.teacher_model_configs = dict.fromkeys(routes)

    async def score(sequence_ids, routing_key, mm_processor_output, **kwargs):
        i = mm_processor_output[0]["sample"]
        assert routing_key == routes[i]
        ids = torch.tensor(sequence_ids).unsqueeze(-1)
        return ids, ids.float() * -0.5

    client.compute_teacher_logprobs_single = score
    manager._scoring_client = client
    metrics = manager.compute_batch(batch)
    assert metrics["shared_opd/math/samples"] == 2
    assert metrics["shared_opd/code/samples"] == 1
    assert manager.wake.await_count == manager.sleep.await_count == 2
    for key, row in zip(keys, rows, strict=True):
        ids, logprobs = written[key]
        torch.testing.assert_close(ids[:, 0], row["input_ids"])
        torch.testing.assert_close(logprobs[:, 0], row["input_ids"] * -0.5)
