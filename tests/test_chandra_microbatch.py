import asyncio

import pytest

from QueryLake.operation_classes.ray_chandra_class import ChandraRequest, _MicroBatcher


@pytest.mark.asyncio
async def test_microbatcher_batches_requests():
    batches = []

    async def process(requests):
        batches.append(len(requests))
        return [req.prompt for req in requests]

    batcher = _MicroBatcher(process, max_batch_size=4, max_batch_wait_ms=10)
    batcher.start()

    async def submit(prompt):
        return await batcher.submit(ChandraRequest(image="img", prompt=prompt, max_new_tokens=1))

    results = await asyncio.gather(
        submit("a"),
        submit("b"),
        submit("c"),
        submit("d"),
    )
    assert results == ["a", "b", "c", "d"]
    assert batches[0] == 4
    await batcher.shutdown()
