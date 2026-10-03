from __future__ import annotations

import numpy as np
import pytest

from nearfield360.perception.inference.memory import (
    TENSORRT_ALIGNMENT_BYTES,
    CUDAPinnedBufferPool,
    MemoryLayoutPlan,
    PinnedMemoryBuffer,
    is_cuda_pinned_memory_available,
)


def test_is_cuda_pinned_memory_available() -> None:
    res = is_cuda_pinned_memory_available()
    assert isinstance(res, bool)


def test_pinned_memory_buffer_alignment_and_views() -> None:
    shape = (1, 3, 480, 640)
    buf = PinnedMemoryBuffer(shape=shape, dtype=np.float32, name="cam_front")

    assert buf.name == "cam_front"
    assert buf.shape == shape
    assert buf.dtype == np.dtype(np.float32)
    assert buf.size_bytes == 1 * 3 * 480 * 640 * 4

    # Direct DMA alignment assertion
    assert buf.address % TENSORRT_ALIGNMENT_BYTES == 0

    arr = buf.as_numpy()
    assert arr.shape == shape
    assert arr.dtype == np.float32

    # Zero-copy verification: mutating numpy view updates underlying memory
    arr[0, 0, 0, 0] = 42.5
    assert buf.as_numpy()[0, 0, 0, 0] == 42.5

    # Description
    desc = buf.describe()
    assert desc.name == "cam_front"
    assert desc.size_bytes == buf.size_bytes
    assert desc.shape == shape

    buf.close()


def test_pinned_memory_buffer_copy_from() -> None:
    buf = PinnedMemoryBuffer(shape=(2, 4), dtype=np.float32)
    source = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

    buf.copy_from(source)
    np.testing.assert_array_equal(buf.as_numpy(), source)

    # Shape mismatch error
    with pytest.raises(ValueError, match="does not match buffer shape"):
        buf.copy_from(np.ones((2, 5), dtype=np.float32))

    buf.close()


def test_cuda_pinned_buffer_pool_reuse() -> None:
    pool = CUDAPinnedBufferPool()
    shape = (1, 3, 64, 64)

    buf1 = pool.acquire(shape, dtype=np.float32, name="input1")
    addr1 = buf1.address
    assert pool.in_use_count == 1
    assert pool.peak_in_use_count == 1

    pool.release(buf1)
    assert pool.in_use_count == 0

    # Acquiring identical signature should reuse the existing buffer
    buf2 = pool.acquire(shape, dtype=np.float32, name="input2")
    assert buf2.address == addr1
    assert pool.in_use_count == 1

    pool.release(buf2)
    pool.clear()
    assert pool.in_use_count == 0


def test_cuda_pinned_buffer_pool_context_manager() -> None:
    pool = CUDAPinnedBufferPool()
    shape = (4, 10)

    with pool.use_buffer(shape, dtype=np.int32, name="boxes") as buf:
        assert pool.in_use_count == 1
        arr = buf.as_numpy()
        arr.fill(7)
        assert buf.as_numpy()[0, 0] == 7

    assert pool.in_use_count == 0
    pool.clear()


def test_memory_layout_plan() -> None:
    in_shapes = ((1, 3, 480, 640),)
    out_shapes = ((1, 10, 480, 640), (1, 10, 4))

    plan = MemoryLayoutPlan.create(
        batch_size=4,
        input_shapes=in_shapes,
        output_shapes=out_shapes,
        dtype=np.float32,
        pcie_bandwidth_gb_s=16.0,
    )

    assert plan.batch_size == 4
    # 4 * 3 * 480 * 640 * 4 = 14,745,600 bytes
    expected_in = 4 * 3 * 480 * 640 * 4
    assert plan.total_input_bytes == expected_in
    assert plan.total_memory_bytes == plan.total_input_bytes + plan.total_output_bytes
    assert plan.estimated_h2d_ms > 0.0
    assert plan.estimated_d2h_ms > 0.0
    assert len(plan.input_buffers) == 1
    assert len(plan.output_buffers) == 2
    assert plan.input_buffers[0].shape == (4, 3, 480, 640)
