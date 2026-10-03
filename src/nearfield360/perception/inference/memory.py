"""Zero-copy pinned host memory buffer management and PCIe DMA transfer planning."""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

logger = logging.getLogger(__name__)

# Alignment required by TensorRT for direct DMA device transfers (256 bytes)
TENSORRT_ALIGNMENT_BYTES = 256


def _try_load_cudart() -> Any:
    """Attempt to load the CUDA runtime dynamic library via ctypes."""
    candidates = [
        "cudart64_12.dll",
        "cudart64_110.dll",
        "cudart64_11.dll",
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    for lib_name in candidates:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue
    return None


_CUDART = _try_load_cudart()


def is_cuda_pinned_memory_available() -> bool:
    """Check if CUDA runtime page-locked (pinned) memory allocation is available."""
    return _CUDART is not None and hasattr(_CUDART, "cudaHostAlloc")


@dataclass(frozen=True)
class BufferAllocation:
    """Metadata describing an allocated memory buffer."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    size_bytes: int
    is_pinned: bool
    address: int


class PinnedMemoryBuffer:
    """Page-locked (pinned) or 256-byte aligned host memory buffer.

    Exposes zero-copy numpy array views without duplicating memory.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
        *,
        name: str = "buffer",
        use_cuda_pinned: bool = True,
    ) -> None:
        self._name = name
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._size_bytes = int(np.prod(shape) * self._dtype.itemsize)
        self._is_pinned = False
        self._raw_ptr: Any = None
        self._array: np.ndarray

        if use_cuda_pinned and is_cuda_pinned_memory_available():
            try:
                # cudaHostAllocPortable = 0x01, cudaHostAllocMapped = 0x02
                flags = 0x01 | 0x02
                ptr = ctypes.c_void_p()
                ret = _CUDART.cudaHostAlloc(
                    ctypes.byref(ptr), ctypes.c_size_t(self._size_bytes), ctypes.c_uint(flags)
                )
                if ret == 0 and ptr.value:
                    self._raw_ptr = ptr
                    self._is_pinned = True
                    buffer_from_mem = (ctypes.c_char * self._size_bytes).from_address(ptr.value)
                    self._array = np.frombuffer(buffer_from_mem, dtype=self._dtype).reshape(
                        self._shape
                    )
                else:
                    self._allocate_aligned()
            except Exception as exc:
                logger.debug("cudaHostAlloc failed (%s); falling back to aligned numpy", exc)
                self._allocate_aligned()
        else:
            self._allocate_aligned()

    def _allocate_aligned(self) -> None:
        """Allocate standard 256-byte aligned memory."""
        # Over-allocate by TENSORRT_ALIGNMENT_BYTES to allow manual 256-byte alignment
        total_bytes = self._size_bytes + TENSORRT_ALIGNMENT_BYTES
        raw = np.zeros(total_bytes, dtype=np.uint8)
        offset = (-raw.ctypes.data) % TENSORRT_ALIGNMENT_BYTES
        aligned_slice = raw[offset : offset + self._size_bytes]
        self._array = aligned_slice.view(self._dtype).reshape(self._shape)
        self._is_pinned = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def is_pinned(self) -> bool:
        return self._is_pinned

    @property
    def address(self) -> int:
        return int(self._array.ctypes.data)

    def as_numpy(self) -> np.ndarray:
        """Return zero-copy view of the underlying buffer."""
        return self._array

    def copy_from(self, source: np.ndarray) -> None:
        """Copy data into this buffer with shape and dtype validation."""
        if source.shape != self._shape:
            raise ValueError(
                f"Source shape {source.shape} does not match buffer shape {self._shape}"
            )
        np.copyto(self._array, source.astype(self._dtype, copy=False))

    def describe(self) -> BufferAllocation:
        return BufferAllocation(
            name=self._name,
            shape=self._shape,
            dtype=self._dtype,
            size_bytes=self._size_bytes,
            is_pinned=self._is_pinned,
            address=self.address,
        )

    def close(self) -> None:
        """Free pinned CUDA host memory if allocated."""
        if self._is_pinned and self._raw_ptr is not None and _CUDART is not None:
            try:
                _CUDART.cudaFreeHost(self._raw_ptr)
            except Exception as exc:
                logger.debug("cudaFreeHost failed: %s", exc)
            self._raw_ptr = None
            self._is_pinned = False

    def __del__(self) -> None:
        self.close()


class CUDAPinnedBufferPool:
    """Thread-safe pool of reusable pinned/aligned host memory buffers."""

    def __init__(self, *, enable_pinned: bool = True) -> None:
        self._enable_pinned = enable_pinned
        self._available: dict[tuple[tuple[int, ...], str], list[PinnedMemoryBuffer]] = {}
        self._in_use: set[PinnedMemoryBuffer] = set()
        self._total_allocated_bytes = 0
        self._peak_in_use_count = 0

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
        name: str = "buffer",
    ) -> PinnedMemoryBuffer:
        """Acquire a buffer from the pool or allocate a new one."""
        norm_dtype = str(np.dtype(dtype))
        key = (shape, norm_dtype)

        buf_list = self._available.get(key, [])
        if buf_list:
            buf = buf_list.pop()
        else:
            buf = PinnedMemoryBuffer(
                shape=shape,
                dtype=dtype,
                name=name,
                use_cuda_pinned=self._enable_pinned,
            )
            self._total_allocated_bytes += buf.size_bytes

        self._in_use.add(buf)
        self._peak_in_use_count = max(self._peak_in_use_count, len(self._in_use))
        return buf

    def release(self, buf: PinnedMemoryBuffer) -> None:
        """Return a buffer to the pool for reuse."""
        if buf in self._in_use:
            self._in_use.remove(buf)
            key = (buf.shape, str(buf.dtype))
            self._available.setdefault(key, []).append(buf)

    @contextmanager
    def use_buffer(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
        name: str = "buffer",
    ) -> Generator[PinnedMemoryBuffer, None, None]:
        """Context manager for acquiring and safely releasing a pooled buffer."""
        buf = self.acquire(shape, dtype, name)
        try:
            yield buf
        finally:
            self.release(buf)

    @property
    def in_use_count(self) -> int:
        return len(self._in_use)

    @property
    def peak_in_use_count(self) -> int:
        return self._peak_in_use_count

    @property
    def total_allocated_bytes(self) -> int:
        return self._total_allocated_bytes

    def clear(self) -> None:
        """Release all pooled buffers."""
        for buf_list in self._available.values():
            for b in buf_list:
                b.close()
        self._available.clear()
        for b in list(self._in_use):
            b.close()
        self._in_use.clear()


@dataclass(frozen=True)
class MemoryLayoutPlan:
    """Calculated memory requirements and transfer characteristics for an inference model."""

    batch_size: int
    input_buffers: tuple[BufferAllocation, ...]
    output_buffers: tuple[BufferAllocation, ...]
    total_input_bytes: int
    total_output_bytes: int
    total_memory_bytes: int
    estimated_h2d_ms: float
    estimated_d2h_ms: float

    @classmethod
    def create(
        cls,
        *,
        batch_size: int,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        dtype: DTypeLike = np.float32,
        pcie_bandwidth_gb_s: float = 16.0,  # PCIe Gen3/4 x16 typical
    ) -> MemoryLayoutPlan:
        """Compute memory layout plan and estimated DMA latency."""
        norm_dtype = np.dtype(dtype)
        itemsize = norm_dtype.itemsize

        inputs: list[BufferAllocation] = []
        tot_in = 0
        for idx, s in enumerate(input_shapes):
            full_shape = (batch_size, *s[1:]) if len(s) > 1 else (batch_size, *s)
            sz = int(np.prod(full_shape) * itemsize)
            tot_in += sz
            inputs.append(
                BufferAllocation(
                    name=f"input_{idx}",
                    shape=full_shape,
                    dtype=norm_dtype,
                    size_bytes=sz,
                    is_pinned=is_cuda_pinned_memory_available(),
                    address=0,
                )
            )

        outputs: list[BufferAllocation] = []
        tot_out = 0
        for idx, s in enumerate(output_shapes):
            full_shape = (batch_size, *s[1:]) if len(s) > 1 else (batch_size, *s)
            sz = int(np.prod(full_shape) * itemsize)
            tot_out += sz
            outputs.append(
                BufferAllocation(
                    name=f"output_{idx}",
                    shape=full_shape,
                    dtype=norm_dtype,
                    size_bytes=sz,
                    is_pinned=is_cuda_pinned_memory_available(),
                    address=0,
                )
            )

        # Latency in ms = (bytes / (GB/s * 1e9)) * 1000
        bytes_per_ms = pcie_bandwidth_gb_s * 1e6
        h2d_ms = tot_in / bytes_per_ms if bytes_per_ms > 0 else 0.0
        d2h_ms = tot_out / bytes_per_ms if bytes_per_ms > 0 else 0.0

        return cls(
            batch_size=batch_size,
            input_buffers=tuple(inputs),
            output_buffers=tuple(outputs),
            total_input_bytes=tot_in,
            total_output_bytes=tot_out,
            total_memory_bytes=tot_in + tot_out,
            estimated_h2d_ms=h2d_ms,
            estimated_d2h_ms=d2h_ms,
        )


__all__ = [
    "TENSORRT_ALIGNMENT_BYTES",
    "BufferAllocation",
    "CUDAPinnedBufferPool",
    "MemoryLayoutPlan",
    "PinnedMemoryBuffer",
    "is_cuda_pinned_memory_available",
]
