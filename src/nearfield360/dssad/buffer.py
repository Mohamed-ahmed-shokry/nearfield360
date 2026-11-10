"""High-frequency circular FIFO ring buffer for continuous EDR telemetry buffering."""

from __future__ import annotations

from collections import deque

from nearfield360.dssad.models import TelemetrySnapshot


class TelemetryRingBuffer:
    """Fixed-capacity circular FIFO buffer for continuous vehicle telemetry streams.

    Enables microsecond-precision temporal slicing for pre-crash and post-crash forensic
    investigations required by UN Regulation No. 160.
    """

    def __init__(self, capacity: int = 1500) -> None:
        if capacity <= 0:
            raise ValueError(f"Ring buffer capacity must be strictly positive, got {capacity}")
        self._capacity = capacity
        self._buffer: deque[TelemetrySnapshot] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Maximum number of telemetry snapshots retained in buffer."""
        return self._capacity

    def append(self, snapshot: TelemetrySnapshot) -> None:
        """Append a new telemetry snapshot, evicting the oldest element if full."""
        self._buffer.append(snapshot)

    def extend(self, snapshots: list[TelemetrySnapshot]) -> None:
        """Extend buffer with multiple snapshots in chronological order."""
        for s in snapshots:
            self.append(s)

    def latest(self) -> TelemetrySnapshot | None:
        """Return the most recently recorded telemetry sample, or None if empty."""
        return self._buffer[-1] if self._buffer else None

    def oldest(self) -> TelemetrySnapshot | None:
        """Return the oldest stored telemetry sample, or None if empty."""
        return self._buffer[0] if self._buffer else None

    def all_samples(self) -> list[TelemetrySnapshot]:
        """Return all buffered samples in chronological order."""
        return list(self._buffer)

    def __len__(self) -> int:
        """Return the current number of buffered telemetry snapshots."""
        return len(self._buffer)

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self._buffer.clear()

    def get_window(self, start_ns: int, end_ns: int) -> list[TelemetrySnapshot]:
        """Retrieve all telemetry samples within the inclusive time interval [start_ns, end_ns]."""
        if start_ns > end_ns:
            return []
        return [s for s in self._buffer if start_ns <= s.timestamp_ns <= end_ns]

    def get_pre_post_window(
        self, trigger_ns: int, pre_duration_s: float, post_duration_s: float
    ) -> tuple[list[TelemetrySnapshot], list[TelemetrySnapshot]]:
        """Slice telemetry into pre-trigger and post-trigger windows around trigger_ns.

        Args:
            trigger_ns: Monotonic timestamp in nanoseconds of the incident trigger.
            pre_duration_s: Lookback duration in seconds (e.g. 5.0 s).
            post_duration_s: Post-trigger recording window in seconds (e.g. 2.0 s).

        Returns:
            Tuple of (pre_trigger_telemetry, post_trigger_telemetry).
        """
        pre_start_ns = trigger_ns - int(pre_duration_s * 1e9)
        post_end_ns = trigger_ns + int(post_duration_s * 1e9)

        pre_telemetry: list[TelemetrySnapshot] = []
        post_telemetry: list[TelemetrySnapshot] = []

        for sample in self._buffer:
            if pre_start_ns <= sample.timestamp_ns <= trigger_ns:
                pre_telemetry.append(sample)
            elif trigger_ns < sample.timestamp_ns <= post_end_ns:
                post_telemetry.append(sample)

        return pre_telemetry, post_telemetry
