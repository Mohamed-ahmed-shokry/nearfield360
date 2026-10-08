"""Unit tests for spatial SlotTracker multi-frame temporal memory."""

from nearfield360.config import MissionConfig
from nearfield360.mission.slot_tracker import SlotTracker
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotApproachPath,
    SlotOccupancyStatus,
)


def _make_slot(
    slot_id: str = "slot_1",
    center: tuple[float, float] = (4.0, 2.5),
    heading: float = 0.0,
    status: SlotOccupancyStatus = SlotOccupancyStatus.VACANT,
    confidence: float = 0.9,
    is_feasible: bool = True,
) -> ParkingSlot:
    corners = (
        ParkingSlotCorner(x=center[0] - 2.0, y=center[1] - 1.2, z=0.0),
        ParkingSlotCorner(x=center[0] - 2.0, y=center[1] + 1.2, z=0.0),
        ParkingSlotCorner(x=center[0] + 2.0, y=center[1] + 1.2, z=0.0),
        ParkingSlotCorner(x=center[0] + 2.0, y=center[1] - 1.2, z=0.0),
    )
    approach = SlotApproachPath(
        entry_point=(center[0] - 3.0, center[1]),
        target_point=center,
        entry_heading_rad=heading,
        maneuver_length_m=3.0,
        clearance_margin_m=0.3,
        is_feasible=is_feasible,
    )
    return ParkingSlot(
        slot_id=slot_id,
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=corners,
        center=center,
        heading_rad=heading,
        width_m=2.4,
        length_m=5.0,
        status=status,
        confidence=confidence,
        approach_path=approach,
    )


def test_slot_tracker_initialization_and_single_frame() -> None:
    config = MissionConfig(slot_confirm_frames=2)
    tracker = SlotTracker(config)

    s1 = _make_slot("s1", center=(3.0, 2.0))
    tracks = tracker.update([s1], frame_index=0)

    assert len(tracks) == 1
    assert tracks[0].track_id == "track_001"
    assert tracks[0].hit_count == 1
    assert tracks[0].miss_count == 0
    assert tracks[0].is_confirmed is False
    assert len(tracker.get_active_tracks(confirmed_only=True)) == 0


def test_slot_tracker_multi_frame_confirmation() -> None:
    config = MissionConfig(slot_confirm_frames=2)
    tracker = SlotTracker(config)

    s1_f0 = _make_slot("s1", center=(3.0, 2.0))
    tracker.update([s1_f0], frame_index=0)

    # Frame 1: slight jitter in detected position
    s1_f1 = _make_slot("s1", center=(3.05, 2.02))
    tracks = tracker.update([s1_f1], frame_index=1)

    assert len(tracks) == 1
    assert tracks[0].hit_count == 2
    assert tracks[0].is_confirmed is True
    # Smoothed center between 3.0 and 3.05
    assert 3.0 < tracks[0].slot.center[0] < 3.05
    assert len(tracker.get_active_tracks(confirmed_only=True)) == 1


def test_slot_tracker_ego_displacement_transformation() -> None:
    config = MissionConfig(slot_confirm_frames=2)
    tracker = SlotTracker(config)

    # Ego vehicle is at 0, observes slot at (5.0, 2.0)
    s1 = _make_slot("s1", center=(5.0, 2.0))
    tracker.update([s1], frame_index=0)

    # Ego vehicle moves forward 1.0m (dx=1.0, dy=0.0, d_theta=0.0)
    # The slot in ego's new frame should be at (4.0, 2.0)
    s1_f1 = _make_slot("s1", center=(4.0, 2.0))
    tracks = tracker.update([s1_f1], frame_index=1, ego_displacement=(1.0, 0.0, 0.0))

    assert len(tracks) == 1
    assert tracks[0].hit_count == 2
    assert abs(tracks[0].slot.center[0] - 4.0) < 0.05


def test_slot_tracker_miss_and_pruning() -> None:
    config = MissionConfig(slot_confirm_frames=1, slot_max_miss_frames=2)
    tracker = SlotTracker(config)

    s1 = _make_slot("s1", center=(4.0, 2.0))
    tracker.update([s1], frame_index=0)
    assert len(tracker.get_active_tracks()) == 1

    # Frame 1: empty detection -> miss 1
    t1 = tracker.update([], frame_index=1)
    assert len(t1) == 1
    assert t1[0].miss_count == 1

    # Frame 2: empty detection -> miss 2
    t2 = tracker.update([], frame_index=2)
    assert len(t2) == 1
    assert t2[0].miss_count == 2

    # Frame 3: empty detection -> miss 3 > max_miss_frames (2) -> pruned
    t3 = tracker.update([], frame_index=3)
    assert len(t3) == 0


def test_slot_tracker_get_best_target_slot() -> None:
    config = MissionConfig(slot_confirm_frames=1)
    tracker = SlotTracker(config)

    # Slot 1: behind vehicle (x = -3.0)
    s1 = _make_slot("s1", center=(-3.0, 2.0), status=SlotOccupancyStatus.VACANT)
    # Slot 2: occupied
    s2 = _make_slot("s2", center=(3.0, -2.0), status=SlotOccupancyStatus.OCCUPIED)
    # Slot 3: vacant ahead with feasible path
    s3 = _make_slot("s3", center=(4.0, 2.5), status=SlotOccupancyStatus.VACANT, is_feasible=True)

    tracker.update([s1, s2, s3], frame_index=0)

    best = tracker.get_best_target_slot(current_pose=(0.0, 0.0, 0.0))
    assert best is not None
    assert best.slot.center == (4.0, 2.5)
    assert best.slot.status == SlotOccupancyStatus.VACANT


def test_slot_tracker_reset() -> None:
    tracker = SlotTracker()
    tracker.update([_make_slot("s1")], frame_index=0)
    assert len(tracker.tracks) == 1

    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker.get_best_target_slot() is None
