"""Unit tests for 2D metric Kalman filter and multi-object tracker."""

from __future__ import annotations

from nearfield360.config import TrackingConfig
from nearfield360.tracking.kalman import KalmanFilter2D
from nearfield360.tracking.models import GroundFootprint, TrackState
from nearfield360.tracking.tracker import MultiObjectTracker


def test_kalman_filter_predict_and_update() -> None:
    # Stationary object at (5.0, 2.0)
    kf = KalmanFilter2D(initial_pos=(5.0, 2.0), initial_vel=(1.0, 0.0), dt=0.1)

    # Predict: x moves by 1.0 * 0.1 = 0.1 -> (5.1, 2.0)
    pred_x, pred_y = kf.predict()
    assert abs(pred_x - 5.1) < 1e-4
    assert abs(pred_y - 2.0) < 1e-4

    # Update with measurement at (5.12, 2.02)
    upd_x, upd_y = kf.update((5.12, 2.02))
    assert 5.05 < upd_x < 5.15
    assert 1.95 < upd_y < 2.05

    assert kf.speed > 0.0
    assert kf.position_uncertainty > 0.0


def test_multi_object_tracker_tracks_moving_obstacle() -> None:
    # Vehicle moving longitudinally from x=10.0 towards vehicle at -2.0 m/s
    dt = 0.1
    true_vx = -2.0
    cfg = TrackingConfig(dt=dt, min_hits=3, max_age=3, max_distance=3.0)
    tracker = MultiObjectTracker(config=cfg)

    track_id: int | None = None
    curr_x = 10.0

    for step in range(15):
        curr_x += true_vx * dt
        # Add slight measurement noise
        meas_x = curr_x + (0.02 if step % 2 == 0 else -0.02)
        fp = GroundFootprint(
            center_x=meas_x,
            center_y=0.0,
            width=1.8,
            length=4.5,
            class_id=0,
            class_name="vehicles",
            camera="FV",
            confidence=0.95,
        )

        tracks = tracker.update([fp])
        assert len(tracks) == 1
        obs = tracks[0]

        if track_id is None:
            track_id = obs.track_id
        else:
            # Identity must be preserved across frames
            assert obs.track_id == track_id

        if step < 2:
            assert obs.state == TrackState.TENTATIVE
        else:
            assert obs.state == TrackState.CONFIRMED

    # After 15 steps, velocity estimate should have converged near -2.0 m/s
    assert tracks[0].velocity[0] < -1.5
    assert tracks[0].velocity[0] > -2.5
    assert abs(tracks[0].velocity[1]) < 0.5


def test_multi_object_tracker_distinguishes_multiple_classes() -> None:
    cfg = TrackingConfig(min_hits=1)
    tracker = MultiObjectTracker(config=cfg)

    v_fp = GroundFootprint(
        center_x=5.0,
        center_y=-1.0,
        width=1.8,
        length=4.5,
        class_id=0,
        class_name="vehicles",
        camera="FV",
        confidence=0.9,
    )
    p_fp = GroundFootprint(
        center_x=3.0,
        center_y=1.5,
        width=0.6,
        length=0.6,
        class_id=1,
        class_name="person",
        camera="FV",
        confidence=0.85,
    )

    tracks = tracker.update([v_fp, p_fp])
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2
    classes = {t.class_name for t in tracks}
    assert classes == {"vehicles", "person"}


def test_multi_object_tracker_handles_loss_and_pruning() -> None:
    cfg = TrackingConfig(min_hits=1, max_age=3)
    tracker = MultiObjectTracker(config=cfg)

    fp = GroundFootprint(
        center_x=4.0,
        center_y=0.0,
        width=0.6,
        length=0.6,
        class_id=1,
        class_name="person",
        camera="FV",
        confidence=0.9,
    )

    # Frame 1: Detected
    t1 = tracker.update([fp])
    assert len(t1) == 1
    assert t1[0].state == TrackState.CONFIRMED
    orig_id = t1[0].track_id

    # Frame 2: Missed
    t2 = tracker.update([])
    assert len(t2) == 1
    assert t2[0].state == TrackState.LOST

    # Frame 3: Reacquired
    re_fp = GroundFootprint(
        center_x=4.05,
        center_y=0.0,
        width=0.6,
        length=0.6,
        class_id=1,
        class_name="person",
        camera="FV",
        confidence=0.9,
    )
    t3 = tracker.update([re_fp])
    assert len(t3) == 1
    assert t3[0].state == TrackState.CONFIRMED
    assert t3[0].track_id == orig_id

    # Missed for max_age consecutive frames -> should be pruned
    tracker.update([])
    tracker.update([])
    t_pruned = tracker.update([])
    assert len(t_pruned) == 0
    assert tracker.active_tracks_count == 0
