"""Tests for `sleap_io.model.diagnostics`."""

import numpy as np
import pytest

from sleap_io import Instance, LabeledFrame, Labels, PredictedInstance, Skeleton
from sleap_io.io.main import load_slp
from sleap_io.model.diagnostics import (
    LOW_CONFIDENCE,
    SPURIOUS_TRACK_OCCUPANCY,
    TELEPORT_FACTOR,
    Diagnostics,
    Finding,
    diagnose,
)
from sleap_io.model.instance import Track
from sleap_io.model.video import Video


def make_labels(
    tracks_xy: dict[str, list[tuple[float, float] | None]],
    node_names: tuple[str, ...] = ("a", "b"),
    scores: float | None = 1.0,
    offset: tuple[float, float] = (0.0, 5.0),
) -> Labels:
    """Build a synthetic tracked `Labels` from per-track centroid paths.

    Args:
        tracks_xy: Maps a track name to a per-frame list of centroid positions,
            with `None` for frames where the track is absent. All lists must be
            the same length.
        node_names: Skeleton node names. Nodes are placed at the centroid plus
            successive multiples of `offset`.
        scores: Point score assigned to every node, or `None` to create user
            instances (which carry no scores).
        offset: Per-node displacement from the centroid.

    Returns:
        A `Labels` with one video and one frame per position in the paths.
    """
    skeleton = Skeleton(
        list(node_names),
        edges=[(node_names[i], node_names[i + 1]) for i in range(len(node_names) - 1)],
    )
    video = Video(filename="synthetic.mp4")
    track_objs = {name: Track(name) for name in tracks_xy}
    n_frames = len(next(iter(tracks_xy.values())))

    frames = []
    for frame_idx in range(n_frames):
        instances = []
        for name, path in tracks_xy.items():
            xy = path[frame_idx]
            if xy is None:
                continue
            points = np.array(
                [[xy[0] + i * offset[0], xy[1] + i * offset[1]] for i in range(2)]
            )[: len(node_names)]
            if scores is None:
                instances.append(
                    Instance.from_numpy(
                        points, skeleton=skeleton, track=track_objs[name]
                    )
                )
            else:
                instances.append(
                    PredictedInstance.from_numpy(
                        points,
                        skeleton=skeleton,
                        point_scores=np.full(len(node_names), scores),
                        score=scores,
                        track=track_objs[name],
                    )
                )
        frames.append(
            LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )

    return Labels(frames)


def straight_path(n_frames: int, step: float = 1.0) -> list[tuple[float, float]]:
    """Return a constant-velocity path of `n_frames` positions."""
    return [(10.0 + i * step, 10.0) for i in range(n_frames)]


def test_diagnose_clean_labels_has_no_findings():
    labels = make_labels({"a": straight_path(20), "b": straight_path(20)})

    report = labels.diagnose()

    assert report.findings == []
    assert report.n_frames == 20
    assert report.n_tracks == 2
    assert report.tracked


def test_diagnose_reports_shape_and_names():
    labels = make_labels({"a": straight_path(5)}, node_names=("head", "tail"))

    report = labels.diagnose()

    assert report.n_nodes == 2
    assert report.node_names == ["head", "tail"]
    assert report.track_occupancy.shape == (1,)
    assert report.node_visibility.shape == (2,)


def test_spurious_tracks_flagged():
    # One track spans every frame; the other appears in a single frame.
    n = 300
    blip: list[tuple[float, float] | None] = [None] * n
    blip[0] = (50.0, 50.0)
    labels = make_labels({"real": straight_path(n), "blip": blip})

    report = labels.diagnose()

    assert report.n_spurious_tracks == 1
    codes = [f.code for f in report.findings]
    assert "spurious_tracks" in codes
    finding = next(f for f in report.findings if f.code == "spurious_tracks")
    assert "1 of 2 tracks" in finding.message
    assert report.track_occupancy.min() < SPURIOUS_TRACK_OCCUPANCY


def test_track_fragmentation_flagged():
    # Present, gone, present again -> one interruption.
    path: list[tuple[float, float] | None] = list(straight_path(9))
    path[3] = None
    path[4] = None
    labels = make_labels({"a": path})

    report = labels.diagnose()

    assert report.track_breaks.sum() == 1
    assert "track_fragmentation" in [f.code for f in report.findings]


def test_teleport_frames_reported():
    path = list(straight_path(30))
    path[15] = (10_000.0, 10_000.0)
    labels = make_labels({"a": path})

    report = labels.diagnose()

    finding = next(f for f in report.findings if f.code == "displacement_outliers")
    # The jump into frame 15 and the jump back out of it both register.
    assert 15 in finding.frames
    assert 16 in finding.frames
    assert report.displacement["max"] > TELEPORT_FACTOR * report.displacement["median"]


def test_low_confidence_flagged():
    labels = make_labels({"a": straight_path(10)}, scores=LOW_CONFIDENCE / 2)

    report = labels.diagnose()

    finding = next(f for f in report.findings if f.code == "low_confidence")
    assert report.confidence["fraction_below"] == 1.0
    assert "100.0%" in finding.message


def test_confidence_is_nan_for_user_instances():
    labels = make_labels({"a": straight_path(10)}, scores=None)

    report = labels.diagnose()

    assert np.isnan(report.confidence["mean"])
    assert np.isnan(report.confidence["fraction_below"])
    assert "low_confidence" not in [f.code for f in report.findings]


def test_low_node_visibility_flagged():
    labels = make_labels({"a": straight_path(10)}, node_names=("head", "tail"))
    # Blank the second node everywhere except the first frame.
    for lf in labels.labeled_frames[1:]:
        lf.instances[0].numpy()  # touch to ensure points are materialized
        lf.instances[0].points["xy"][1] = np.nan

    report = labels.diagnose()

    finding = next(f for f in report.findings if f.code == "low_node_visibility")
    assert "tail" in finding.message
    assert report.node_visibility[1] < report.node_visibility[0]


def test_node_visibility_ignores_absent_instances():
    # A track that is absent for most frames must not drag node visibility down:
    # the denominator is present instances, not track slots.
    n = 10
    blip: list[tuple[float, float] | None] = [None] * n
    blip[0] = (50.0, 50.0)
    labels = make_labels({"real": straight_path(n), "blip": blip})

    report = labels.diagnose()

    assert np.allclose(report.node_visibility, 1.0)
    assert "low_node_visibility" not in [f.code for f in report.findings]


def test_unstable_segments_flagged():
    labels = make_labels({"a": straight_path(20)})
    # Wobble the second node so the single edge's length varies wildly.
    for i, lf in enumerate(labels.labeled_frames):
        lf.instances[0].points["xy"][1, 1] = 10.0 + (50.0 if i % 2 else 1.0)

    report = labels.diagnose()

    finding = next(f for f in report.findings if f.code == "unstable_segments")
    assert "a -- b" in finding.message
    assert report.segment_cv[0] > 0.25


def test_stable_segments_not_flagged():
    labels = make_labels({"a": straight_path(20)})

    report = labels.diagnose()

    assert report.segment_cv.shape == (1,)
    assert report.segment_cv[0] == pytest.approx(0.0)
    assert "unstable_segments" not in [f.code for f in report.findings]


def test_segment_cv_nan_when_edge_never_observed():
    labels = make_labels({"a": straight_path(4)})
    for lf in labels.labeled_frames:
        lf.instances[0].points["xy"][1] = np.nan

    report = labels.diagnose()

    assert np.isnan(report.segment_cv[0])
    assert "unstable_segments" not in [f.code for f in report.findings]


def test_skeleton_without_edges_yields_empty_segments():
    labels = make_labels({"a": straight_path(5)}, node_names=("only",))

    report = labels.diagnose()

    assert report.segment_cv.shape == (0,)
    assert report.segment_names == []


def test_empty_labels_reports_no_data():
    report = Labels().diagnose()

    assert [f.code for f in report.findings] == ["no_data"]
    assert report.findings[0].severity == "error"
    assert report.n_frames == 0
    assert report.to_dict()["n_frames"] == 0


def test_untracked_fallback_skips_track_findings():
    labels = make_labels({"a": straight_path(5)})
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            inst.track = None
    labels.tracks = []

    report = labels.diagnose()

    assert not report.tracked
    assert "spurious_tracks" not in [f.code for f in report.findings]
    assert "track_fragmentation" not in [f.code for f in report.findings]


def test_untracked_can_be_forced():
    labels = make_labels({"a": straight_path(5), "b": straight_path(5)})

    report = labels.diagnose(untracked=True)

    assert not report.tracked
    assert report.n_tracks == 2


def test_single_frame_labels():
    labels = make_labels({"a": straight_path(1)})

    report = labels.diagnose()

    assert report.n_frames == 1
    assert report.displacement == {"median": 0.0, "p99": 0.0, "max": 0.0}
    assert report.outlier_frames == []
    assert report.track_breaks.tolist() == [0]


def test_findings_sorted_most_severe_first():
    report = Diagnostics(
        n_frames=1,
        n_tracks=1,
        n_nodes=1,
        node_names=["a"],
        tracked=True,
        track_occupancy=np.ones(1),
        node_visibility=np.ones(1),
        confidence={"mean": 1.0, "p05": 1.0, "p50": 1.0, "p95": 1.0},
        track_breaks=np.zeros(1, dtype=int),
        displacement={"median": 0.0, "p99": 0.0, "max": 0.0},
        outlier_frames=[],
        segment_cv=np.zeros(0),
        segment_names=[],
        findings=[
            Finding("c", "info", "m", "s"),
            Finding("a", "error", "m", "s"),
            Finding("b", "warning", "m", "s"),
        ],
    )
    report.findings.sort(
        key=lambda f: {"error": 0, "warning": 1, "info": 2}[f.severity]
    )

    assert [f.code for f in report.findings] == ["a", "b", "c"]


def test_finding_str_includes_suggestion_and_frames():
    finding = Finding("code", "warning", "something happened", "do this", frames=[1, 2])

    text = str(finding)

    assert "[warning] something happened" in text
    assert "-> do this" in text
    assert "frames: 1, 2" in text


def test_finding_str_truncates_long_frame_lists():
    finding = Finding("code", "warning", "m", "s", frames=list(range(10)))

    text = str(finding)

    assert "0, 1, 2, 3, 4, ... (+5)" in text


def test_finding_str_without_frames_has_no_frame_line():
    assert "frames:" not in str(Finding("code", "info", "m", "s"))


def test_summary_and_str_agree():
    labels = make_labels({"a": straight_path(5)})

    report = labels.diagnose()

    assert str(report) == report.summary()
    assert "No issues found." in report.summary()


def test_summary_omits_confidence_for_user_instances():
    labels = make_labels({"a": straight_path(5)}, scores=None)

    summary = labels.diagnose().summary()

    assert "confidence:" not in summary
    assert "displacement px/frame:" in summary


def test_to_dict_is_json_serializable():
    import json

    labels = make_labels({"a": straight_path(5)})

    payload = labels.diagnose().to_dict()

    assert json.loads(json.dumps(payload))["n_frames"] == 5
    assert payload["node_names"] == ["a", "b"]
    assert isinstance(payload["track_occupancy"], list)


def test_diagnose_function_matches_method(centered_pair):
    labels = load_slp(centered_pair)

    assert diagnose(labels).to_dict() == labels.diagnose().to_dict()


def test_diagnose_on_real_predictions(centered_pair):
    labels = load_slp(centered_pair)

    report = labels.diagnose()

    # 27 tracks for a two-fly recording: the spurious-track finding is the
    # headline result on this fixture.
    codes = [f.code for f in report.findings]
    assert "spurious_tracks" in codes
    assert report.n_spurious_tracks > 0
    assert report.confidence["mean"] > 0
    assert len(report.segment_names) == len(report.segment_cv)


def test_summary_renders_findings():
    path = list(straight_path(30))
    path[15] = (10_000.0, 10_000.0)

    summary = make_labels({"a": path}).diagnose().summary()

    assert "No issues found." not in summary
    assert "[warning]" in summary
    assert "frames:" in summary


def test_segment_cv_nan_for_zero_length_edge():
    # Both nodes sit at the same coordinate, so the edge has no length to vary.
    labels = make_labels({"a": straight_path(10)}, offset=(0.0, 0.0))

    report = labels.diagnose()

    assert np.isnan(report.segment_cv[0])
    assert "unstable_segments" not in [f.code for f in report.findings]
