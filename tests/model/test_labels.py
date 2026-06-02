"""Test methods and functions in the sleap_io.model.labels file."""

import copy
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from shapely.geometry import box

import sleap_io
from sleap_io import (
    Instance,
    LabeledFrame,
    PredictedInstance,
    RecordingSession,
    Skeleton,
    SuggestionFrame,
    Track,
    Video,
    load_slp,
    save_slp,
)
from sleap_io.io.video_reading import CropVideoBackend
from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.centroid import PredictedCentroid, UserCentroid
from sleap_io.model.label_image import (
    LabelImage,
    PredictedLabelImage,
    UserLabelImage,
)
from sleap_io.model.labels import Labels
from sleap_io.model.mask import PredictedSegmentationMask, UserSegmentationMask
from sleap_io.model.matching import (
    InstanceMatcher,
    InstanceMatchMethod,
    SkeletonMatcher,
    SkeletonMatchMethod,
    SkeletonMismatchError,
    TrackMatcher,
    TrackMatchMethod,
    VideoMatcher,
    VideoMatchMethod,
)
from sleap_io.model.roi import PredictedROI, UserROI


def test_labels():
    """Test methods in the `Labels` data structure."""
    skel = Skeleton(["A", "B"])
    labels = Labels(
        [
            LabeledFrame(
                video=Video(filename="test"),
                frame_idx=0,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel),
                    PredictedInstance([[4, 5], [6, 7]], skeleton=skel),
                ],
            )
        ]
    )

    assert len(labels) == 1
    assert type(labels[0]) is LabeledFrame
    assert labels[0].frame_idx == 0

    with pytest.raises(IndexError):
        labels[None]

    # Test Labels.__iter__ method
    for lf_idx, lf in enumerate(labels):
        assert lf == labels[lf_idx]

    assert str(labels) == (
        "Labels(labeled_frames=1, videos=1, skeletons=1, tracks=0, suggestions=0, "
        "sessions=0)"
    )


def test_update(slp_real_data):
    base_labels = load_slp(slp_real_data)

    labels = Labels(base_labels.labeled_frames)
    assert len(labels.videos) == len(base_labels.videos) == 1
    assert len(labels.tracks) == len(base_labels.tracks) == 0
    assert len(labels.skeletons) == len(base_labels.skeletons) == 1

    new_video = Video.from_filename("fake.mp4")
    labels.suggestions.append(SuggestionFrame(video=new_video, frame_idx=0))

    new_track = Track("new_track")
    labels[0][0].track = new_track

    new_skel = Skeleton(["A", "B"])
    new_video2 = Video.from_filename("fake2.mp4")
    labels.append(
        LabeledFrame(
            video=new_video2,
            frame_idx=0,
            instances=[
                Instance.from_numpy(np.array([[0, 1], [2, 3]]), skeleton=new_skel)
            ],
        ),
        update=False,
    )

    labels.update()
    assert new_video in labels.videos
    assert new_video2 in labels.videos
    assert new_track in labels.tracks
    assert new_skel in labels.skeletons


def test_append_extend():
    labels = Labels()

    new_skel = Skeleton(["A", "B"])
    new_video = Video.from_filename("fake.mp4")
    new_track = Track("new_track")
    labels.append(
        LabeledFrame(
            video=new_video,
            frame_idx=0,
            instances=[
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=new_skel, track=new_track
                )
            ],
        ),
        update=True,
    )
    assert labels.videos == [new_video]
    assert labels.skeletons == [new_skel]
    assert labels.tracks == [new_track]

    new_video2 = Video.from_filename("fake.mp4")
    new_skel2 = Skeleton(["A", "B", "C"])
    new_track2 = Track("new_track2")
    labels.extend(
        [
            LabeledFrame(
                video=new_video,
                frame_idx=1,
                instances=[
                    Instance.from_numpy(
                        np.array([[0, 1], [2, 3]]), skeleton=new_skel, track=new_track2
                    )
                ],
            ),
            LabeledFrame(
                video=new_video2,
                frame_idx=0,
                instances=[
                    Instance.from_numpy(
                        np.array([[0, 1], [2, 3], [4, 5]]), skeleton=new_skel2
                    )
                ],
            ),
        ],
        update=True,
    )

    assert labels.videos == [new_video, new_video2]
    assert labels.skeletons == [new_skel, new_skel2]
    assert labels.tracks == [new_track, new_track2]


def test_update_dedupes_same_order_skeletons():
    """Structurally-equal, same-order skeletons collapse to one on update."""
    skel1 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    skel2 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    assert skel1 is not skel2

    video = Video.from_filename("fake.mp4")
    inst1 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel1)
    inst2 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel2)
    lf1 = LabeledFrame(video=video, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video, frame_idx=1, instances=[inst2])

    labels = Labels(labeled_frames=[lf1, lf2])

    assert len(labels.skeletons) == 1
    canonical = labels.skeletons[0]
    assert inst1.skeleton is canonical
    assert inst2.skeleton is canonical


def test_append_dedupes_same_order_skeletons():
    """`append` deduplicates structurally-equal, same-order skeletons."""
    skel1 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    skel2 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])

    video = Video.from_filename("fake.mp4")
    inst1 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel1)
    inst2 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel2)

    labels = Labels()
    labels.append(LabeledFrame(video=video, frame_idx=0, instances=[inst1]))
    labels.append(LabeledFrame(video=video, frame_idx=1, instances=[inst2]))

    assert len(labels.skeletons) == 1
    canonical = labels.skeletons[0]
    assert inst1.skeleton is canonical
    assert inst2.skeleton is canonical


def test_extend_dedupes_same_order_skeletons():
    """`extend` deduplicates structurally-equal, same-order skeletons."""
    skel1 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    skel2 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])

    video = Video.from_filename("fake.mp4")
    inst1 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel1)
    inst2 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel2)

    labels = Labels()
    labels.extend(
        [
            LabeledFrame(video=video, frame_idx=0, instances=[inst1]),
            LabeledFrame(video=video, frame_idx=1, instances=[inst2]),
        ]
    )

    assert len(labels.skeletons) == 1
    canonical = labels.skeletons[0]
    assert inst1.skeleton is canonical
    assert inst2.skeleton is canonical


def test_dedup_preserves_per_node_xy():
    """Reassigning to a same-order canonical skeleton moves no point data."""
    skel1 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    skel2 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])

    video = Video.from_filename("fake.mp4")
    inst1 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel1)
    inst2 = Instance.from_numpy(np.array([[3, 3], [4, 4], [5, 5]]), skeleton=skel2)

    before = {n: inst2[n]["xy"].copy() for n in ["A", "B", "C"]}
    orig = inst2.numpy().copy()

    labels = Labels(
        labeled_frames=[
            LabeledFrame(video=video, frame_idx=0, instances=[inst1]),
            LabeledFrame(video=video, frame_idx=1, instances=[inst2]),
        ]
    )

    assert len(labels.skeletons) == 1
    canonical = labels.skeletons[0]
    assert inst2.skeleton is canonical
    for node in ["A", "B", "C"]:
        assert_equal(inst2[node]["xy"], before[node])
    assert_equal(inst2.numpy(), orig)
    assert inst2.points["name"].tolist() == canonical.node_names


def test_dedup_predicted_instance():
    """Dedup preserves xy, per-node scores and instance score for predictions."""
    skel1 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    skel2 = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])

    video = Video.from_filename("fake.mp4")
    inst1 = Instance.from_numpy(np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel1)
    pred = PredictedInstance.from_numpy(
        points_data=np.array([[3, 3], [4, 4], [5, 5]]),
        skeleton=skel2,
        point_scores=np.array([0.1, 0.2, 0.3]),
        score=0.9,
    )

    before_xy = {n: pred[n]["xy"].copy() for n in ["A", "B", "C"]}
    before_score = {n: float(pred[n]["score"]) for n in ["A", "B", "C"]}

    labels = Labels(
        labeled_frames=[
            LabeledFrame(video=video, frame_idx=0, instances=[inst1]),
            LabeledFrame(video=video, frame_idx=1, instances=[pred]),
        ]
    )

    assert len(labels.skeletons) == 1
    assert pred.skeleton is labels.skeletons[0]
    for node in ["A", "B", "C"]:
        assert_equal(pred[node]["xy"], before_xy[node])
        assert float(pred[node]["score"]) == before_score[node]
    assert pred.score == 0.9


def test_reordered_equal_skeletons_not_merged():
    """Structurally-equal but reordered skeletons are kept distinct and safe."""
    skel_abc = Skeleton(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    skel_cba = Skeleton(nodes=["C", "B", "A"], edges=[("A", "B"), ("B", "C")])

    assert skel_abc.matches(skel_cba) is True
    assert skel_abc.matches(skel_cba, require_same_order=True) is False

    video = Video.from_filename("fake.mp4")
    inst_abc = Instance.from_numpy(
        np.array([[0, 0], [1, 1], [2, 2]]), skeleton=skel_abc
    )
    inst_cba = Instance.from_numpy(
        np.array([[9, 9], [1, 1], [7, 7]]), skeleton=skel_cba
    )

    labels = Labels(
        labeled_frames=[
            LabeledFrame(video=video, frame_idx=0, instances=[inst_abc]),
            LabeledFrame(video=video, frame_idx=1, instances=[inst_cba]),
        ]
    )

    # Reordered-equal skeletons are intentionally NOT merged.
    assert len(labels.skeletons) == 2
    # No silent corruption: node "A" still maps to its original positional value.
    assert_equal(inst_abc["A"]["xy"], [0, 0])
    assert not np.array_equal(inst_abc["A"]["xy"], [2, 2])


def test_register_skeleton_no_op_on_same_object():
    """Re-running update with one skeleton instance does not duplicate it."""
    skel = Skeleton(nodes=["A", "B"], edges=[("A", "B")])
    video = Video.from_filename("fake.mp4")
    inst = Instance.from_numpy(np.array([[0, 0], [1, 1]]), skeleton=skel)

    labels = Labels(
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, instances=[inst])]
    )
    assert len(labels.skeletons) == 1

    labels.update()
    assert len(labels.skeletons) == 1
    assert inst.skeleton is labels.skeletons[0]


def test_explicitly_registered_compatible_skeletons_preserved():
    """Distinct-but-compatible skeletons added explicitly are not auto-merged."""
    skel1 = Skeleton(nodes=["A", "B"], edges=[("A", "B")])
    skel2 = Skeleton(nodes=["A", "B"], edges=[("A", "B")])

    video = Video.from_filename("fake.mp4")
    labels = Labels(skeletons=[skel1, skel2], videos=[video])

    inst1 = Instance.from_numpy(np.array([[0, 0], [1, 1]]), skeleton=skel1)
    inst2 = Instance.from_numpy(np.array([[2, 2], [3, 3]]), skeleton=skel2)
    labels.append(LabeledFrame(video=video, frame_idx=0, instances=[inst1]))
    labels.append(LabeledFrame(video=video, frame_idx=1, instances=[inst2]))

    # Both skeletons were explicitly registered, so neither instance is rebound
    # and both skeletons are preserved (compatible skeletons stay distinct for
    # workflows like `fix --consolidate-skeletons`).
    assert len(labels.skeletons) == 2
    assert inst1.skeleton is skel1
    assert inst2.skeleton is skel2


def test_labels_numpy(labels_predictions: Labels):
    """Test the numpy method and its inverse update_from_numpy."""
    # Test conversion to numpy array
    tracks_arr = labels_predictions.numpy()

    # Verify the shape
    assert tracks_arr.shape[0] > 0  # At least one frame
    assert tracks_arr.shape[1] > 0  # At least one track
    assert tracks_arr.shape[2] > 0  # At least one node
    assert tracks_arr.shape[3] == 2  # x, y coordinates

    # Create a modified copy of the array
    modified_arr = tracks_arr.copy()

    # Modify some points
    if not np.all(np.isnan(modified_arr[0, 0, 0])):
        modified_arr[0, 0, 0] = modified_arr[0, 0, 0] + 5  # Move x by 5 pixels

    # Create a new labels object to test update_from_numpy
    new_labels = Labels()
    new_labels.videos = [labels_predictions.video]
    new_labels.skeletons = [labels_predictions.skeleton]
    new_labels.tracks = labels_predictions.tracks

    # Test update_from_numpy
    new_labels.update_from_numpy(modified_arr)

    # Test getting numpy array with confidence scores
    tracks_arr_with_conf = labels_predictions.numpy(return_confidence=True)
    assert tracks_arr_with_conf.shape[3] == 3  # x, y, confidence

    # Test update_from_numpy with confidence scores
    confidence_arr = np.ones_like(tracks_arr_with_conf)
    confidence_arr[:, :, :, :2] = tracks_arr  # Set xy coords
    confidence_arr[:, :, :, 2] = 0.75  # Set all confidence to 0.75

    # Test update with confidence scores
    new_labels.update_from_numpy(confidence_arr)

    # Verify confidence scores were updated by checking scores directly in instances
    found_score = False

    # Check by accessing through numpy with scores=True
    for lf in new_labels.labeled_frames:
        for inst in lf.predicted_instances:
            if isinstance(inst, PredictedInstance):
                # Get numpy representation with scores
                points_with_scores = inst.numpy(scores=True)
                if points_with_scores.shape[1] == 3:  # Has scores column
                    # Check if any scores are close to 0.75
                    score_matches = np.isclose(points_with_scores[:, 2], 0.75)
                    if np.any(score_matches):
                        found_score = True
                        break
        if found_score:
            break

    assert found_score, "No points with confidence scores found after update"


def test_labels_find(slp_typical):
    labels = load_slp(slp_typical)

    results = labels.find(video=labels.video, frame_idx=0)
    assert len(results) == 1
    lf = results[0]
    assert lf.frame_idx == 0

    labels.labeled_frames.append(LabeledFrame(video=labels.video, frame_idx=1))

    results = labels.find(video=labels.video)
    assert len(results) == 2

    results = labels.find(video=labels.video, frame_idx=2)
    assert len(results) == 0

    results = labels.find(video=labels.video, frame_idx=2, return_new=True)
    assert len(results) == 1
    assert results[0].frame_idx == 2
    assert len(results[0]) == 0


def test_labels_video():
    labels = Labels()

    with pytest.raises(ValueError):
        labels.video

    vid = Video(filename="test")
    labels.videos.append(vid)
    assert labels.video == vid

    labels.videos.append(Video(filename="test2"))
    with pytest.raises(ValueError):
        labels.video


def test_labels_skeleton():
    labels = Labels()

    with pytest.raises(ValueError):
        labels.skeleton

    skel = Skeleton(["A"])
    labels.skeletons.append(skel)
    assert labels.skeleton == skel

    labels.skeletons.append(Skeleton(["B"]))
    with pytest.raises(ValueError):
        labels.skeleton


def test_load_SLP(slp_minimal, tmp_path):
    """Test loading SLP file with uppercase extension (.SLP)."""
    # Copy the existing fixture to a temp file with uppercase extension
    uppercase_slp_path = tmp_path / "minimal_instance.SLP"
    shutil.copy(slp_minimal, uppercase_slp_path)

    # Test with string path
    labels = load_slp(str(uppercase_slp_path))
    assert len(labels) == 1
    assert len(labels[0]) == 2

    # Test with Path object
    labels = load_slp(uppercase_slp_path)
    assert len(labels) == 1
    assert len(labels[0]) == 2


def test_labels_getitem(slp_typical):
    labels = load_slp(slp_typical)
    labels.labeled_frames.append(LabeledFrame(video=labels.video, frame_idx=1))
    assert len(labels) == 2
    assert labels[0].frame_idx == 0
    assert len(labels[:2]) == 2
    assert len(labels[[0, 1]]) == 2
    assert len(labels[np.array([0, 1])]) == 2
    assert labels[(labels.video, 0)].frame_idx == 0

    with pytest.raises(IndexError):
        labels[(labels.video, 2000)]

    assert len(labels[labels.video]) == 2

    with pytest.raises(IndexError):
        labels[Video(filename="test")]

    with pytest.raises(IndexError):
        labels[None]


def test_labels_getitem_list_of_tuples(slp_typical):
    labels = load_slp(slp_typical)
    labels.labeled_frames.append(LabeledFrame(video=labels.video, frame_idx=1))
    assert len(labels) == 2

    keys = [(labels.video, 0), (labels.video, 1)]
    lfs = labels[keys]
    assert len(lfs) == 2
    assert lfs[0].frame_idx == 0
    assert lfs[1].frame_idx == 1

    keys = [(labels.video, 1), (labels.video, 0)]
    lfs = labels[keys]
    assert len(lfs) == 2
    assert lfs[0].frame_idx == 1
    assert lfs[1].frame_idx == 0

    keys = []
    lfs = labels[keys]
    assert len(lfs) == 0


def test_match_video_foreign_instance(slp_typical):
    """match_video resolves a foreign Video with the same path to the canonical."""
    labels = load_slp(slp_typical)
    canonical = labels.video

    # A freshly created Video is not identity-equal (Video is eq=False).
    foreign = Video(filename=canonical.filename, open_backend=False)
    assert foreign is not canonical
    assert labels.match_video(foreign) is canonical

    # An identity argument is returned unchanged.
    assert labels.match_video(canonical) is canonical


def test_match_video_by_path(slp_typical):
    """match_video resolves str and Path filenames to the canonical Video."""
    labels = load_slp(slp_typical)
    canonical = labels.video

    assert labels.match_video(str(canonical.filename)) is canonical
    assert labels.match_video(Path(canonical.filename)) is canonical


def test_match_video_no_match(slp_typical):
    """match_video returns None when no video matches."""
    labels = load_slp(slp_typical)
    assert labels.match_video("not_in_project.mp4") is None
    assert labels.match_video(Video(filename="not_in_project.mp4")) is None


def test_match_video_basename_fallback():
    """match_video falls back to basename matching for relocated files."""
    video = Video(filename="/original/dir/video.mp4", open_backend=False)
    labels = Labels(videos=[video])

    # Same basename in a different directory still resolves.
    assert labels.match_video("/new/location/video.mp4") is video


def test_match_video_definitive_over_basename():
    """An exact path match wins over a shared basename (no false ambiguity)."""
    v1 = Video(filename="/dir1/vid.mp4", open_backend=False)
    v2 = Video(filename="/dir2/vid.mp4", open_backend=False)
    labels = Labels(videos=[v1, v2])

    # Exact path -> definitive tier resolves to v1 despite the shared basename.
    assert labels.match_video("/dir1/vid.mp4") is v1
    assert labels.match_video("/dir2/vid.mp4") is v2


def test_match_video_ambiguous_raises():
    """match_video raises when multiple videos match by basename."""
    v1 = Video(filename="/dir1/vid.mp4", open_backend=False)
    v2 = Video(filename="/dir2/vid.mp4", open_backend=False)
    labels = Labels(videos=[v1, v2])

    with pytest.raises(ValueError, match="Ambiguous video match"):
        labels.match_video("/elsewhere/vid.mp4")


def test_match_video_ambiguous_definitive_raises():
    """match_video raises when multiple videos share the exact same path."""
    v1 = Video(filename="/dir/vid.mp4", open_backend=False)
    v2 = Video(filename="/dir/vid.mp4", open_backend=False)
    labels = Labels(videos=[v1, v2])

    with pytest.raises(ValueError, match="by file identity"):
        labels.match_video("/dir/vid.mp4")


def test_match_video_explicit_method():
    """match_video accepts an explicit method string or VideoMatcher."""
    v1 = Video(filename="/dir1/vid.mp4", open_backend=False)
    v2 = Video(filename="/dir2/other.mp4", open_backend=False)
    labels = Labels(videos=[v1, v2])

    # Basename method resolves a relocated file.
    assert labels.match_video("/x/vid.mp4", method="basename") is v1
    # Path method (lenient by default) also matches by basename.
    assert labels.match_video("/x/other.mp4", method="path") is v2
    # A VideoMatcher instance works too.
    matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)
    assert labels.match_video("/x/vid.mp4", method=matcher) is v1


def test_match_video_explicit_method_ambiguous():
    """Explicit-method matching raises on ambiguous matches."""
    v1 = Video(filename="/dir1/vid.mp4", open_backend=False)
    v2 = Video(filename="/dir2/vid.mp4", open_backend=False)
    labels = Labels(videos=[v1, v2])

    with pytest.raises(ValueError, match="Ambiguous video match"):
        labels.match_video("/x/vid.mp4", method="basename")


def test_match_video_auto_matcher_instance():
    """An AUTO VideoMatcher instance uses the same tiered cascade as method='auto'."""
    v1 = Video(filename="/dir1/vid.mp4", open_backend=False)
    v2 = Video(filename="/dir2/vid.mp4", open_backend=False)
    labels = Labels(videos=[v1, v2])

    # An exact path resolves via the definitive tier despite the shared basename,
    # exactly as method="auto" does (not the simplified pairwise AUTO check, which
    # would treat the basename match as a second, ambiguous candidate).
    matcher = VideoMatcher(method=VideoMatchMethod.AUTO)
    assert labels.match_video("/dir1/vid.mp4", method=matcher) is v1
    assert labels.match_video("/dir1/vid.mp4", method="auto") is v1


def test_match_video_bad_type():
    """match_video raises TypeError for unsupported argument types."""
    labels = Labels(videos=[Video(filename="vid.mp4", open_backend=False)])
    with pytest.raises(TypeError):
        labels.match_video(42)


def test_match_video_bad_method_type():
    """match_video raises TypeError for an unsupported method argument."""
    labels = Labels(videos=[Video(filename="vid.mp4", open_backend=False)])
    with pytest.raises(TypeError, match="method"):
        labels.match_video("vid.mp4", method=42)


def test_match_video_bad_method_string():
    """match_video raises ValueError for an unrecognized method string."""
    labels = Labels(videos=[Video(filename="vid.mp4", open_backend=False)])
    with pytest.raises(ValueError):
        labels.match_video("vid.mp4", method="not_a_method")


def test_match_video_hdf5_pkg(slp_minimal_pkg):
    """match_video resolves embedded HDF5 videos in a .pkg.slp file."""
    labels = load_slp(slp_minimal_pkg)
    canonical = labels.video

    # Foreign Video carrying the HDF5 backend resolves on dataset identity.
    foreign = Video(filename=canonical.filename, open_backend=False)
    foreign.backend = canonical.backend
    assert labels.match_video(foreign) is canonical

    # Resolving by the .pkg.slp path alone also works (single embedded video).
    assert labels.match_video(canonical.filename) is canonical


def test_find_foreign_video(slp_typical):
    """Find accepts a foreign Video, str, or Path argument."""
    labels = load_slp(slp_typical)
    canonical = labels.video
    labels.labeled_frames.append(LabeledFrame(video=canonical, frame_idx=1))

    foreign = Video(filename=canonical.filename, open_backend=False)
    assert len(labels.find(foreign)) == 2
    assert len(labels.find(str(canonical.filename))) == 2
    assert len(labels.find(Path(canonical.filename))) == 2
    assert len(labels.find(foreign, frame_idx=0)) == 1


def test_find_returns_new_for_unmatched_path(slp_typical):
    """find(..., return_new=True) attaches a usable Video for an unmatched path."""
    labels = load_slp(slp_typical)
    results = labels.find("unmatched.mp4", frame_idx=0, return_new=True)
    assert len(results) == 1
    assert results[0].frame_idx == 0
    assert isinstance(results[0].video, Video)
    assert Path(results[0].video.filename).name == "unmatched.mp4"


def test_getitem_by_path(slp_typical):
    """__getitem__ accepts str/Path videos and (path, frame_idx) tuples."""
    labels = load_slp(slp_typical)
    canonical = labels.video
    labels.labeled_frames.append(LabeledFrame(video=canonical, frame_idx=1))

    assert len(labels[str(canonical.filename)]) == 2
    assert len(labels[Path(canonical.filename)]) == 2
    assert labels[(str(canonical.filename), 0)].frame_idx == 0

    foreign = Video(filename=canonical.filename, open_backend=False)
    assert len(labels[foreign]) == 2

    # List of (path, frame_idx) tuples.
    keys = [(str(canonical.filename), 0), (str(canonical.filename), 1)]
    lfs = labels[keys]
    assert [lf.frame_idx for lf in lfs] == [0, 1]

    with pytest.raises(IndexError):
        labels["not_in_project.mp4"]


def test_extract_foreign_video(slp_typical):
    """Extract resolves foreign Video / path arguments via __getitem__."""
    labels = load_slp(slp_typical)
    canonical = labels.video
    foreign = Video(filename=canonical.filename, open_backend=False)

    assert len(labels.extract([(foreign, 0)])) == 1
    assert len(labels.extract(str(canonical.filename))) == 1


def test_get_queries_foreign_video():
    """get_* and numpy resolve a foreign Video argument."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="/data/vid.mp4", open_backend=False)
    inst = Instance.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skel)
    centroid = UserCentroid(x=5.0, y=10.0)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst], centroids=[centroid])
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skel])

    foreign = Video(filename="/data/vid.mp4", open_backend=False)
    assert len(labels.get_centroids(video=foreign)) == 1
    assert len(labels.get_centroids(video="/data/vid.mp4")) == 1
    assert labels.numpy(video=foreign, untracked=True).shape[0] == 1
    # Integer index still selects the video.
    assert labels.numpy(video=0, untracked=True).shape[0] == 1


def test_match_video_image_sequence(centered_pair_frame_paths):
    """match_video resolves image-sequence videos by their full filename list."""
    video = Video(filename=list(centered_pair_frame_paths), open_backend=False)
    labels = Labels(videos=[video])

    foreign = Video(filename=list(centered_pair_frame_paths), open_backend=False)
    assert labels.match_video(foreign) is video

    # A partially overlapping sequence is not an "auto" match.
    partial = Video(filename=list(centered_pair_frame_paths[:1]), open_backend=False)
    assert labels.match_video(partial) is None


def test_match_video_distinct_crops(centered_pair_low_quality_path):
    """match_video keeps mosaic tiles distinct and resolves each correctly.

    Two equal-size crops of one physical file share a root file. Without
    crop-aware path rungs, a query for one tile would mis-resolve to the other
    (or raise an ambiguity error). It must resolve to the matching tile.
    """
    src = Video.from_filename(centered_pair_low_quality_path)
    left = src.crop((0, 0, 192, 384))
    right = src.crop((192, 0, 384, 384))
    labels = Labels(videos=[left, right])

    # A fresh right-equivalent crop resolves to right, not left, no ambiguity.
    right_eq = src.crop((192, 0, 384, 384))
    assert labels.match_video(right_eq) is right
    # And a left-equivalent resolves to left.
    left_eq = src.crop((0, 0, 192, 384))
    assert labels.match_video(left_eq) is left


def test_match_video_identical_crop(centered_pair_low_quality_path):
    """match_video resolves an identical crop to the stored crop."""
    src = Video.from_filename(centered_pair_low_quality_path)
    a = src.crop((0, 0, 192, 384))
    labels = Labels(videos=[a])

    a_eq = src.crop((0, 0, 192, 384))
    assert labels.match_video(a_eq) is a


def test_match_video_crop_not_basename_shadowed(centered_pair_low_quality_path):
    """A crop query is not basename-matched to a different crop of one file."""
    src = Video.from_filename(centered_pair_low_quality_path)
    left = src.crop((0, 0, 192, 384))
    labels = Labels(videos=[left])

    # A different crop of the same source must not resolve to the left tile.
    right = src.crop((192, 0, 384, 384))
    assert labels.match_video(right) is None


def test_add_video_distinct_crops_not_collapsed(centered_pair_low_quality_path):
    """add_video keeps two distinct crops of one file as separate videos."""
    src = Video.from_filename(centered_pair_low_quality_path)
    left = src.crop((0, 0, 192, 384))
    right = src.crop((192, 0, 384, 384))
    labels = Labels()
    labels.add_video(left)
    labels.add_video(right)
    assert len(labels.videos) == 2


def test_add_video_identical_crops_dedup(centered_pair_low_quality_path):
    """add_video dedups two identical crops of one file to a single video."""
    src = Video.from_filename(centered_pair_low_quality_path)
    a = src.crop((0, 0, 192, 384))
    a_eq = src.crop((0, 0, 192, 384))
    labels = Labels()
    labels.add_video(a)
    returned = labels.add_video(a_eq)
    assert len(labels.videos) == 1
    assert returned is a


def test_add_video_noncrop_same_file_dedup(centered_pair_low_quality_path):
    """add_video still dedups non-crop videos of the same file (additive)."""
    v1 = Video.from_filename(centered_pair_low_quality_path)
    v2 = Video.from_filename(centered_pair_low_quality_path)
    labels = Labels()
    labels.add_video(v1)
    returned = labels.add_video(v2)
    assert len(labels.videos) == 1
    assert returned is v1


def test_labels_save(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    labels.save(tmp_path / "test.slp")
    assert (tmp_path / "test.slp").exists()


def test_labels_clean_unchanged(slp_real_data):
    labels = load_slp(slp_real_data)
    assert len(labels) == 10
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    assert labels[1].frame_idx == 990
    assert len(labels[1]) == 2
    assert len(labels.skeletons) == 1
    assert len(labels.videos) == 1
    assert len(labels.tracks) == 0
    labels.clean(
        frames=True, empty_instances=True, skeletons=True, tracks=True, videos=True
    )
    assert len(labels) == 10
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    assert labels[1].frame_idx == 990
    assert len(labels[1]) == 2
    assert len(labels.skeletons) == 1
    assert len(labels.videos) == 1
    assert len(labels.tracks) == 0


def test_labels_clean_frames(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    labels[0].instances = []
    labels.clean(
        frames=True, empty_instances=False, skeletons=False, tracks=False, videos=False
    )
    assert len(labels) == 9
    assert labels[0].frame_idx == 990
    assert len(labels[0]) == 2


def test_labels_negative_frames():
    """Test negative_frames property returns only negative frames."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test")

    lf_regular = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance([[0, 1], [2, 3]], skeleton=skel)],
    )
    lf_negative = LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[],
        is_negative=True,
    )
    lf_empty = LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[],
        is_negative=False,
    )

    labels = Labels(
        labeled_frames=[lf_regular, lf_negative, lf_empty],
        videos=[video],
        skeletons=[skel],
    )

    negative = labels.negative_frames
    assert len(negative) == 1
    assert negative[0].frame_idx == 1
    assert negative[0].is_negative is True


def test_labels_user_labeled_frames_includes_negative():
    """Test user_labeled_frames includes negative frames."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test")

    lf_regular = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance([[0, 1], [2, 3]], skeleton=skel)],
    )
    lf_negative = LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[],
        is_negative=True,
    )
    lf_empty = LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[],
        is_negative=False,
    )
    lf_pred_only = LabeledFrame(
        video=video,
        frame_idx=3,
        instances=[PredictedInstance([[0, 1], [2, 3]], skeleton=skel)],
    )

    labels = Labels(
        labeled_frames=[lf_regular, lf_negative, lf_empty, lf_pred_only],
        videos=[video],
        skeletons=[skel],
    )

    user_frames = labels.user_labeled_frames
    # Should include frame 0 (has user instances) and frame 1 (is negative)
    # Should NOT include frame 2 (empty, not negative) or frame 3 (pred only)
    assert len(user_frames) == 2
    frame_indices = {lf.frame_idx for lf in user_frames}
    assert frame_indices == {0, 1}


def test_labels_clean_preserves_negative_frames():
    """Test that clean() preserves negative frames but removes empty non-negative."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test")

    lf_regular = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance([[0, 1], [2, 3]], skeleton=skel)],
    )
    lf_negative = LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[],
        is_negative=True,
    )
    lf_empty = LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[],
        is_negative=False,
    )

    labels = Labels(
        labeled_frames=[lf_regular, lf_negative, lf_empty],
        videos=[video],
        skeletons=[skel],
    )

    assert len(labels) == 3

    labels.clean(frames=True, skeletons=False, tracks=False, videos=False)

    # Should keep frame 0 (has instances) and frame 1 (is negative)
    # Should remove frame 2 (empty, not negative)
    assert len(labels) == 2
    frame_indices = {lf.frame_idx for lf in labels.labeled_frames}
    assert frame_indices == {0, 1}

    # Verify negative frame is preserved
    assert any(lf.is_negative for lf in labels.labeled_frames)


def test_labels_clean_empty_instances(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 2
    labels[0].instances = [
        Instance.from_numpy(
            np.full((len(labels.skeleton), 2), np.nan), skeleton=labels.skeleton
        )
    ]
    labels.clean(
        frames=False, empty_instances=True, skeletons=False, tracks=False, videos=False
    )
    assert len(labels) == 10
    assert labels[0].frame_idx == 0
    assert len(labels[0]) == 0

    labels.clean(
        frames=True, empty_instances=True, skeletons=False, tracks=False, videos=False
    )
    assert len(labels) == 9


def test_labels_clean_skeletons(slp_real_data):
    labels = load_slp(slp_real_data)
    labels.skeletons.append(Skeleton(["A", "B"]))
    assert len(labels.skeletons) == 2
    labels.clean(
        frames=False, empty_instances=False, skeletons=True, tracks=False, videos=False
    )
    assert len(labels) == 10
    assert len(labels.skeletons) == 1


def test_labels_clean_tracks(slp_real_data):
    labels = load_slp(slp_real_data)
    labels.tracks.append(Track(name="test1"))
    labels.tracks.append(Track(name="test2"))
    assert len(labels.tracks) == 2
    labels[0].instances[0].track = labels.tracks[1]
    labels.clean(
        frames=False, empty_instances=False, skeletons=False, tracks=True, videos=False
    )
    assert len(labels) == 10
    assert len(labels.tracks) == 1
    assert labels[0].instances[0].track == labels.tracks[0]
    assert labels.tracks[0].name == "test2"


def test_labels_clean_videos(slp_real_data):
    labels = load_slp(slp_real_data)
    labels.videos.append(Video(filename="test2"))
    assert len(labels.videos) == 2
    labels.clean(
        frames=False, empty_instances=False, skeletons=False, tracks=False, videos=True
    )
    assert len(labels) == 10
    assert len(labels.videos) == 1
    assert labels.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"


def test_labels_remove_predictions(slp_real_data):
    labels = load_slp(slp_real_data)
    assert len(labels) == 10
    assert sum([len(lf.predicted_instances) for lf in labels]) == 12
    labels.remove_predictions(clean=False)
    assert len(labels) == 10
    assert sum([len(lf.predicted_instances) for lf in labels]) == 0

    labels = load_slp(slp_real_data)
    labels.remove_predictions(clean=True)
    assert len(labels) == 5
    assert sum([len(lf.predicted_instances) for lf in labels]) == 0


def test_replace_videos(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"
    labels.replace_videos(new_videos=[Video.from_filename("fake.mp4")])

    for lf in labels:
        assert lf.video.filename == "fake.mp4"

    for sf in labels.suggestions:
        assert sf.video.filename == "fake.mp4"

    assert labels.video.filename == "fake.mp4"


def test_set_video_color_mode_grayscale(slp_minimal):
    """Test setting all videos to grayscale mode."""
    labels = load_slp(slp_minimal)
    labels.set_video_color_mode("grayscale")
    for video in labels.videos:
        assert video.grayscale is True


def test_set_video_color_mode_rgb(slp_minimal):
    """Test setting all videos to RGB mode."""
    labels = load_slp(slp_minimal)
    labels.set_video_color_mode("rgb")
    for video in labels.videos:
        assert video.grayscale is False


def test_set_video_color_mode_auto(slp_minimal):
    """Test resetting all videos to auto-detection."""
    labels = load_slp(slp_minimal)
    # First set to grayscale
    labels.set_video_color_mode("grayscale")
    # Then reset to auto
    labels.set_video_color_mode("auto")
    for video in labels.videos:
        assert video.grayscale is None


def test_set_video_color_mode_default_is_auto(slp_minimal):
    """Test that default mode is 'auto'."""
    labels = load_slp(slp_minimal)
    labels.set_video_color_mode()  # No argument
    for video in labels.videos:
        assert video.grayscale is None


def test_set_video_color_mode_propagates_to_source_video():
    """Test that color mode propagates through source_video chain."""
    # Create a video with source_video chain (simulating embedded video)
    source_video = Video(filename="original.mp4", open_backend=False)
    source_video.backend_metadata["shape"] = (100, 480, 640, 3)

    embedded_video = Video(filename="embedded.pkg.slp", open_backend=False)
    embedded_video.backend_metadata["shape"] = (10, 480, 640, 3)
    embedded_video.source_video = source_video

    skeleton = Skeleton(["head", "tail"])
    labels = Labels(skeletons=[skeleton], videos=[embedded_video])

    # Set color mode
    labels.set_video_color_mode("grayscale")

    # Both embedded and source should have the setting
    assert embedded_video.backend_metadata.get("grayscale") is True
    assert source_video.backend_metadata.get("grayscale") is True


def test_set_video_color_mode_propagates_multi_level_chain():
    """Test that color mode propagates through multi-level source_video chain."""
    # Create multi-level chain: embedded <- intermediate <- original
    original = Video(filename="original.mp4", open_backend=False)
    original.backend_metadata["shape"] = (100, 480, 640, 3)

    intermediate = Video(filename="intermediate.pkg.slp", open_backend=False)
    intermediate.backend_metadata["shape"] = (50, 480, 640, 3)
    intermediate.source_video = original

    embedded = Video(filename="final.pkg.slp", open_backend=False)
    embedded.backend_metadata["shape"] = (10, 480, 640, 3)
    embedded.source_video = intermediate

    skeleton = Skeleton(["head", "tail"])
    labels = Labels(skeletons=[skeleton], videos=[embedded])

    # Set color mode
    labels.set_video_color_mode("rgb")

    # All levels should have the setting
    assert embedded.backend_metadata.get("grayscale") is False
    assert intermediate.backend_metadata.get("grayscale") is False
    assert original.backend_metadata.get("grayscale") is False


def test_replace_filenames():
    labels = Labels(videos=[Video.from_filename("a.mp4"), Video.from_filename("b.mp4")])

    with pytest.raises(ValueError):
        labels.replace_filenames()

    with pytest.raises(ValueError):
        labels.replace_filenames(new_filenames=[], filename_map={})

    with pytest.raises(ValueError):
        labels.replace_filenames(new_filenames=[], prefix_map={})

    with pytest.raises(ValueError):
        labels.replace_filenames(filename_map={}, prefix_map={})

    with pytest.raises(ValueError):
        labels.replace_filenames(new_filenames=[], filename_map={}, prefix_map={})

    labels.replace_filenames(new_filenames=["c.mp4", "d.mp4"])
    assert [v.filename for v in labels.videos] == ["c.mp4", "d.mp4"]

    with pytest.raises(ValueError):
        labels.replace_filenames(["f.mp4"])

    labels.replace_filenames(
        filename_map={"c.mp4": "/a/b/c.mp4", "d.mp4": "/a/b/d.mp4"}
    )
    assert [Path(v.filename).as_posix() for v in labels.videos] == [
        "/a/b/c.mp4",
        "/a/b/d.mp4",
    ]

    labels.replace_filenames(prefix_map={"/a/b/": "/A/B"})
    assert [Path(v.filename).as_posix() for v in labels.videos] == [
        "/A/B/c.mp4",
        "/A/B/d.mp4",
    ]

    labels = Labels(videos=[Video.from_filename(["imgs/img0.png", "imgs/img1.png"])])
    labels.replace_filenames(
        filename_map={
            "imgs/img0.png": "train/imgs/img0.png",
            "imgs/img1.png": "train/imgs/img1.png",
        }
    )
    assert labels.video.filename == ["train/imgs/img0.png", "train/imgs/img1.png"]

    labels.replace_filenames(prefix_map={"train/": "test/"})
    assert labels.video.filename == ["test/imgs/img0.png", "test/imgs/img1.png"]


def test_replace_filenames_open_videos():
    """Test that open_videos=False prevents video backend from opening."""
    # Create videos that won't exist on disk
    labels = Labels(
        videos=[
            Video.from_filename("nonexistent_a.mp4"),
            Video.from_filename("nonexistent_b.mp4"),
        ]
    )

    # Test with open_videos=True (default) - should attempt to open backend
    labels.replace_filenames(
        new_filenames=["new_nonexistent_a.mp4", "new_nonexistent_b.mp4"]
    )
    # Backend should be None since files don't exist
    assert labels.videos[0].backend is None
    assert labels.videos[1].backend is None
    assert labels.videos[0].filename == "new_nonexistent_a.mp4"
    assert labels.videos[1].filename == "new_nonexistent_b.mp4"

    # Test with open_videos=False - should not attempt to open backend
    labels.replace_filenames(
        new_filenames=["final_a.mp4", "final_b.mp4"], open_videos=False
    )
    # Backend should still be None, but filenames should be updated
    assert labels.videos[0].backend is None
    assert labels.videos[1].backend is None
    assert labels.videos[0].filename == "final_a.mp4"
    assert labels.videos[1].filename == "final_b.mp4"

    # Test with filename_map and open_videos=False
    labels.replace_filenames(
        filename_map={"final_a.mp4": "mapped_a.mp4", "final_b.mp4": "mapped_b.mp4"},
        open_videos=False,
    )
    assert labels.videos[0].filename == "mapped_a.mp4"
    assert labels.videos[1].filename == "mapped_b.mp4"

    # Test with prefix_map and open_videos=False
    labels.replace_filenames(prefix_map={"mapped_": "prefixed_"}, open_videos=False)
    assert labels.videos[0].filename == "prefixed_a.mp4"
    assert labels.videos[1].filename == "prefixed_b.mp4"


def test_replace_filenames_cross_platform():
    """Test that prefix_map handles Windows and Linux paths correctly."""
    # Test Windows paths being replaced with Linux paths
    labels = Labels(
        videos=[
            Video.from_filename(r"C:\data\videos\test1.mp4"),
            Video.from_filename(r"C:\data\videos\test2.mp4"),
        ]
    )

    # Replace Windows prefix with Linux prefix - should match despite backslashes
    labels.replace_filenames(
        prefix_map={r"C:\data\videos": "/mnt/storage/videos"}, open_videos=False
    )
    assert labels.videos[0].filename == "/mnt/storage/videos/test1.mp4"
    assert labels.videos[1].filename == "/mnt/storage/videos/test2.mp4"

    # Test Linux paths being replaced with Windows paths
    labels = Labels(
        videos=[
            Video.from_filename("/home/user/data/vid1.mp4"),
            Video.from_filename("/home/user/data/vid2.mp4"),
        ]
    )

    # Replace Linux prefix with Windows prefix
    labels.replace_filenames(
        prefix_map={"/home/user/data": r"D:\mydata"}, open_videos=False
    )
    assert labels.videos[0].filename == r"D:\mydata/vid1.mp4"
    assert labels.videos[1].filename == r"D:\mydata/vid2.mp4"

    # Test mixed separators in the source paths
    labels = Labels(
        videos=[
            Video.from_filename(r"C:/mixed\path/file1.mp4"),
            Video.from_filename("C:/mixed/path/file2.mp4"),
        ]
    )

    # Should match both despite different separators
    labels.replace_filenames(
        prefix_map={"C:/mixed/path": "/unified/path"}, open_videos=False
    )
    assert labels.videos[0].filename == "/unified/path/file1.mp4"
    assert labels.videos[1].filename == "/unified/path/file2.mp4"

    # Test with trailing separators
    labels = Labels(
        videos=[
            Video.from_filename("/data/videos/test.mp4"),
        ]
    )

    # Old prefix with trailing slash, new without
    labels.replace_filenames(
        prefix_map={"/data/videos/": "/new/location"}, open_videos=False
    )
    assert labels.videos[0].filename == "/new/location/test.mp4"

    # Reset for next test
    labels = Labels(
        videos=[
            Video.from_filename("/data/videos/test.mp4"),
        ]
    )

    # Old prefix without trailing slash, new with
    labels.replace_filenames(
        prefix_map={"/data/videos": "/new/location/"}, open_videos=False
    )
    assert labels.videos[0].filename == "/new/location/test.mp4"

    # Test with list of filenames (ImageVideo case)
    labels = Labels(
        videos=[
            Video.from_filename(["/data/imgs/img0.png", "/data/imgs/img1.png"]),
        ]
    )

    # Replace prefix in list of filenames
    labels.replace_filenames(prefix_map={"/data/imgs": "/new/imgs"}, open_videos=False)
    assert labels.videos[0].filename == ["/new/imgs/img0.png", "/new/imgs/img1.png"]

    # Test list with some non-matching files
    labels = Labels(
        videos=[
            Video.from_filename(["/data/imgs/img0.png", "/other/path/img1.png"]),
        ]
    )
    labels.replace_filenames(prefix_map={"/data/imgs": "/new/imgs"}, open_videos=False)
    assert labels.videos[0].filename == ["/new/imgs/img0.png", "/other/path/img1.png"]

    # Test list with trailing separator and new prefix ending with separator
    labels = Labels(
        videos=[
            Video.from_filename(["/data/imgs/img0.png", "/data/imgs/img1.png"]),
        ]
    )
    labels.replace_filenames(
        prefix_map={"/data/imgs/": "/new/imgs/"}, open_videos=False
    )
    assert labels.videos[0].filename == ["/new/imgs/img0.png", "/new/imgs/img1.png"]

    # Test with new_prefix ending with slash but remainder starting with slash
    labels = Labels(
        videos=[
            Video.from_filename(["/data/imgs/img0.png"]),
        ]
    )
    labels.replace_filenames(prefix_map={"/data/imgs": "/new/imgs/"}, open_videos=False)
    assert labels.videos[0].filename == ["/new/imgs/img0.png"]

    # Test case where old_prefix has sep and new doesn't, with list
    labels = Labels(
        videos=[
            Video.from_filename(["/data/imgs/file.png"]),
        ]
    )
    labels.replace_filenames(prefix_map={"/data/imgs/": "/new/imgs"}, open_videos=False)
    assert labels.videos[0].filename == ["/new/imgs/file.png"]


def test_replace_filenames_edge_cases_windows_paths():
    """Test edge cases in replace_filenames to improve coverage."""
    # Test lines 1088-1095: list case where old_ends_with_sep=True after sanitization
    # This requires Windows-style paths with trailing backslash
    # Case 1: new_prefix doesn't end with separator (lines 1091-1093)
    labels = Labels(
        videos=[
            Video.from_filename([r"C:\prefix\file1.mp4", r"C:\prefix\file2.mp4"]),
        ]
    )
    # Windows path with single backslash at end -
    # preserves trailing / after sanitization
    labels.replace_filenames(
        prefix_map={"C:\\prefix\\": "/newprefix"}, open_videos=False
    )
    assert labels.videos[0].filename == ["/newprefix/file1.mp4", "/newprefix/file2.mp4"]

    # Case 2: new_prefix ends with separator (line 1095)
    labels = Labels(
        videos=[
            Video.from_filename([r"C:\prefix\file1.mp4", r"C:\prefix\file2.mp4"]),
        ]
    )
    labels.replace_filenames(
        prefix_map={"C:\\prefix\\": "/newprefix/"}, open_videos=False
    )
    assert labels.videos[0].filename == ["/newprefix/file1.mp4", "/newprefix/file2.mp4"]

    # Case 3: empty new_prefix (line 1095)
    labels = Labels(
        videos=[
            Video.from_filename([r"C:\prefix\file1.mp4", r"C:\prefix\file2.mp4"]),
        ]
    )
    labels.replace_filenames(prefix_map={"C:\\prefix\\": ""}, open_videos=False)
    assert labels.videos[0].filename == ["file1.mp4", "file2.mp4"]

    # Test lines 1125-1128: non-list case where
    # old_ends_with_sep=True after sanitization
    # Case 1: new_prefix doesn't end with separator
    labels = Labels(
        videos=[
            Video.from_filename(r"C:\prefix\file.mp4"),
        ]
    )
    labels.replace_filenames(
        prefix_map={"C:\\prefix\\": "/newprefix"}, open_videos=False
    )
    assert labels.videos[0].filename == "/newprefix/file.mp4"

    # Case 2: new_prefix ends with separator
    labels = Labels(
        videos=[
            Video.from_filename(r"C:\prefix\file.mp4"),
        ]
    )
    labels.replace_filenames(
        prefix_map={"C:\\prefix\\": "/newprefix/"}, open_videos=False
    )
    assert labels.videos[0].filename == "/newprefix/file.mp4"

    # Case 3: empty new_prefix
    labels = Labels(
        videos=[
            Video.from_filename(r"C:\prefix\file.mp4"),
        ]
    )
    labels.replace_filenames(prefix_map={"C:\\prefix\\": ""}, open_videos=False)
    assert labels.videos[0].filename == "file.mp4"


def test_split(slp_real_data, tmp_path):
    # n = 0
    labels = Labels()
    split1, split2 = labels.split(0.5)
    assert len(split1) == len(split2) == 0

    # n = 1
    labels.append(LabeledFrame(video=Video("test.mp4"), frame_idx=0))
    split1, split2 = labels.split(0.5)
    assert len(split1) == len(split2) == 1
    assert split1[0].frame_idx == 0
    assert split2[0].frame_idx == 0

    split1, split2 = labels.split(0.999)
    assert len(split1) == len(split2) == 1
    assert split1[0].frame_idx == 0
    assert split2[0].frame_idx == 0

    split1, split2 = labels.split(n=1)
    assert len(split1) == len(split2) == 1
    assert split1[0].frame_idx == 0
    assert split2[0].frame_idx == 0

    # Real data
    labels = load_slp(slp_real_data)
    assert len(labels) == 10

    split1, split2 = labels.split(n=0.6)
    assert len(split1) == 6
    assert len(split2) == 4

    # Rounding errors
    split1, split2 = labels.split(n=0.001)
    assert len(split1) == 1
    assert len(split2) == 9

    split1, split2 = labels.split(n=0.999)
    assert len(split1) == 9
    assert len(split2) == 1

    # Integer
    split1, split2 = labels.split(n=8)
    assert len(split1) == 8
    assert len(split2) == 2

    # Serialization round trip
    split1.save(tmp_path / "split1.slp")
    split1_ = load_slp(tmp_path / "split1.slp")
    assert len(split1) == len(split1_)
    assert split1.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"
    assert split1_.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"

    split2.save(tmp_path / "split2.slp")
    split2_ = load_slp(tmp_path / "split2.slp")
    assert len(split2) == len(split2_)
    assert split2.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"
    assert split2_.video.filename == "tests/data/videos/centered_pair_low_quality.mp4"

    # Serialization round trip with embedded data
    labels = load_slp(slp_real_data)
    labels.save(tmp_path / "test.pkg.slp", embed=True)
    pkg = load_slp(tmp_path / "test.pkg.slp")

    split1, split2 = pkg.split(n=0.8)
    assert len(split1) == 8
    assert len(split2) == 2
    assert split1.video.filename == (tmp_path / "test.pkg.slp").as_posix()
    assert split2.video.filename == (tmp_path / "test.pkg.slp").as_posix()
    assert (
        split1.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert (
        split2.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )

    split1.save(tmp_path / "split1.pkg.slp", embed=True, embed_inplace=True)
    split2.save(tmp_path / "split2.pkg.slp", embed=True, embed_inplace=True)
    assert pkg.video.filename == (tmp_path / "test.pkg.slp").as_posix()
    assert (
        Path(split1.video.filename).as_posix()
        == (tmp_path / "split1.pkg.slp").as_posix()
    )
    assert (
        Path(split2.video.filename).as_posix()
        == (tmp_path / "split2.pkg.slp").as_posix()
    )
    assert (
        split1.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert (
        split2.video.source_video.filename
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )

    split1_ = load_slp(tmp_path / "split1.pkg.slp")
    split2_ = load_slp(tmp_path / "split2.pkg.slp")
    assert len(split1_) == 8
    assert len(split2_) == 2
    assert (
        Path(split1_.video.filename).as_posix()
        == (tmp_path / "split1.pkg.slp").as_posix()
    )
    assert (
        Path(split2_.video.filename).as_posix()
        == (tmp_path / "split2.pkg.slp").as_posix()
    )
    # Check original_video field for the ultimate source
    assert hasattr(split1_.video, "original_video")
    assert (
        Path(split1_.video.original_video.filename).as_posix()
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert hasattr(split2_.video, "original_video")
    assert (
        Path(split2_.video.original_video.filename).as_posix()
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )


def test_extract_with_suggestions():
    video1 = Video(filename="v1.mp4")
    video2 = Video(filename="v2.mp4")
    skel = Skeleton(["a", "b"])
    labels = Labels(
        labeled_frames=[
            LabeledFrame(video1, 0, [Instance.from_numpy(np.zeros((2, 2)), skel)]),
            LabeledFrame(video1, 1, [Instance.from_numpy(np.zeros((2, 2)), skel)]),
            LabeledFrame(video2, 0, [Instance.from_numpy(np.zeros((2, 2)), skel)]),
        ],
        suggestions=[
            SuggestionFrame(video1, 2),
            SuggestionFrame(video2, 1),
        ],
    )
    assert len(labels.videos) == 2
    assert len(labels.suggestions) == 2

    # Extract LFs from video 1.
    extracted = labels.extract([0, 1])
    assert len(extracted) == 2
    assert len(extracted.videos) == 1
    assert extracted.videos[0].matches_content(video1)
    assert extracted.videos[0].matches_path(video1)
    assert len(extracted.suggestions) == 1
    assert extracted.suggestions[0].video.matches_content(video1)
    assert extracted.suggestions[0].video.matches_path(video1)
    assert extracted.suggestions[0].frame_idx == 2
    # Check that the suggestion video is the same object as the LF video.
    assert extracted.suggestions[0].video is extracted.videos[0]

    # Extract LFs from video 2.
    extracted = labels.extract([2])
    assert len(extracted) == 1
    assert len(extracted.videos) == 1
    assert extracted.videos[0].matches_content(video2)
    assert extracted.videos[0].matches_path(video2)
    assert len(extracted.suggestions) == 1
    assert extracted.suggestions[0].video.matches_content(video2)
    assert extracted.suggestions[0].video.matches_path(video2)
    assert extracted.suggestions[0].frame_idx == 1
    assert extracted.suggestions[0].video is extracted.videos[0]

    # Extract LFs from both.
    extracted = labels.extract([0, 2])
    assert len(extracted) == 2
    assert len(extracted.videos) == 2
    assert len(extracted.suggestions) == 2
    assert extracted.suggestions[0].video.matches_content(video1)
    assert extracted.suggestions[1].video.matches_content(video2)


def test_make_training_splits(slp_real_data):
    labels = load_slp(slp_real_data)
    assert len(labels.user_labeled_frames) == 5

    train, val = labels.make_training_splits(0.8)
    assert len(train) == 4
    assert len(val) == 1

    train, val = labels.make_training_splits(3)
    assert len(train) == 3
    assert len(val) == 2

    train, val = labels.make_training_splits(0.8, 0.2)
    assert len(train) == 4
    assert len(val) == 1

    train, val, test = labels.make_training_splits(0.8, 0.1, 0.1)
    assert len(train) == 4
    assert len(val) == 1
    assert len(test) == 1

    train, val, test = labels.make_training_splits(n_train=0.6, n_test=1)
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1

    train, val, test = labels.make_training_splits(n_train=1, n_val=1, n_test=1)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1

    train, val, test = labels.make_training_splits(n_train=0.4, n_val=0.4, n_test=0.2)
    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 1


def test_make_training_splits_save(slp_real_data, tmp_path):
    labels = load_slp(slp_real_data)

    train, val, test = labels.make_training_splits(0.6, 0.2, 0.2, save_dir=tmp_path)

    train_, val_, test_ = (
        load_slp(tmp_path / "train.pkg.slp"),
        load_slp(tmp_path / "val.pkg.slp"),
        load_slp(tmp_path / "test.pkg.slp"),
    )

    assert len(train_) == len(train)
    assert len(val_) == len(val)
    assert len(test_) == len(test)

    assert train_.provenance["source_labels"] == slp_real_data
    assert val_.provenance["source_labels"] == slp_real_data
    assert test_.provenance["source_labels"] == slp_real_data


@pytest.mark.parametrize("embed", [True, False])
def test_make_training_splits_save_with_embed(slp_real_data, tmp_path, embed):
    labels = load_slp(slp_real_data)

    train, val, test = labels.make_training_splits(
        0.6, 0.2, 0.2, save_dir=tmp_path, embed=embed
    )

    if embed:
        train_, val_, test_ = (
            load_slp(tmp_path / "train.pkg.slp"),
            load_slp(tmp_path / "val.pkg.slp"),
            load_slp(tmp_path / "test.pkg.slp"),
        )
    else:
        train_, val_, test_ = (
            load_slp(tmp_path / "train.slp"),
            load_slp(tmp_path / "val.slp"),
            load_slp(tmp_path / "test.slp"),
        )

    assert len(train_) == len(train)
    assert len(val_) == len(val)
    assert len(test_) == len(test)

    if embed:
        assert train_.provenance["source_labels"] == slp_real_data
        assert val_.provenance["source_labels"] == slp_real_data
        assert test_.provenance["source_labels"] == slp_real_data
    else:
        assert train_.video.filename == labels.video.filename
        assert val_.video.filename == labels.video.filename
        assert test_.video.filename == labels.video.filename

    if embed:
        for labels_ in [train_, val_, test_]:
            for lf in labels_:
                assert lf.image.shape == (384, 384, 1)


def test_labels_instances():
    labels = Labels()
    labels.append(
        LabeledFrame(
            video=Video("test.mp4"),
            frame_idx=0,
            instances=[
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=Skeleton(["A", "B"])
                )
            ],
        )
    )
    assert len(list(labels.instances)) == 1

    labels.append(
        LabeledFrame(
            video=labels.video,
            frame_idx=1,
            instances=[
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=labels.skeleton
                ),
                Instance.from_numpy(
                    np.array([[0, 1], [2, 3]]), skeleton=labels.skeleton
                ),
            ],
        )
    )
    assert len(list(labels.instances)) == 3


def test_labels_rename_nodes(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]

    labels.rename_nodes({"head": "front", "abdomen": "back"})
    assert labels.skeleton.node_names == ["front", "back"]

    labels.skeletons.append(Skeleton(["A", "B"]))
    with pytest.raises(ValueError):
        labels.rename_nodes({"A": "a", "B": "b"})
    labels.rename_nodes({"A": "a", "B": "b"}, skeleton=labels.skeletons[1])
    assert labels.skeletons[1].node_names == ["a", "b"]


def test_labels_remove_nodes(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]
    assert_allclose(
        labels[0][0].numpy(), [[91.886988, 204.018843], [151.536969, 159.825034]]
    )

    labels.remove_nodes(["head"])
    assert labels.skeleton.node_names == ["abdomen"]
    assert_allclose(labels[0][0].numpy(), [[151.536969, 159.825034]])

    for inst in labels.instances:
        assert inst.numpy().shape == (1, 2)

    labels.skeletons.append(Skeleton())
    with pytest.raises(ValueError):
        labels.remove_nodes(["head"])


def test_labels_reorder_nodes(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]
    assert_allclose(
        labels[0][0].numpy(), [[91.886988, 204.018843], [151.536969, 159.825034]]
    )

    labels.reorder_nodes(["abdomen", "head"])
    assert labels.skeleton.node_names == ["abdomen", "head"]
    assert_allclose(
        labels[0][0].numpy(), [[151.536969, 159.825034], [91.886988, 204.018843]]
    )

    labels.skeletons.append(Skeleton())
    with pytest.raises(ValueError):
        labels.reorder_nodes(["head", "abdomen"])


def test_labels_replace_skeleton(slp_real_data):
    labels = load_slp(slp_real_data)
    assert labels.skeleton.node_names == ["head", "abdomen"]
    inst = labels[0][0]
    assert_allclose(inst.numpy(), [[91.886988, 204.018843], [151.536969, 159.825034]])

    # Replace with full mapping
    new_skel = Skeleton(["ABDOMEN", "HEAD"])
    labels.replace_skeleton(new_skel, node_map={"abdomen": "ABDOMEN", "head": "HEAD"})
    assert labels.skeleton == new_skel
    inst = labels[0][0]
    assert inst.skeleton == new_skel
    assert_allclose(inst.numpy(), [[151.536969, 159.825034], [91.886988, 204.018843]])

    # Replace with partial (inferred) mapping
    new_skel = Skeleton(["x", "ABDOMEN"])
    labels.replace_skeleton(new_skel)
    assert labels.skeleton == new_skel
    inst = labels[0][0]
    assert inst.skeleton == new_skel
    assert_allclose(inst.numpy(), [[np.nan, np.nan], [151.536969, 159.825034]])

    # Replace with no mapping
    new_skel = Skeleton(["front", "back"])
    labels.replace_skeleton(new_skel)
    assert labels.skeleton == new_skel
    inst = labels[0][0]
    assert inst.skeleton == new_skel
    assert_allclose(inst.numpy(), [[np.nan, np.nan], [np.nan, np.nan]])

    # Test line 927: ValueError when multiple skeletons and no old_skeleton specified
    labels.skeletons.append(Skeleton(["node1", "node2"]))
    assert len(labels.skeletons) == 2

    new_skel = Skeleton(["A", "B"])
    with pytest.raises(ValueError, match="Old skeleton must be specified"):
        labels.replace_skeleton(new_skel)


def test_labels_trim(centered_pair, tmpdir):
    labels = load_slp(centered_pair)

    new_path = tmpdir / "trimmed.slp"
    trimmed_labels = labels.trim(new_path, np.arange(100, 200))
    assert len(trimmed_labels) == 100
    # make sure both paths are pathlib.Path before assertion. This make the comparison
    # robust against different path format, etc on different systems
    assert Path(trimmed_labels.video.filename) == Path(new_path).with_suffix(".mp4")
    assert trimmed_labels.video.shape == (100, 384, 384, 1)
    assert trimmed_labels[0].frame_idx == 0
    assert_equal(trimmed_labels[0].numpy(), labels[(labels.video, 100)].numpy())

    labels.videos.append(Video.from_filename("fake.mp4"))
    with pytest.raises(ValueError):
        labels.trim(new_path, np.arange(100, 200))

    labels.trim(new_path, np.arange(100, 200), video=0)


def test_labels_trim_with_suggestions(slp_real_data, tmp_path):
    """Test that Labels.trim() correctly handles suggestions."""
    labels = load_slp(slp_real_data)

    # Check that we have suggestions
    assert len(labels.suggestions) > 0, "Test requires labels with suggestions"

    # Define frame range
    start_frame = 200
    end_frame = 800

    # Trim to a range that should include some suggestions
    new_path = tmp_path / "trimmed.slp"
    trimmed_labels = labels.trim(new_path, np.arange(start_frame, end_frame))

    # Verify trimmed labels saved successfully
    assert new_path.exists()
    assert new_path.with_suffix(".mp4").exists()

    # Verify suggestions are filtered and adjusted
    for sf in trimmed_labels.suggestions:
        # All suggestions should reference the new video
        assert sf.video == trimmed_labels.video
        # Frame indices should be adjusted (original - start_frame)
        assert sf.frame_idx >= 0
        assert sf.frame_idx < (end_frame - start_frame)


def test_labels_sessions():
    labels = Labels()
    assert labels.sessions == []
    labels.__str__()

    session = RecordingSession()
    labels.sessions.append(session)
    assert labels.sessions == [session]
    labels.__str__()


def test_labels_numpy_user_instances():
    """Test the user_instances parameter in Labels.numpy method."""
    # Create a simple skeleton for testing
    skeleton = Skeleton(["A", "B"])
    video = Video(filename="test_video.mp4")

    # Create 3 tracks
    track1 = Track(name="track1")
    track2 = Track(name="track2")
    track3 = Track(name="track3")

    # Create 3 frames with different combinations of user and predicted instances
    frames = []

    # Frame 0: User and predicted instance with same track (track1)
    #          Predicted instance with track2
    lf0 = LabeledFrame(video=video, frame_idx=0)
    user_inst0 = Instance([[10, 10], [20, 20]], skeleton=skeleton, track=track1)
    # Create predicted instance with point scores
    pred_inst0 = PredictedInstance.from_numpy(
        [[11, 11], [21, 21]],
        skeleton=skeleton,
        point_scores=[0.8, 0.8],  # Adding point scores
        score=0.8,
        track=track1,
    )
    pred_inst1 = PredictedInstance.from_numpy(
        [[30, 30], [40, 40]],
        skeleton=skeleton,
        point_scores=[0.9, 0.9],  # Adding point scores
        score=0.9,
        track=track2,
    )
    lf0.instances = [user_inst0, pred_inst0, pred_inst1]
    frames.append(lf0)

    # Frame 1: User instance linked to predicted via from_predicted (no track)
    #          Another predicted instance with track3
    lf1 = LabeledFrame(video=video, frame_idx=1)
    pred_inst2 = PredictedInstance.from_numpy(
        [[12, 12], [22, 22]],
        skeleton=skeleton,
        point_scores=[0.7, 0.7],  # Adding point scores
        score=0.7,
    )
    user_inst1 = Instance([[15, 15], [25, 25]], skeleton=skeleton)
    user_inst1.from_predicted = pred_inst2
    pred_inst3 = PredictedInstance.from_numpy(
        [[35, 35], [45, 45]],
        skeleton=skeleton,
        point_scores=[0.85, 0.85],  # Adding point scores
        score=0.85,
        track=track3,
    )
    lf1.instances = [pred_inst2, user_inst1, pred_inst3]
    frames.append(lf1)

    # Frame 2: Single user instance and single predicted instance (trivial case)
    lf2 = LabeledFrame(video=video, frame_idx=2)
    user_inst2 = Instance([[50, 50], [60, 60]], skeleton=skeleton)
    pred_inst4 = PredictedInstance.from_numpy(
        [[55, 55], [65, 65]],
        skeleton=skeleton,
        point_scores=[0.95, 0.95],  # Adding point scores
        score=0.95,
    )
    lf2.instances = [user_inst2, pred_inst4]
    frames.append(lf2)

    # Create labels with all these frames
    labels = Labels(labeled_frames=frames)
    labels.tracks = [track1, track2, track3]

    # Test 1: With user_instances=True (default)
    # For tracked instances
    tracks = labels.numpy(untracked=False)
    # Shape should be (3 frames, 3 tracks, 2 nodes, 2 coordinates)
    assert tracks.shape == (3, 3, 2, 2)
    # Track1 in frame0 should be the user instance
    assert_equal(tracks[0, 0], [[10, 10], [20, 20]])
    # Track2 in frame0 should be the predicted instance
    assert_equal(tracks[0, 1], [[30, 30], [40, 40]])
    # Track3 in frame1 should be the predicted instance
    assert_equal(tracks[1, 2], [[35, 35], [45, 45]])

    # With confidence scores
    tracks_conf = labels.numpy(untracked=False, return_confidence=True)
    # Shape should be (3 frames, 3 tracks, 2 nodes, 3 values)
    assert tracks_conf.shape == (3, 3, 2, 3)
    # User instance should have confidence 1.0
    assert_equal(tracks_conf[0, 0, 0, 2], 1.0)
    assert_equal(tracks_conf[0, 0, 1, 2], 1.0)
    # Predicted instance should have its original confidence
    assert_allclose(tracks_conf[0, 1, 0, 2], 0.9)

    # Test 2: For untracked instances
    untracked = labels.numpy(untracked=True)
    # Shape should be (3 frames, max_instances_per_frame=2 [user and predicted],
    # 2 nodes, 2 coordinates)
    assert untracked.shape == (3, 2, 2, 2)
    # Frame0 should have user instance first, then predicted instance track2
    assert_equal(untracked[0, 0], [[10, 10], [20, 20]])
    assert_equal(untracked[0, 1], [[30, 30], [40, 40]])
    # Frame1 should have user instance first, then predicted instance track3
    assert_equal(untracked[1, 0], [[15, 15], [25, 25]])
    assert_equal(untracked[1, 1], [[35, 35], [45, 45]])
    # Frame2 should have both instances
    assert_equal(untracked[2, 0], [[50, 50], [60, 60]])
    assert_equal(untracked[2, 1], [[55, 55], [65, 65]])

    # Test 3: with return_confidence=True
    untracked_conf = labels.numpy(untracked=True, return_confidence=True)
    # Shape should be (3 frames, max_instances_per_frame=2 [user and predicted],
    # 2 nodes, 3 values)
    assert untracked_conf.shape == (3, 2, 2, 3)
    # Frame0 should have user instance first, then predicted instance track2
    assert_equal(untracked_conf[0, 0, 0, 2], 1.0)
    assert_equal(untracked_conf[0, 0, 1, 2], 1.0)
    # Predicted instance should have its original confidence
    assert_allclose(untracked_conf[0, 1, 0, 2], 0.9)
    # Frame1 should have user instance first, then predicted instance track3
    assert_equal(untracked_conf[1, 0, 0, 2], 1.0)
    assert_equal(untracked_conf[1, 0, 1, 2], 1.0)
    # Predicted instance should have its original confidence
    assert_allclose(untracked_conf[1, 1, 0, 2], 0.85)
    # Frame2 should have both instances
    assert_equal(untracked_conf[2, 0, 0, 2], 1.0)
    assert_equal(untracked_conf[2, 0, 1, 2], 1.0)
    assert_allclose(untracked_conf[2, 1, 0, 2], 0.95)

    # Test 4: With user_instances=False
    # For tracked instances
    pred_only_tracks = labels.numpy(untracked=False, user_instances=False)
    # Shape should be (3 frames, 3 tracks, 2 nodes, 2 coordinates)
    assert pred_only_tracks.shape == (3, 3, 2, 2)
    # Track1 in frame0 should be the predicted instance now
    assert_equal(pred_only_tracks[0, 0], [[11, 11], [21, 21]])

    # Test 5: For untracked instances with user_instances=False
    pred_only_untracked = labels.numpy(untracked=True, user_instances=False)
    # Shape should be (3 frames, max_predicted_instances_per_frame=2, 2 nodes,
    # 2 coordinates)
    assert pred_only_untracked.shape == (3, 2, 2, 2)
    # Frame0 should have both predicted instances
    assert_equal(pred_only_untracked[0, 0], [[11, 11], [21, 21]])
    assert_equal(pred_only_untracked[0, 1], [[30, 30], [40, 40]])
    # Frame1 should have both predicted instances
    assert_equal(pred_only_untracked[1, 0], [[12, 12], [22, 22]])
    assert_equal(pred_only_untracked[1, 1], [[35, 35], [45, 45]])
    # Frame2 should have only one predicted instance
    assert_equal(pred_only_untracked[2, 0], [[55, 55], [65, 65]])
    assert np.isnan(pred_only_untracked[2, 1, 0, 0])  # Second slot should be empty

    # Test 6: Single instance project (the trivial case)
    # Create a single instance project
    single_frames = []
    lf_single = LabeledFrame(video=video, frame_idx=0)
    user_inst_single = Instance([[70, 70], [80, 80]], skeleton=skeleton)
    pred_inst_single = PredictedInstance.from_numpy(
        [[75, 75], [85, 85]],
        skeleton=skeleton,
        point_scores=[0.9, 0.9],  # Adding point scores
        score=0.9,
    )
    lf_single.instances = [user_inst_single, pred_inst_single]
    single_frames.append(lf_single)

    labels_single = Labels(labeled_frames=single_frames)

    # For single instance projects, user instances should be preferred
    single_tracks = labels_single.numpy()
    # Shape should be (1 frame, 1 instance, 2 nodes, 2 coordinates)
    assert single_tracks.shape == (1, 1, 2, 2)
    # Should be the user instance
    assert_equal(single_tracks[0, 0], [[70, 70], [80, 80]])

    # With user_instances=False
    single_tracks_pred = labels_single.numpy(user_instances=False)
    # Should be the predicted instance
    assert_equal(single_tracks_pred[0, 0], [[75, 75], [85, 85]])


def test_update_from_numpy(labels_predictions):
    """Test updating instances from numpy arrays."""
    import numpy as np

    from sleap_io import Track

    # Get original numpy representation
    original_arr = labels_predictions.numpy(return_confidence=True)

    # Modify the numpy array - shift all points by (10, 20)
    modified_arr = original_arr.copy()
    modified_arr[:, :, :, 0] += 10  # shift x by 10
    modified_arr[:, :, :, 1] += 20  # shift y by 20

    # Update the labels with modified array
    labels_predictions.update_from_numpy(modified_arr)

    # Get the updated numpy representation
    updated_arr = labels_predictions.numpy(return_confidence=True)

    # Verify points were updated correctly
    assert np.allclose(
        updated_arr[:, :, :, :2], modified_arr[:, :, :, :2], equal_nan=True
    )

    # Test creating new instances
    # Make a copy of the original labels
    import copy

    labels_copy = copy.deepcopy(labels_predictions)

    # Create new data for a non-existent track
    new_track = Track("new_track")
    labels_copy.tracks.append(new_track)

    # Add a new track to the array
    n_frames, n_tracks, n_nodes, n_dims = original_arr.shape
    new_arr = np.full(
        (n_frames, n_tracks + 1, n_nodes, n_dims), np.nan, dtype="float32"
    )
    new_arr[:, :-1] = modified_arr

    # Add data for the new track in the first frame
    new_arr[0, -1, :, 0] = 100  # x coordinates
    new_arr[0, -1, :, 1] = 200  # y coordinates
    new_arr[0, -1, :, 2] = 0.95  # confidence

    # Update with the new array
    tracks = labels_copy.tracks
    labels_copy.update_from_numpy(new_arr, tracks=tracks)

    # Verify the new instance was created
    first_frame = labels_copy.labeled_frames[0]

    # Check if a track with the name "new_track" exists in any of the first
    # frame's instances
    has_new_track = any(
        inst.track and inst.track.name == "new_track" for inst in first_frame.instances
    )

    # This should now pass
    assert has_new_track, "New track instance not found in frame instances"

    # Also check in predicted_instances
    has_new_track_pred = any(
        inst.track and inst.track.name == "new_track"
        for inst in first_frame.predicted_instances
    )

    # This should also pass
    assert has_new_track_pred, "New track instance not found in predicted_instances"


def test_update_from_numpy_errors():
    """Test error handling in update_from_numpy."""
    import numpy as np
    import pytest

    from sleap_io import Labels, Skeleton, Track, Video

    # Create a basic labels object
    labels = Labels()
    labels.videos.append(Video("test1.mp4"))
    labels.videos.append(Video("test2.mp4"))
    labels.skeletons.append(Skeleton(["A", "B"]))

    # 1. Test array with incorrect dimensions
    with pytest.raises(ValueError, match="Array must have 4 dimensions"):
        # Create a 3D array instead of 4D
        invalid_arr = np.zeros((2, 3, 2))
        labels.update_from_numpy(invalid_arr)

    # 2. Test multiple videos but no video specified
    with pytest.raises(ValueError, match="Video must be specified"):
        # Valid 4D array but no video specified with multiple videos
        valid_arr = np.zeros((2, 1, 2, 3))
        labels.update_from_numpy(valid_arr)

    # 3. Test tracks mismatch
    with pytest.raises(ValueError, match="Number of tracks in array .* doesn't match"):
        # Valid array with more tracks than in labels
        labels.tracks = [Track("track1")]
        valid_arr = np.zeros((2, 2, 2, 3))  # 2 tracks in array, 1 in labels
        labels.update_from_numpy(valid_arr, video=labels.videos[0])

    # 4. Test no skeletons
    labels_no_skeleton = Labels()
    labels_no_skeleton.videos.append(Video("test.mp4"))
    # Add a track to match the array dimension to avoid the track mismatch error
    labels_no_skeleton.tracks.append(Track("track1"))
    with pytest.raises(ValueError, match="No skeletons available"):
        valid_arr = np.zeros((2, 1, 2, 3))
        labels_no_skeleton.update_from_numpy(valid_arr)


def test_update_from_numpy_no_create_missing(labels_predictions):
    """Test update_from_numpy with create_missing=False."""
    # Get original numpy representation and copy labels
    original_arr = labels_predictions.numpy(return_confidence=True)
    labels_copy = copy.deepcopy(labels_predictions)

    # Count initial frames
    initial_frame_count = len(labels_copy.labeled_frames)

    # Create modified array with new frame indices
    # This extends the array to have frames beyond what currently exists
    n_frames, n_tracks, n_nodes, n_dims = original_arr.shape
    extended_arr = np.full(
        (n_frames + 3, n_tracks, n_nodes, n_dims), np.nan, dtype="float32"
    )
    extended_arr[:n_frames] = original_arr

    # Add data for a new frame that doesn't exist yet
    extended_arr[n_frames, 0, :, 0] = 100  # x coordinates
    extended_arr[n_frames, 0, :, 1] = 200  # y coordinates
    extended_arr[n_frames, 0, :, 2] = 0.9  # confidence

    # Update with create_missing=False
    labels_copy.update_from_numpy(extended_arr, create_missing=False)

    # The frame count should not have changed
    assert len(labels_copy.labeled_frames) == initial_frame_count, (
        "New frames should not be created with create_missing=False"
    )


def test_update_from_numpy_update_user_instances(labels_predictions):
    """Test updating user instances with update_from_numpy."""
    # Get original data
    labels_copy = copy.deepcopy(labels_predictions)
    video = labels_copy.videos[0]
    skeleton = labels_copy.skeletons[0]
    tracks = labels_copy.tracks

    # Find an existing frame to modify
    existing_frame = labels_copy.labeled_frames[0]

    # Clear and add our test instance
    existing_frame.instances = []

    # Create points data with the right shape for this skeleton
    points_data = np.full((len(skeleton.nodes), 2), np.nan)
    # Set only a few points with valid data
    points_data[0] = [50.0, 60.0]
    points_data[1] = [70.0, 80.0]

    # Add a user instance with the first track
    user_instance = Instance(points=points_data, skeleton=skeleton, track=tracks[0])
    existing_frame.instances = [user_instance]

    # Create array with modified data for this frame
    n_frames = 1
    arr = np.full(
        (n_frames, len(tracks), len(skeleton.nodes), 3), np.nan, dtype="float32"
    )

    # Set coordinates for first track (index 0)
    arr[0, 0, 0, 0] = 150.0  # New x for first point
    arr[0, 0, 0, 1] = 160.0  # New y for first point
    arr[0, 0, 1, 0] = 170.0  # New x for second point
    arr[0, 0, 1, 1] = 180.0  # New y for second point
    arr[0, 0, :, 2] = 1.0  # Confidence scores

    # Update with our array (which will update the first frame in the labels)
    labels_copy.update_from_numpy(arr, video=video)

    # Find the updated instance in the first frame
    updated_instance = None
    for inst in labels_copy.labeled_frames[0].instances:
        if inst.track == tracks[0]:
            updated_instance = inst
            break

    assert updated_instance is not None, "User instance not found in updated frame"

    # Verify the user instance was updated
    points = updated_instance.numpy()
    assert np.allclose(points[0], [150.0, 160.0]), (
        "User instance first point should be updated"
    )
    assert np.allclose(points[1], [170.0, 180.0]), (
        "User instance second point should be updated"
    )


def test_update_from_numpy_without_confidence():
    """Test update_from_numpy with array without confidence scores."""
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array WITHOUT confidence scores (only x,y)
    arr = np.full((1, 1, 2, 2), np.nan, dtype="float32")
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 1, 0] = 30.0  # x for second point
    arr[0, 0, 1, 1] = 40.0  # y for second point

    # Update with the array
    labels.update_from_numpy(arr)

    # Verify a new instance was created with correct points
    assert len(labels[0].instances) == 1, "Should create one instance"
    points = labels[0].instances[0].numpy()
    assert np.allclose(points[0], [10.0, 20.0]), "First point should be set correctly"
    assert np.allclose(points[1], [30.0, 40.0]), "Second point should be set correctly"


def test_update_from_numpy_int_video_index():
    """Test update_from_numpy with integer video index."""
    import numpy as np

    from sleap_io import Labels, Skeleton, Track, Video

    # Create a labels object with multiple videos
    labels = Labels()
    video1 = Video("test1.mp4")
    video2 = Video("test2.mp4")
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")

    labels.videos.append(video1)
    labels.videos.append(video2)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create array with data
    arr = np.full((1, 1, 2, 3), np.nan, dtype="float32")
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 0, 2] = 1.0  # confidence

    # Update using the second video by index (1)
    labels.update_from_numpy(arr, video=1)

    # Verify a new frame was created for the second video
    assert len(labels.labeled_frames) == 1, "Should create one frame"
    assert labels.labeled_frames[0].video == video2, (
        "Should create frame for second video"
    )


def test_update_from_numpy_special_case():
    """Test the special case handling in update_from_numpy."""
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("new_track")  # This will be the "new" track

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)  # Add both tracks

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than we'll specify in the tracks parameter
    arr = np.full((1, 2, 2, 3), np.nan, dtype="float32")
    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 0, 2] = 1.0  # confidence

    # Data for "additional" track in last column
    arr[0, 1, 0, 0] = 30.0  # x for first point
    arr[0, 1, 0, 1] = 40.0  # y for first point
    arr[0, 1, 0, 2] = 0.9  # confidence

    # Update with ONLY the first track in the tracks parameter
    # This will trigger the special case where n_tracks_arr > len(tracks)
    provided_tracks = [
        track1,
        track2,
    ]  # Only specifying one track for a two-track array
    labels.update_from_numpy(arr, tracks=provided_tracks)

    # Verify instances were created for both tracks
    assert len(labels[0].instances) == 2, "Should have two instances"

    # Find instance with track1
    track1_instance = next(
        (inst for inst in labels[0].instances if inst.track == track1), None
    )
    assert track1_instance is not None, "No instance found for track1"
    assert np.allclose(track1_instance.numpy()[0], [10.0, 20.0]), (
        "First track instance not updated correctly"
    )

    # Find instance with track2
    track2_instance = next(
        (inst for inst in labels[0].instances if inst.track == track2), None
    )
    assert track2_instance is not None, "No instance created for track2"
    assert np.allclose(track2_instance.numpy()[0], [30.0, 40.0]), (
        "Second track instance not created correctly"
    )


def test_update_from_numpy_confidence_scores():
    """Test updating confidence scores in existing predicted instances.

    Uses update_from_numpy to update scores.
    """
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create a frame with a predicted instance
    frame = LabeledFrame(video=video, frame_idx=0)
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.8]),
        score=0.75,
        track=track,
    )
    frame.instances.append(pred_inst)
    labels.append(frame)

    # Create array with updated confidence scores but same positions
    arr = np.full((1, 1, 2, 3), np.nan, dtype="float32")
    arr[0, 0, 0, 0] = 10.0  # same x
    arr[0, 0, 0, 1] = 20.0  # same y
    arr[0, 0, 0, 2] = 0.95  # updated confidence
    arr[0, 0, 1, 0] = 30.0  # same x
    arr[0, 0, 1, 1] = 40.0  # same y
    arr[0, 0, 1, 2] = 0.98  # updated confidence

    # Update the labels with the array
    labels.update_from_numpy(arr)

    # Verify the instance's confidence scores were updated
    updated_inst = labels[0].instances[0]
    assert isinstance(updated_inst, PredictedInstance), (
        "Instance should remain a PredictedInstance"
    )
    assert np.isclose(updated_inst["A"]["score"], 0.95), (
        "First node confidence score should be updated"
    )
    assert np.isclose(updated_inst["B"]["score"], 0.98), (
        "Second node confidence score should be updated"
    )

    # Check with numpy method that includes scores
    points_with_scores = updated_inst.numpy(scores=True)
    assert np.isclose(points_with_scores[0, 2], 0.95), (
        "First node confidence should be updated in numpy output"
    )
    assert np.isclose(points_with_scores[1, 2], 0.98), (
        "Second node confidence should be updated in numpy output"
    )


def test_update_from_numpy_inferred_tracks():
    """Test update_from_numpy using tracks inferred from labels."""
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("track2")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)

    # Create a frame with an existing instance
    frame = LabeledFrame(video=video, frame_idx=0)
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.8]),
        score=0.75,
        track=track1,
    )
    frame.instances.append(pred_inst)
    labels.append(frame)

    # Create array matching the number of tracks in labels
    arr = np.full((1, 2, 2, 3), np.nan, dtype="float32")
    # Data for first track
    arr[0, 0, 0, 0] = 15.0  # updated x
    arr[0, 0, 0, 1] = 25.0  # updated y
    arr[0, 0, 0, 2] = 0.9  # confidence

    # Data for second track
    arr[0, 1, 0, 0] = 50.0  # x
    arr[0, 1, 0, 1] = 60.0  # y
    arr[0, 1, 0, 2] = 0.85  # confidence

    # Update WITHOUT specifying tracks explicitly - should use the tracks from labels
    labels.update_from_numpy(arr)

    # Verify both tracks have instances
    instances = labels[0].instances
    assert len(instances) == 2, "Should have two instances"

    # Find instance with track1
    track1_instance = next((inst for inst in instances if inst.track == track1), None)
    assert track1_instance is not None, "No instance found for track1"
    assert np.allclose(track1_instance.numpy()[0], [15.0, 25.0]), (
        "First track instance not updated correctly"
    )

    # Find instance with track2
    track2_instance = next((inst for inst in instances if inst.track == track2), None)
    assert track2_instance is not None, "No instance created for track2"
    assert np.allclose(track2_instance.numpy()[0], [50.0, 60.0]), (
        "Second track instance not created correctly"
    )


def test_update_from_numpy_special_case_new_track():
    """Test the special case for adding a new track in update_from_numpy."""
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    new_track = Track("new_track")  # This will be added later

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(new_track)

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than current labels.tracks
    arr = np.full((1, 2, 2, 3), np.nan, dtype="float32")
    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 0, 2] = 0.9  # confidence

    # Data for new track
    arr[0, 1, 0, 0] = 30.0  # x for first point
    arr[0, 1, 0, 1] = 40.0  # y for first point
    arr[0, 1, 0, 2] = 0.8  # confidence

    # Update with the array - this should trigger the special case
    tracks = labels.tracks
    labels.update_from_numpy(arr, tracks=tracks)

    # Verify the instance for the new track was created
    assert len(labels[0].instances) == 2, "Should create instances for both tracks"

    # Find the new track instance
    new_track_instance = next(
        (inst for inst in labels[0].instances if inst.track == new_track), None
    )

    # Verify it was created and has the right data
    assert new_track_instance is not None, "New track instance should be created"
    assert np.allclose(new_track_instance.numpy()[0], [30.0, 40.0]), (
        "New track data should be set correctly"
    )
    assert np.isclose(new_track_instance.numpy(scores=True)[0, 2], 0.8), (
        "New track confidence should be set"
    )


def test_update_from_numpy_nan_handling():
    """Test handling of NaN values in update_from_numpy."""
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B", "C"])
    track = Track("track1")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create a frame with a predicted instance that has some points
    frame = LabeledFrame(video=video, frame_idx=0)
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.8, 0.9]),
        score=0.8,
        track=track,
    )
    frame.instances.append(pred_inst)
    labels.append(frame)

    # Create array with some NaN values
    arr = np.full((1, 1, 3, 3), np.nan, dtype="float32")
    # First point has values
    arr[0, 0, 0, 0] = 15.0  # updated x
    arr[0, 0, 0, 1] = 25.0  # updated y
    arr[0, 0, 0, 2] = 0.95  # updated confidence

    # Second point has NaN (should not update the existing point)
    arr[0, 0, 1, 0] = np.nan
    arr[0, 0, 1, 1] = np.nan
    arr[0, 0, 1, 2] = np.nan

    # Third point has values
    arr[0, 0, 2, 0] = 55.0  # updated x
    arr[0, 0, 2, 1] = 65.0  # updated y
    arr[0, 0, 2, 2] = 0.98  # updated confidence

    # Update the labels with the array
    labels.update_from_numpy(arr)

    # Verify only non-NaN values were updated
    updated_inst = labels[0].instances[0]
    points = updated_inst.numpy()

    # First point should be updated
    assert np.allclose(points[0], [15.0, 25.0]), "First point should be updated"

    # Second point should remain unchanged
    assert np.allclose(points[1], [30.0, 40.0]), "Second point should remain unchanged"

    # Third point should be updated
    assert np.allclose(points[2], [55.0, 65.0]), "Third point should be updated"

    # Check confidence scores
    if isinstance(updated_inst, PredictedInstance):
        scores = updated_inst.numpy(scores=True)[:, 2]
        assert np.isclose(scores[0], 0.95), "First point confidence should be updated"
        assert np.isclose(scores[1], 0.8), (
            "Second point confidence should remain unchanged"
        )
        assert np.isclose(scores[2], 0.98), "Third point confidence should be updated"


def test_update_from_numpy_more_tracks_than_provided():
    """Test update_from_numpy special case with more tracks in array.

    Tests when array has more tracks than the provided track list.
    """
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("track2")
    track3 = Track("track3")  # Will be the last track in provided list

    # Add to labels
    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)
    labels.tracks.append(track3)

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than we'll provide explicitly
    # Shape: (n_frames=1, n_tracks=3, n_nodes=2, n_dims=3)
    arr = np.full((1, 3, 2, 3), np.nan, dtype="float32")

    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first node
    arr[0, 0, 0, 1] = 20.0  # y for first node
    arr[0, 0, 0, 2] = 0.8  # confidence

    # Data for second track
    arr[0, 1, 0, 0] = 30.0  # x for first node
    arr[0, 1, 0, 1] = 40.0  # y for first node
    arr[0, 1, 0, 2] = 0.9  # confidence

    # Data for third track in array
    arr[0, 2, 0, 0] = 50.0  # x for first node
    arr[0, 2, 0, 1] = 60.0  # y for first node
    arr[0, 2, 0, 2] = 1.0  # confidence

    # The key to hit the special case: provide a tracks list SHORTER than array
    # tracks dimension
    provided_tracks = [track1, track3]  # Only providing track1 and track3

    # Update with our array - this will trigger the special case
    labels.update_from_numpy(arr, tracks=provided_tracks)

    # Verify track1's instance was created correctly
    # First track in provided_tracks matches first column in array
    track1_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track1.name:
            track1_instance = inst
            break

    assert track1_instance is not None, "track1 instance should be created"
    assert np.allclose(track1_instance.numpy()[0], [10.0, 20.0]), (
        "Track1 coordinates should match"
    )

    # Verify track3's instance was created correctly
    track3_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track3.name:
            track3_instance = inst
            break

    # Based on how the special case works in the implementation,
    # track3 (last in provided_tracks) should be assigned the data from arr[0, 1]
    # (i.e., the second column in the array)
    assert track3_instance is not None, "track3 instance should be created"
    assert np.allclose(track3_instance.numpy()[0], [30.0, 40.0]), (
        "Track3 coordinates should match"
    )

    # Verify there's no extra instance with track2
    track2_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track2.name:
            track2_instance = inst
            break

    assert track2_instance is None, "Should not create an instance for track2"


def test_update_from_numpy_special_case_without_confidence():
    """Test update_from_numpy special case with more tracks in array.

    Tests case without confidence scores.
    """
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("track2")  # Will be passed in tracks list
    track3 = Track("track3")  # Will be used as the new track

    # Add to labels
    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)
    labels.tracks.append(track3)  # Add to labels.tracks

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than we'll provide explicitly
    # Shape: (n_frames=1, n_tracks=3, n_nodes=2, n_dims=2) - NO CONFIDENCE SCORES
    arr = np.full((1, 3, 2, 2), np.nan, dtype="float32")

    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first node
    arr[0, 0, 0, 1] = 20.0  # y for first node

    # Data for second track
    arr[0, 1, 0, 0] = 30.0  # x for first node
    arr[0, 1, 0, 1] = 40.0  # y for first node

    # Data for third track (will be matched with track3, the last in provided_tracks)
    arr[0, 2, 0, 0] = 50.0  # x for first node
    arr[0, 2, 0, 1] = 60.0  # y for first node

    # The key to hit the special case: provide a tracks list SHORTER than array
    # tracks dimension
    # and ensure we're testing the "else:" branch (no confidence scores)
    provided_tracks = [track1, track3]  # Only providing track1 and track3

    # Update with our array - this will trigger the special case without confidence
    labels.update_from_numpy(arr, tracks=provided_tracks)

    # Verify track1's instance was created correctly (first track in provided_tracks)
    track1_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track1.name:
            track1_instance = inst
            break

    assert track1_instance is not None, "track1 instance should be created"
    assert np.allclose(track1_instance.numpy()[0], [10.0, 20.0]), (
        "Track1 coordinates should match"
    )

    # Verify track3's instance was created correctly using data from the last column
    track3_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track3.name:
            track3_instance = inst
            break

    assert track3_instance is not None, "track3 instance should be created"
    assert np.allclose(track3_instance.numpy()[0], [30.0, 40.0]), (
        "Track3 coordinates should match"
    )

    # Verify that confidence scores were set to 1.0 by default
    if isinstance(track3_instance, PredictedInstance):
        # Convert the points to a numpy array with scores
        points_with_scores = track3_instance.numpy(scores=True)
        # Check if any scores are close to 1.0 (default value)
        assert np.isclose(points_with_scores[0, 2], 1.0), (
            "Default confidence score should be 1.0"
        )


def test_update_from_numpy_special_case_new_track_v2():
    """Test the special case for adding a new track in update_from_numpy."""
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    new_track = Track("new_track")  # This will be added later

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(new_track)

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than current labels.tracks
    arr = np.full((1, 2, 2, 3), np.nan, dtype="float32")
    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first point
    arr[0, 0, 0, 1] = 20.0  # y for first point
    arr[0, 0, 0, 2] = 0.9  # confidence

    # Data for new track
    arr[0, 1, 0, 0] = 30.0  # x for first point
    arr[0, 1, 0, 1] = 40.0  # y for first point
    arr[0, 1, 0, 2] = 0.8  # confidence

    # Update with the array - this should trigger the special case
    tracks = labels.tracks
    labels.update_from_numpy(arr, tracks=tracks)

    # Verify the instance for the new track was created
    assert len(labels[0].instances) == 2, "Should create instances for both tracks"

    # Find the new track instance
    new_track_instance = next(
        (inst for inst in labels[0].instances if inst.track == new_track), None
    )

    # Verify it was created and has the right data
    assert new_track_instance is not None, "New track instance should be created"
    assert np.allclose(new_track_instance.numpy()[0], [30.0, 40.0]), (
        "New track data should be set correctly"
    )
    assert np.isclose(new_track_instance.numpy(scores=True)[0, 2], 0.8), (
        "New track confidence should be set"
    )


def test_update_from_numpy_nan_handling_v2():
    """Test handling of NaN values in update_from_numpy."""
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B", "C"])
    track = Track("track1")

    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track)

    # Create a frame with a predicted instance that has some points
    frame = LabeledFrame(video=video, frame_idx=0)
    pred_inst = PredictedInstance.from_numpy(
        points_data=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        skeleton=skeleton,
        point_scores=np.array([0.7, 0.8, 0.9]),
        score=0.8,
        track=track,
    )
    frame.instances.append(pred_inst)
    labels.append(frame)

    # Create array with some NaN values
    arr = np.full((1, 1, 3, 3), np.nan, dtype="float32")
    # First point has values
    arr[0, 0, 0, 0] = 15.0  # updated x
    arr[0, 0, 0, 1] = 25.0  # updated y
    arr[0, 0, 0, 2] = 0.95  # updated confidence

    # Second point has NaN (should not update the existing point)
    arr[0, 0, 1, 0] = np.nan
    arr[0, 0, 1, 1] = np.nan
    arr[0, 0, 1, 2] = np.nan

    # Third point has values
    arr[0, 0, 2, 0] = 55.0  # updated x
    arr[0, 0, 2, 1] = 65.0  # updated y
    arr[0, 0, 2, 2] = 0.98  # updated confidence

    # Update the labels with the array
    labels.update_from_numpy(arr)

    # Verify only non-NaN values were updated
    updated_inst = labels[0].instances[0]
    points = updated_inst.numpy()

    # First point should be updated
    assert np.allclose(points[0], [15.0, 25.0]), "First point should be updated"

    # Second point should remain unchanged
    assert np.allclose(points[1], [30.0, 40.0]), "Second point should remain unchanged"

    # Third point should be updated
    assert np.allclose(points[2], [55.0, 65.0]), "Third point should be updated"

    # Check confidence scores
    if isinstance(updated_inst, PredictedInstance):
        scores = updated_inst.numpy(scores=True)[:, 2]
        assert np.isclose(scores[0], 0.95), "First point confidence should be updated"
        assert np.isclose(scores[1], 0.8), (
            "Second point confidence should remain unchanged"
        )
        assert np.isclose(scores[2], 0.98), "Third point confidence should be updated"


def test_update_from_numpy_more_tracks_than_provided_v2():
    """Test update_from_numpy special case with more tracks in array.

    Tests when array has more tracks than the provided track list.
    """
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("track2")
    track3 = Track("track3")  # Will be the last track in provided list

    # Add to labels
    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)
    labels.tracks.append(track3)

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than we'll provide explicitly
    # Shape: (n_frames=1, n_tracks=3, n_nodes=2, n_dims=3)
    arr = np.full((1, 3, 2, 3), np.nan, dtype="float32")

    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first node
    arr[0, 0, 0, 1] = 20.0  # y for first node
    arr[0, 0, 0, 2] = 0.8  # confidence

    # Data for second track
    arr[0, 1, 0, 0] = 30.0  # x for first node
    arr[0, 1, 0, 1] = 40.0  # y for first node
    arr[0, 1, 0, 2] = 0.9  # confidence

    # Data for third track in array
    arr[0, 2, 0, 0] = 50.0  # x for first node
    arr[0, 2, 0, 1] = 60.0  # y for first node
    arr[0, 2, 0, 2] = 1.0  # confidence

    # The key to hit the special case: provide a tracks list SHORTER than array
    # tracks dimension
    provided_tracks = [track1, track3]  # Only providing track1 and track3

    # Update with our array - this will trigger the special case
    labels.update_from_numpy(arr, tracks=provided_tracks)

    # Verify track1's instance was created correctly
    # First track in provided_tracks matches first column in array
    track1_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track1.name:
            track1_instance = inst
            break

    assert track1_instance is not None, "track1 instance should be created"
    assert np.allclose(track1_instance.numpy()[0], [10.0, 20.0]), (
        "Track1 coordinates should match"
    )

    # Verify track3's instance was created correctly
    track3_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track3.name:
            track3_instance = inst
            break

    # Based on how the special case works in the implementation,
    # track3 (last in provided_tracks) should be assigned the data from arr[0, 1]
    # (i.e., the second column in the array)
    assert track3_instance is not None, "track3 instance should be created"
    assert np.allclose(track3_instance.numpy()[0], [30.0, 40.0]), (
        "Track3 coordinates should match"
    )

    # Verify there's no extra instance with track2
    track2_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track2.name:
            track2_instance = inst
            break

    assert track2_instance is None, "Should not create an instance for track2"


def test_update_from_numpy_special_case_without_confidence_v2():
    """Test update_from_numpy special case with more tracks in array.

    Tests case without confidence scores.
    """
    # Create a basic labels object
    labels = Labels()
    video = Video("test.mp4")
    skeleton = Skeleton(["A", "B"])
    track1 = Track("track1")
    track2 = Track("track2")  # Will be passed in tracks list
    track3 = Track("track3")  # Will be used as the new track

    # Add to labels
    labels.videos.append(video)
    labels.skeletons.append(skeleton)
    labels.tracks.append(track1)
    labels.tracks.append(track2)
    labels.tracks.append(track3)  # Add to labels.tracks

    # Create a frame
    frame = LabeledFrame(video=video, frame_idx=0)
    labels.append(frame)

    # Create array with MORE tracks than we'll provide explicitly
    # Shape: (n_frames=1, n_tracks=3, n_nodes=2, n_dims=2) - NO CONFIDENCE SCORES
    arr = np.full((1, 3, 2, 2), np.nan, dtype="float32")

    # Data for first track
    arr[0, 0, 0, 0] = 10.0  # x for first node
    arr[0, 0, 0, 1] = 20.0  # y for first node

    # Data for second track
    arr[0, 1, 0, 0] = 30.0  # x for first node
    arr[0, 1, 0, 1] = 40.0  # y for first node

    # Data for third track (will be matched with track3, the last in provided_tracks)
    arr[0, 2, 0, 0] = 50.0  # x for first node
    arr[0, 2, 0, 1] = 60.0  # y for first node

    # The key to hit the special case: provide a tracks list SHORTER than array
    # tracks dimension
    # and ensure we're testing the "else:" branch (no confidence scores)
    provided_tracks = [track1, track3]  # Only providing track1 and track3

    # Update with our array - this will trigger the special case without confidence
    labels.update_from_numpy(arr, tracks=provided_tracks)

    # Verify track1's instance was created correctly (first track in provided_tracks)
    track1_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track1.name:
            track1_instance = inst
            break

    assert track1_instance is not None, "track1 instance should be created"
    assert np.allclose(track1_instance.numpy()[0], [10.0, 20.0]), (
        "Track1 coordinates should match"
    )

    # Verify track3's instance was created correctly using data from the last column
    track3_instance = None
    for inst in labels[0].instances:
        if inst.track and inst.track.name == track3.name:
            track3_instance = inst
            break

    assert track3_instance is not None, "track3 instance should be created"
    assert np.allclose(track3_instance.numpy()[0], [30.0, 40.0]), (
        "Track3 coordinates should match"
    )

    # Verify that confidence scores were set to 1.0 by default
    if isinstance(track3_instance, PredictedInstance):
        # Convert the points to a numpy array with scores
        points_with_scores = track3_instance.numpy(scores=True)
        # Check if any scores are close to 1.0 (default value)
        assert np.isclose(points_with_scores[0, 2], 1.0), (
            "Default confidence score should be 1.0"
        )


def test_labels_numpy_with_confidence(labels_predictions: Labels):
    """Test the numpy method with confidence scores enabled."""
    # Test getting numpy array with confidence scores
    tracks_arr_with_conf = labels_predictions.numpy(return_confidence=True)
    assert tracks_arr_with_conf.shape[3] == 3  # x, y, confidence

    # Modify some confidence values
    modified_conf = tracks_arr_with_conf.copy()
    if not np.all(np.isnan(modified_conf[0, 0])):
        modified_conf[0, 0, :, 2] = (
            0.75  # Set confidence to 0.75 for first track, first frame
        )

    # Create a new labels object to test update_from_numpy
    new_labels = Labels()
    new_labels.videos = [labels_predictions.video]
    new_labels.skeletons = [labels_predictions.skeleton]
    new_labels.tracks = labels_predictions.tracks

    # Test update_from_numpy with confidence scores
    new_labels.update_from_numpy(modified_conf)

    # Verify confidence scores were updated
    for lf in new_labels.labeled_frames:
        for inst in lf.predicted_instances:
            if isinstance(inst, PredictedInstance):
                # Get numpy representation with scores
                points_with_scores = inst.numpy(scores=True)
                if points_with_scores.shape[1] == 3:  # Has scores column
                    # Look for updated confidence scores
                    if np.any(np.isclose(points_with_scores[:, 2], 0.75)):
                        return

    # If we get here, no updated confidence values were found
    assert False, "No updated confidence scores found after update_from_numpy"


def test_from_numpy_basic():
    """Test creating a Labels object from a numpy array using from_numpy classmethod."""
    # Create test data
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["head", "thorax", "abdomen"])

    # Create a simple array with 2 frames, 1 track, 3 nodes
    arr = np.zeros((2, 1, 3, 2), dtype=np.float32)

    # Set coordinates for first frame
    arr[0, 0, 0] = [10, 20]  # head
    arr[0, 0, 1] = [15, 25]  # thorax
    arr[0, 0, 2] = [20, 30]  # abdomen

    # Set coordinates for second frame
    arr[1, 0, 0] = [12, 22]  # head
    arr[1, 0, 1] = [17, 27]  # thorax
    arr[1, 0, 2] = [22, 32]  # abdomen

    # Create Labels object from the array
    labels = Labels.from_numpy(arr, videos=[video], skeletons=[skeleton])

    # Verify basic properties
    assert len(labels) == 2  # 2 frames
    assert len(labels.videos) == 1
    assert labels.videos[0] == video
    assert len(labels.skeletons) == 1
    assert labels.skeletons[0] == skeleton
    assert len(labels.tracks) == 1  # 1 auto-created track
    assert labels.tracks[0].name == "track_0"  # Default name

    # Verify the first frame data
    assert labels[0].frame_idx == 0
    assert len(labels[0].instances) == 1
    assert_allclose(
        labels[0].instances[0].numpy(), np.array([[10, 20], [15, 25], [20, 30]])
    )

    # Verify the second frame data
    assert labels[1].frame_idx == 1
    assert len(labels[1].instances) == 1
    assert_allclose(
        labels[1].instances[0].numpy(), np.array([[12, 22], [17, 27], [22, 32]])
    )


def test_from_numpy_with_single_skeleton():
    """Test from_numpy with a single Skeleton object rather than a list."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    arr = np.zeros((1, 1, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = [5, 10]
    arr[0, 0, 1] = [15, 20]

    # Pass skeleton directly instead of in a list
    labels = Labels.from_numpy(arr, videos=[video], skeletons=skeleton)

    assert len(labels.skeletons) == 1
    assert labels.skeleton == skeleton
    assert_allclose(labels[0].instances[0].numpy(), np.array([[5, 10], [15, 20]]))


def test_from_numpy_with_provided_tracks():
    """Test from_numpy with user-provided tracks."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    # Create custom tracks
    track1 = Track("custom_track_1")
    track2 = Track("custom_track_2")

    # Array with 2 tracks
    arr = np.zeros((1, 2, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = [1, 2]  # First track, first node
    arr[0, 0, 1] = [3, 4]  # First track, second node
    arr[0, 1, 0] = [5, 6]  # Second track, first node
    arr[0, 1, 1] = [7, 8]  # Second track, second node

    # Create with provided tracks
    labels = Labels.from_numpy(
        arr, videos=[video], skeletons=[skeleton], tracks=[track1, track2]
    )

    # Verify tracks were used
    assert len(labels.tracks) == 2
    assert labels.tracks[0] == track1
    assert labels.tracks[1] == track2

    # Verify instance data is correct
    inst1 = next(i for i in labels[0].instances if i.track == track1)
    inst2 = next(i for i in labels[0].instances if i.track == track2)

    assert_allclose(inst1.numpy(), np.array([[1, 2], [3, 4]]))
    assert_allclose(inst2.numpy(), np.array([[5, 6], [7, 8]]))


def test_from_numpy_partial_tracks():
    """Test from_numpy when fewer tracks are provided than in the array."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])
    track = Track("provided_track")

    # Array with 2 tracks
    arr = np.zeros((1, 2, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = [1, 2]
    arr[0, 0, 1] = [3, 4]
    arr[0, 1, 0] = [5, 6]
    arr[0, 1, 1] = [7, 8]

    # Only provide one track for a two-track array
    labels = Labels.from_numpy(
        arr, videos=[video], skeletons=[skeleton], tracks=[track]
    )

    # Should auto-create the missing track
    assert len(labels.tracks) == 2
    assert labels.tracks[0] == track
    assert labels.tracks[1].name == "track_0"

    # Verify data
    inst1 = next(i for i in labels[0].instances if i.track == track)
    inst2 = next(i for i in labels[0].instances if i.track == labels.tracks[1])

    assert_allclose(inst1.numpy(), np.array([[1, 2], [3, 4]]))
    assert_allclose(inst2.numpy(), np.array([[5, 6], [7, 8]]))


def test_from_numpy_first_frame():
    """Test from_numpy with custom first_frame parameter."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    arr = np.zeros((2, 1, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = [1, 2]
    arr[0, 0, 1] = [3, 4]
    arr[1, 0, 0] = [5, 6]
    arr[1, 0, 1] = [7, 8]

    # Use first_frame=100
    labels = Labels.from_numpy(
        arr, videos=[video], skeletons=[skeleton], first_frame=100
    )

    # Frame indices should start at 100
    assert len(labels) == 2
    assert labels[0].frame_idx == 100
    assert labels[1].frame_idx == 101

    # Verify data
    assert_allclose(labels[0].instances[0].numpy(), np.array([[1, 2], [3, 4]]))
    assert_allclose(labels[1].instances[0].numpy(), np.array([[5, 6], [7, 8]]))


def test_from_numpy_with_confidence():
    """Test from_numpy with confidence scores."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    # Create array with confidence scores (shape ending with 3)
    arr = np.zeros((1, 1, 2, 3), dtype=np.float32)
    arr[0, 0, 0] = [1, 2, 0.8]  # x, y, confidence
    arr[0, 0, 1] = [3, 4, 0.9]  # x, y, confidence

    # Create Labels with confidence
    labels = Labels.from_numpy(arr, videos=[video], skeletons=[skeleton])

    # Verify
    assert len(labels) == 1
    instance = labels[0].instances[0]
    assert isinstance(instance, PredictedInstance)

    # Check points
    points = instance.numpy()
    assert_allclose(points, np.array([[1, 2], [3, 4]]))

    # Check confidence scores
    scores = instance.numpy(scores=True)
    assert_allclose(scores[:, 2], np.array([0.8, 0.9]))


def test_from_numpy_with_return_confidence():
    """Test from_numpy with return_confidence parameter."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    # Create array WITHOUT confidence scores (shape ending with 2)
    arr = np.zeros((1, 1, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = [1, 2]  # x, y
    arr[0, 0, 1] = [3, 4]  # x, y

    # Create Labels with return_confidence=True
    labels = Labels.from_numpy(
        arr, videos=[video], skeletons=[skeleton], return_confidence=True
    )

    # Verify
    assert len(labels) == 1
    instance = labels[0].instances[0]
    assert isinstance(instance, PredictedInstance)

    # Check points
    points = instance.numpy()
    assert_allclose(points, np.array([[1, 2], [3, 4]]))

    # Check confidence scores (should be default 1.0)
    scores = instance.numpy(scores=True)
    assert_allclose(scores[:, 2], np.array([1.0, 1.0]))


def test_from_numpy_with_nan():
    """Test from_numpy with NaN values."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    # Create test data where all points for frame 1 are NaN
    arr = np.zeros((2, 1, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = [1, 2]  # Valid points in frame 0
    arr[0, 0, 1] = [3, 4]  # Valid points in frame 0
    # Make all points in frame 1 NaN
    arr[1, 0, 0] = [np.nan, np.nan]
    arr[1, 0, 1] = [np.nan, np.nan]

    # Create Labels
    labels = Labels.from_numpy(arr, videos=[video], skeletons=[skeleton])

    # Only one frame should be created since the other has all NaN values
    assert len(labels) == 1
    assert labels[0].frame_idx == 0
    assert_allclose(labels[0].instances[0].numpy(), np.array([[1, 2], [3, 4]]))

    # Test with partial NaNs in a track
    arr2 = np.zeros((2, 1, 2, 2), dtype=np.float32)
    arr2[0, 0, 0] = [1, 2]
    arr2[0, 0, 1] = [3, 4]
    arr2[1, 0, 0] = [np.nan, np.nan]  # NaN values for first node in frame 1
    arr2[1, 0, 1] = [7, 8]  # Valid values for second node in frame 1

    # Create Labels
    labels2 = Labels.from_numpy(arr2, videos=[video], skeletons=[skeleton])

    # Both frames should be created since frame 1 has at least one valid node
    assert len(labels2) == 2
    assert labels2[0].frame_idx == 0
    assert labels2[1].frame_idx == 1
    # First frame has both nodes with valid data
    assert_allclose(labels2[0].instances[0].numpy(), np.array([[1, 2], [3, 4]]))
    # Second frame has NaN for first node
    second_frame_points = labels2[1].instances[0].numpy()
    assert np.isnan(second_frame_points[0, 0])
    assert np.isnan(second_frame_points[0, 1])
    # But valid data for second node
    assert_allclose(second_frame_points[1], np.array([7, 8]))


def test_from_numpy_all_nan():
    """Test from_numpy with all NaN values."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    # Create array with all NaNs
    arr = np.full((2, 1, 2, 2), np.nan, dtype=np.float32)

    # Create Labels
    labels = Labels.from_numpy(arr, videos=[video], skeletons=[skeleton])

    # No frames should be created
    assert len(labels) == 0
    assert len(labels.videos) == 1
    assert len(labels.skeletons) == 1
    assert len(labels.tracks) == 1  # Track is still created


def test_from_numpy_multiple_videos():
    """Test from_numpy with multiple videos."""
    video1 = Video(filename="test_video1.mp4")
    video2 = Video(filename="test_video2.mp4")
    skeleton = Skeleton(["A", "B"])

    arr = np.zeros((1, 1, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = [1, 2]
    arr[0, 0, 1] = [3, 4]

    # Create Labels with multiple videos
    labels = Labels.from_numpy(arr, videos=[video1, video2], skeletons=[skeleton])

    # Should use the first video
    assert len(labels) == 1
    assert labels[0].video == video1
    assert len(labels.videos) == 2
    assert labels.videos == [video1, video2]


def test_from_numpy_errors():
    """Test error cases for from_numpy."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    # 1. Invalid array dimensions
    invalid_arr = np.zeros((2, 3, 2))  # 3D instead of 4D
    with pytest.raises(ValueError, match="Array must have 4 dimensions"):
        Labels.from_numpy(invalid_arr, videos=[video], skeletons=[skeleton])

    # 2. Missing videos
    valid_arr = np.zeros((2, 1, 2, 2))
    with pytest.raises(ValueError, match="At least one video must be provided"):
        Labels.from_numpy(valid_arr, videos=[], skeletons=[skeleton])

    # 3. Missing skeletons
    with pytest.raises(ValueError, match="At least one skeleton must be provided"):
        Labels.from_numpy(valid_arr, videos=[video], skeletons=None)

    with pytest.raises(ValueError, match="At least one skeleton must be provided"):
        Labels.from_numpy(valid_arr, videos=[video], skeletons=[])


def test_from_numpy_partial_nan_track():
    """Test from_numpy with one track having all NaN values in a frame."""
    video = Video(filename="test_video.mp4")
    skeleton = Skeleton(["A", "B"])

    # Create test data with 1 frame, 2 tracks, but one track has all NaN values
    arr = np.zeros((1, 2, 2, 2), dtype=np.float32)

    # First track has valid values
    arr[0, 0, 0] = [10, 20]  # First track, first node
    arr[0, 0, 1] = [30, 40]  # First track, second node

    # Second track has all NaN values
    arr[0, 1, 0] = [np.nan, np.nan]  # Second track, first node
    arr[0, 1, 1] = [np.nan, np.nan]  # Second track, second node

    # Create labels from this array
    labels = Labels.from_numpy(arr, videos=[video], skeletons=[skeleton])

    # Should still create both tracks in the Labels object
    assert len(labels.tracks) == 2

    # But only the first track should have an instance in the frame
    assert len(labels[0].instances) == 1
    assert labels[0].instances[0].track == labels.tracks[0]
    assert_allclose(labels[0].instances[0].numpy(), np.array([[10, 20], [30, 40]]))


def test_labels_merge_basic():
    """Test basic merge functionality."""
    # Create first Labels object
    skel1 = Skeleton(nodes=["head", "tail"])
    video1 = Video(filename="video1.mp4")
    track1 = Track(name="track1")

    labels1 = Labels()
    labels1.skeletons.append(skel1)
    labels1.videos.append(video1)
    labels1.tracks.append(track1)

    # Add a frame with an instance
    frame1 = LabeledFrame(
        video=video1,
        frame_idx=0,
        instances=[
            Instance.from_numpy(
                np.array([[10, 10], [20, 20]]), skeleton=skel1, track=track1
            )
        ],
    )
    labels1.append(frame1)

    # Create second Labels object
    skel2 = Skeleton(nodes=["head", "tail"])  # Same structure
    video2 = Video(filename="video2.mp4")  # Different video
    track2 = Track(name="track2")  # Different track

    labels2 = Labels()
    labels2.skeletons.append(skel2)
    labels2.videos.append(video2)
    labels2.tracks.append(track2)

    # Add a frame with an instance
    frame2 = LabeledFrame(
        video=video2,
        frame_idx=0,
        instances=[
            Instance.from_numpy(
                np.array([[30, 30], [40, 40]]), skeleton=skel2, track=track2
            )
        ],
    )
    labels2.append(frame2)

    # Merge labels2 into labels1 with explicit video matcher to ensure no matching
    video_matcher = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
    result = labels1.merge(labels2, video=video_matcher)

    # Check result
    assert result.successful
    assert result.frames_merged == 1
    assert result.instances_added == 1
    assert len(labels1.videos) == 2  # Both videos should be present
    assert len(labels1.tracks) == 2  # Both tracks should be present
    assert len(labels1.labeled_frames) == 2  # Both frames should be present


def test_labels_merge_skeleton_mismatch_strict():
    """Test merge with skeleton mismatch in strict mode."""
    # Create first Labels with one skeleton
    skel1 = Skeleton(nodes=["head", "tail"])
    labels1 = Labels()
    labels1.skeletons.append(skel1)

    # Create second Labels with different skeleton
    skel2 = Skeleton(nodes=["nose", "body", "tail"])  # Different structure
    labels2 = Labels()
    labels2.skeletons.append(skel2)

    # Try to merge with strict validation
    with pytest.raises(SkeletonMismatchError, match="No matching skeleton found"):
        labels1.merge(labels2, validate=True, error_mode="strict")


def test_labels_merge_skeleton_mismatch_warn(capsys):
    """Test merge with skeleton mismatch in warn mode."""
    # Create first Labels with one skeleton
    skel1 = Skeleton(nodes=["head", "tail"])
    labels1 = Labels()
    labels1.skeletons.append(skel1)

    # Create second Labels with different skeleton
    skel2 = Skeleton(nodes=["nose", "body", "tail"], name="different")
    labels2 = Labels()
    labels2.skeletons.append(skel2)

    # Merge with warn mode (should print warning but continue)
    result = labels1.merge(labels2, validate=True, error_mode="warn")

    assert result.successful
    assert len(labels1.skeletons) == 2  # New skeleton should be added

    # Check that a warning was printed
    captured = capsys.readouterr()
    assert "Warning: No matching skeleton" in captured.out


def test_labels_merge_with_progress_callback():
    """Test merge with progress callback."""
    # Create two Labels objects with frames
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video)

    # Add multiple frames to labels2
    for i in range(5):
        frame = LabeledFrame(
            video=video,
            frame_idx=i * 10,  # Different frame indices
            instances=[
                Instance.from_numpy(np.array([[i, i], [i + 10, i + 10]]), skeleton=skel)
            ],
        )
        labels2.append(frame)

    # Track progress
    progress_calls = []

    def progress_callback(current, total, message):
        progress_calls.append((current, total, message))

    # Merge with progress callback
    result = labels1.merge(labels2, progress_callback=progress_callback)

    assert result.successful
    assert len(progress_calls) == 6  # One call per frame + final
    assert progress_calls[0] == (0, 5, "Merging frame 1/5")
    assert progress_calls[-2] == (4, 5, "Merging frame 5/5")
    assert progress_calls[-1] == (5, 5, "Merge complete")


def test_labels_merge_video_not_matched():
    """Test merge when video is not matched - should add new video."""
    skel = Skeleton(nodes=["head", "tail"])
    video1 = Video(filename="video1.mp4")
    video2 = Video(filename="video2.mp4")

    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video1)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video2)

    # Add frame to labels2
    frame = LabeledFrame(
        video=video2,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel)],
    )
    labels2.append(frame)

    # Merge with strict path matching to ensure videos are not matched
    video_matcher = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
    result = labels1.merge(labels2, video=video_matcher)

    assert result.successful
    assert len(labels1.videos) == 2
    assert video2 in labels1.videos


def test_labels_merge_track_matching():
    """Test merge with track matching."""
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")
    track1 = Track(name="mouse1")
    track2 = Track(name="mouse1")  # Same name, different object

    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video)
    labels1.tracks.append(track1)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video)
    labels2.tracks.append(track2)

    # Add frames with tracked instances
    frame1 = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance.from_numpy(
                np.array([[10, 10], [20, 20]]), skeleton=skel, track=track1
            )
        ],
    )
    labels1.append(frame1)

    frame2 = LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[
            Instance.from_numpy(
                np.array([[15, 15], [25, 25]]), skeleton=skel, track=track2
            )
        ],
    )
    labels2.append(frame2)

    # Merge with track name matching
    track_matcher = TrackMatcher(method=TrackMatchMethod.NAME)
    result = labels1.merge(labels2, track=track_matcher)

    assert result.successful
    assert len(labels1.tracks) == 1  # Tracks should be matched by name


def test_labels_merge_conflict_resolution():
    """Test merge with instance conflicts."""
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video)

    # Add overlapping frames
    frame1 = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel),
            PredictedInstance.from_numpy(
                np.array([[30, 30], [40, 40]]), skeleton=skel, score=0.8
            ),
        ],
    )
    labels1.append(frame1)

    frame2 = LabeledFrame(
        video=video,
        frame_idx=0,  # Same frame index
        instances=[
            Instance.from_numpy(
                np.array([[11, 11], [21, 21]]), skeleton=skel
            ),  # Close to first
            PredictedInstance.from_numpy(
                np.array([[31, 31], [41, 41]]), skeleton=skel, score=0.9
            ),  # Better score
        ],
    )
    labels2.append(frame2)

    # Merge with spatial matching
    instance_matcher = InstanceMatcher(
        method=InstanceMatchMethod.SPATIAL, threshold=5.0
    )
    result = labels1.merge(labels2, instance=instance_matcher, frame="auto")

    assert result.successful
    assert result.frames_merged == 1
    assert len(result.conflicts) > 0  # Should have recorded conflicts


def test_labels_merge_frame_strategies():
    """Test different frame merge strategies."""
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Test keep_original strategy
    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video)

    frame1 = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel)],
    )
    labels1.append(frame1)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video)

    frame2 = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skel)],
    )
    labels2.append(frame2)

    # Merge with keep_original
    original_instances = len(labels1.labeled_frames[0].instances)
    result = labels1.merge(labels2, frame="keep_original")

    assert result.successful
    assert (
        len(labels1.labeled_frames[0].instances) == original_instances
    )  # Should keep original


def test_labels_merge_preserves_is_negative_colliding():
    """Merging a negative frame onto a colliding empty frame keeps the flag."""
    skel = Skeleton(["A"])
    video = Video(filename="test.mp4")

    base = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0)],
    )
    incoming = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, is_negative=True)],
    )

    base.merge(incoming)

    assert base.labeled_frames[0].is_negative is True


def test_labels_merge_preserves_is_negative_noncolliding():
    """Merging a negative frame with no collision keeps the flag on the new frame."""
    skel = Skeleton(["A"])
    video = Video(filename="test.mp4")

    base = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[LabeledFrame(video=video, frame_idx=5)],
    )
    incoming = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, is_negative=True)],
    )

    base.merge(incoming)

    new_frame = [lf for lf in base.labeled_frames if lf.frame_idx == 0][0]
    assert new_frame.is_negative is True


def test_labels_merge_is_negative_user_pose_conflict():
    """A user pose cancels the negative flag and records a merge conflict."""
    skel = Skeleton(["A"])
    video = Video(filename="test.mp4")

    base = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, is_negative=True)],
    )
    incoming = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[Instance([[10, 10]], skeleton=skel)],
            )
        ],
    )

    result = base.merge(incoming)

    assert base.labeled_frames[0].is_negative is False
    negative_conflicts = [
        c for c in result.conflicts if c.conflict_type == "negative_flag_conflict"
    ]
    assert len(negative_conflicts) == 1
    assert negative_conflicts[0].resolution == "dropped_for_user_pose"


def test_labels_merge_is_negative_roundtrip(tmp_path):
    """Negative flag survives a save_slp -> load_slp -> merge round-trip."""
    skel = Skeleton(["A"])
    video = Video(filename="test.mp4")

    base = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0)],
    )
    incoming = Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, is_negative=True)],
    )

    base_path = tmp_path / "base.slp"
    incoming_path = tmp_path / "incoming.slp"
    save_slp(base, base_path)
    save_slp(incoming, incoming_path)

    base_loaded = load_slp(base_path)
    incoming_loaded = load_slp(incoming_path)
    base_loaded.merge(incoming_loaded)

    assert base_loaded.labeled_frames[0].is_negative is True


def test_labels_merge_suggestions():
    """Test merging of suggestions."""
    video = Video(filename="test.mp4")

    labels1 = Labels()
    labels1.videos.append(video)

    labels2 = Labels()
    labels2.videos.append(video)

    # Add a suggestion to labels2
    suggestion = SuggestionFrame(video=video, frame_idx=10)
    labels2.suggestions.append(suggestion)

    # Merge
    result = labels1.merge(labels2)

    assert result.successful
    assert len(labels1.suggestions) == 1
    assert labels1.suggestions[0].frame_idx == 10


def test_labels_merge_provenance_tracking():
    """Test that merge history is tracked in provenance."""
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video)

    # Add frames to labels2
    for i in range(3):
        frame = LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[
                Instance.from_numpy(np.array([[i, i], [i + 10, i + 10]]), skeleton=skel)
            ],
        )
        labels2.append(frame)

    # Merge and check provenance
    result = labels1.merge(labels2)

    assert result.successful
    assert "merge_history" in labels1.provenance
    assert len(labels1.provenance["merge_history"]) == 1

    merge_record = labels1.provenance["merge_history"][0]
    assert merge_record["source_labels"]["n_frames"] == 3
    assert merge_record["source_labels"]["n_videos"] == 1
    assert merge_record["source_labels"]["n_skeletons"] == 1
    assert merge_record["strategy"] == "auto"
    # Verify new provenance fields
    assert "source_filename" in merge_record
    assert "target_filename" in merge_record
    assert "sleap_io_version" in merge_record
    # In-memory labels have None for filenames
    assert merge_record["source_filename"] is None
    assert merge_record["target_filename"] is None
    # Version should match current version
    assert merge_record["sleap_io_version"] == sleap_io.__version__


def test_labels_merge_provenance_with_filenames():
    """Test that merge history captures filenames from provenance."""
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Create labels with filenames in provenance
    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video)
    labels1.provenance["filename"] = "/path/to/base_labels.slp"

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video)
    labels2.provenance["filename"] = "/path/to/source_predictions.slp"

    # Add a frame to labels2
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel)],
    )
    labels2.append(frame)

    # Merge
    result = labels1.merge(labels2)

    assert result.successful
    merge_record = labels1.provenance["merge_history"][0]

    # Verify filenames are captured
    assert merge_record["source_filename"] == "/path/to/source_predictions.slp"
    assert merge_record["target_filename"] == "/path/to/base_labels.slp"
    assert merge_record["sleap_io_version"] == sleap_io.__version__


def test_labels_merge_provenance_mixed_filenames():
    """Test merge provenance when only one Labels has a filename."""
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4")

    # Target has filename, source does not
    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video)
    labels1.provenance["filename"] = "/path/to/base_labels.slp"

    labels2 = Labels()  # In-memory, no filename
    labels2.skeletons.append(skel)
    labels2.videos.append(video)

    # Add a frame to labels2
    frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel)],
    )
    labels2.append(frame)

    # Merge
    result = labels1.merge(labels2)

    assert result.successful
    merge_record = labels1.provenance["merge_history"][0]

    # Source is in-memory (None), target has filename
    assert merge_record["source_filename"] is None
    assert merge_record["target_filename"] == "/path/to/base_labels.slp"


def test_labels_merge_custom_matchers():
    """Test merge with custom matchers."""
    # Create labels with specific matching requirements
    skel = Skeleton(nodes=["head", "thorax", "abdomen"])
    video1 = Video(filename="/path/to/video.mp4")
    video2 = Video(filename="/different/path/video.mp4")  # Same basename
    track1 = Track(name="ant1")
    track2 = Track(name="ant2")

    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video1)
    labels1.tracks.append(track1)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video2)
    labels2.tracks.append(track2)

    # Configure custom matchers
    skeleton_matcher = SkeletonMatcher(method=SkeletonMatchMethod.SUBSET)
    video_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)  # Match by basename
    track_matcher = TrackMatcher(
        method=TrackMatchMethod.IDENTITY
    )  # Don't match by name

    # Merge with custom matchers
    result = labels1.merge(
        labels2,
        skeleton=skeleton_matcher,
        video=video_matcher,
        track=track_matcher,
    )

    assert result.successful
    assert len(labels1.videos) == 1  # Videos matched by basename
    assert len(labels1.tracks) == 2  # Tracks not matched (identity matching)


def test_labels_merge_string_api():
    """Test merge with string arguments for matchers (no imports needed)."""
    # Create labels
    skel = Skeleton(nodes=["head", "thorax", "abdomen"])
    video1 = Video(filename="/path/to/video.mp4")
    video2 = Video(filename="/different/path/video.mp4")  # Same basename
    track1 = Track(name="ant1")
    track2 = Track(name="ant1")  # Same name

    labels1 = Labels()
    labels1.skeletons.append(skel)
    labels1.videos.append(video1)
    labels1.tracks.append(track1)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    labels2.videos.append(video2)
    labels2.tracks.append(track2)

    # Add frame to labels2
    inst = Instance.from_numpy(np.array([[10, 10], [20, 20], [30, 30]]), skeleton=skel)
    frame = LabeledFrame(video=video2, frame_idx=0, instances=[inst])
    labels2.append(frame)

    # Merge using STRING arguments instead of Matcher objects
    result = labels1.merge(
        labels2,
        skeleton="structure",  # String instead of SkeletonMatcher
        video="basename",  # String instead of VideoMatcher
        track="name",  # String instead of TrackMatcher
        frame="auto",  # String (already was string)
        instance="spatial",  # String instead of InstanceMatcher
    )

    assert result.successful
    assert len(labels1.videos) == 1  # Videos matched by basename
    assert len(labels1.tracks) == 1  # Tracks matched by name
    assert len(labels1.labeled_frames) == 1


def test_labels_merge_empty():
    """Test merging empty Labels objects."""
    labels1 = Labels()
    labels2 = Labels()

    result = labels1.merge(labels2)

    assert result.successful
    assert result.frames_merged == 0
    assert result.instances_added == 0


def test_labels_merge_error_handling(monkeypatch):
    """Test error handling during merge."""
    labels1 = Labels()
    labels2 = Labels()

    # Add matching skeletons to both labels so matching will be attempted
    skel1 = Skeleton(nodes=["head", "tail"])
    skel2 = Skeleton(nodes=["head", "tail"])
    labels1.skeletons.append(skel1)
    labels2.skeletons.append(skel2)

    # Mock the skeleton matcher to raise an exception during match
    def mock_match(self, skeleton1, skeleton2):
        raise Exception("Test error")

    # Use monkeypatch to mock the match method
    monkeypatch.setattr(SkeletonMatcher, "match", mock_match)

    # Merge should handle the error based on error_mode
    with pytest.raises(Exception, match="Test error"):
        labels1.merge(labels2, error_mode="strict")


def test_labels_merge_map_instance():
    """Test the _map_instance helper method."""
    # Create Labels with skeleton and track
    skel1 = Skeleton(nodes=["head", "tail"])
    skel2 = Skeleton(nodes=["head", "tail"])
    track1 = Track(name="track1")
    track2 = Track(name="track2")

    labels = Labels()
    labels.skeletons.append(skel1)
    labels.tracks.append(track1)

    # Create maps
    skeleton_map = {skel2: skel1}
    track_map = {track2: track1}

    # Create instance with original skeleton and track
    inst = Instance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skel2, track=track2
    )

    # Map the instance
    mapped_inst = labels._map_instance(inst, skeleton_map, track_map)

    assert mapped_inst.skeleton == skel1
    assert mapped_inst.track == track1
    assert np.array_equal(mapped_inst.numpy(), inst.numpy())


def test_labels_merge_predicted_instance_mapping():
    """Test _map_instance with PredictedInstance."""
    skel1 = Skeleton(nodes=["head", "tail"])
    skel2 = Skeleton(nodes=["head", "tail"])
    track1 = Track(name="track1")
    track2 = Track(name="track2")

    labels = Labels()
    labels.skeletons.append(skel1)
    labels.tracks.append(track1)

    # Create maps
    skeleton_map = {skel2: skel1}
    track_map = {track2: track1}

    # Create PredictedInstance with original skeleton and track
    inst = PredictedInstance.from_numpy(
        np.array([[10, 10], [20, 20]]), skeleton=skel2, track=track2, score=0.95
    )

    # Map the instance
    mapped_inst = labels._map_instance(inst, skeleton_map, track_map)

    assert isinstance(mapped_inst, PredictedInstance)
    assert mapped_inst.skeleton == skel1
    assert mapped_inst.track == track1
    assert mapped_inst.score == 0.95
    assert np.array_equal(mapped_inst.numpy(), inst.numpy())


def test_labels_merge_video_basename_with_fallback_dirs(tmp_path):
    """Test merge with VideoMatchMethod.BASENAME using fallback directories.

    This tests the case where videos have the same basename but are in different
    locations, and we use fallback directories to resolve them.
    """
    import shutil

    # Get path to test video
    test_video = Path("tests/data/videos/centered_pair_low_quality.mp4")

    # Create directory structure
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    shared_videos = tmp_path / "shared_videos"

    project_a.mkdir()
    project_b.mkdir()
    shared_videos.mkdir()

    (project_a / "videos").mkdir()
    (project_b / "videos").mkdir()

    # Copy actual video files with same basename to different locations
    video_a_path = project_a / "videos" / "recording.mp4"
    video_shared_path = shared_videos / "recording.mp4"

    shutil.copy(test_video, video_a_path)
    shutil.copy(test_video, video_shared_path)

    # Create Labels with videos referencing different paths but same basename
    skel = Skeleton(nodes=["head", "tail"])

    # Labels A references the video in project_a
    labels_a = Labels()
    labels_a.skeletons.append(skel)
    video_a = Video(filename=str(video_a_path))
    labels_a.videos.append(video_a)

    frame_a = LabeledFrame(
        video=video_a,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel)],
    )
    labels_a.append(frame_a)

    # Labels B references a video in project_b
    labels_b = Labels()
    labels_b.skeletons.append(skel)
    video_b_path = project_b / "videos" / "recording.mp4"  # Same basename
    # Don't copy the file initially to test the fallback mechanism
    # But we need it for saving/loading, so create it temporarily
    shutil.copy(test_video, video_b_path)
    video_b = Video(filename=str(video_b_path))
    labels_b.videos.append(video_b)

    frame_b = LabeledFrame(
        video=video_b,
        frame_idx=1,
        instances=[Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skel)],
    )
    labels_b.append(frame_b)

    # Save and load the labels to test with real file I/O
    labels_a.save(project_a / "labels_a.slp")
    labels_b.save(project_b / "labels_b.slp")

    loaded_a = load_slp(project_a / "labels_a.slp")
    loaded_b = load_slp(project_b / "labels_b.slp")

    # Test 1: Without fallback, videos don't match (different paths)
    video_matcher_no_fallback = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
    result_no_match = loaded_a.merge(loaded_b, video=video_matcher_no_fallback)
    assert result_no_match.successful
    assert len(loaded_a.videos) == 2  # Both videos added (no match)

    # Reload for next test
    loaded_a = load_slp(project_a / "labels_a.slp")
    loaded_b = load_slp(project_b / "labels_b.slp")

    # Test 2: With RESOLVE method (simplified - matches by basename)
    # The videos have same basename so they should match
    video_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

    result = loaded_a.merge(loaded_b, video=video_matcher)
    assert result.successful
    # Videos match because recording.mp4 exists in fallback directory
    assert len(loaded_a.videos) == 1  # Videos matched via fallback
    assert len(loaded_a.labeled_frames) == 2  # Both frames merged

    # Test 3: Multiple videos with same basename in different locations
    labels_c = Labels()
    labels_c.skeletons.append(skel)

    # Add another video with different basename
    other_video_path = project_b / "videos" / "other.mp4"
    shutil.copy(test_video, other_video_path)
    video_c = Video(filename=str(other_video_path))
    labels_c.videos.append(video_c)

    # Also add a video that will match via fallback
    video_d = Video(filename=str(project_b / "videos" / "recording.mp4"))
    labels_c.videos.append(video_d)

    frame_c = LabeledFrame(
        video=video_c,
        frame_idx=2,
        instances=[Instance.from_numpy(np.array([[50, 50], [60, 60]]), skeleton=skel)],
    )
    labels_c.append(frame_c)

    # Reset labels for clean test
    labels_d = Labels()
    labels_d.skeletons.append(skel)
    labels_d.videos.append(Video(filename=str(project_a / "videos" / "recording.mp4")))

    result3 = labels_d.merge(labels_c, video=video_matcher)
    assert result3.successful
    assert len(labels_d.videos) == 2  # recording.mp4 matched, other.mp4 added


def test_labels_merge_video_basename_matching(tmp_path):
    """Test merge with VideoMatchMethod.BASENAME.

    This tests basename matching with videos in different directories.
    """
    import shutil

    # Get path to test video
    test_video = Path("tests/data/videos/centered_pair_low_quality.mp4")

    # Create directory structure
    base = tmp_path / "base"
    base.mkdir()

    subdir1 = base / "data" / "videos"
    subdir2 = base / "backup" / "videos"
    subdir1.mkdir(parents=True)
    subdir2.mkdir(parents=True)

    # Copy video files with same basename to different locations
    video1_path = subdir1 / "experiment.mp4"
    video2_path = subdir2 / "experiment.mp4"
    base_video_path = base / "experiment.mp4"

    shutil.copy(test_video, video1_path)
    shutil.copy(test_video, video2_path)
    shutil.copy(test_video, base_video_path)

    skel = Skeleton(nodes=["head", "tail"])

    # Create Labels referencing videos in different subdirs
    labels1 = Labels()
    labels1.skeletons.append(skel)
    video1 = Video(filename=str(video1_path))
    labels1.videos.append(video1)

    frame1 = LabeledFrame(
        video=video1,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel)],
    )
    labels1.append(frame1)

    labels2 = Labels()
    labels2.skeletons.append(skel)
    video2 = Video(filename=str(video2_path))
    labels2.videos.append(video2)

    frame2 = LabeledFrame(
        video=video2,
        frame_idx=1,
        instances=[Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skel)],
    )
    labels2.append(frame2)

    # Test 1: Videos with same basename but different paths
    # BASENAME method does filename-based matching ignoring directory paths
    video_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

    result = labels1.merge(labels2, video=video_matcher)
    assert result.successful
    # Videos match because they have the same basename (experiment.mp4)
    assert len(labels1.videos) == 1  # Videos matched via base_path lookup

    # Test 2: Test relative path matching (same relative structure)
    # Create labels with absolute paths that have same relative structure
    if os.name != "nt":  # Skip on Windows due to path anchor differences
        # Create two videos with same relative path from root
        abs_path1 = Path("/data/videos/test.mp4")
        abs_path2 = Path("/backup/videos/test.mp4")

        labels3 = Labels()
        labels3.skeletons.append(skel)
        video3 = Video(filename=str(abs_path1))
        labels3.videos.append(video3)

        labels4 = Labels()
        labels4.skeletons.append(skel)
        video4 = Video(filename=str(abs_path2))
        labels4.videos.append(video4)

        # These have different absolute paths but might match on relative structure
        video_matcher2 = VideoMatcher(method=VideoMatchMethod.BASENAME)

        # The merge will attempt relative_to() which will raise ValueError
        # This tests the exception handling
        result2 = labels3.merge(labels4, video=video_matcher2)
        assert result2.successful

    # Test 3: Test when base_path contains the video file directly
    labels5 = Labels()
    labels5.skeletons.append(skel)
    video5 = Video(filename="experiment.mp4")  # Just basename
    labels5.videos.append(video5)

    frame5 = LabeledFrame(
        video=video5,
        frame_idx=2,
        instances=[Instance.from_numpy(np.array([[50, 50], [60, 60]]), skeleton=skel)],
    )
    labels5.append(frame5)

    # This should find experiment.mp4 in base path
    video_matcher3 = VideoMatcher(method=VideoMatchMethod.BASENAME)

    # Merge with labels1 which has full path
    labels6 = Labels()
    labels6.skeletons.append(skel)
    labels6.videos.append(video1)  # Full path to subdir1/experiment.mp4

    result3 = labels6.merge(labels5, video=video_matcher3)
    assert result3.successful
    # Videos should match because experiment.mp4 exists in base_path
    assert len(labels6.videos) == 1  # Videos matched via base_path


def test_labels_merge_video_basename_complex_scenario(tmp_path):
    """Test complex merge scenario with multiple resolution strategies.

    This comprehensive test covers:
    - Multiple videos with same/different basenames
    - Fallback directories and base_path working together
    - Priority order of resolution strategies
    - Edge cases like missing files and path errors
    """
    import shutil

    # Get path to test video
    test_video = Path("tests/data/videos/centered_pair_low_quality.mp4")

    # Create complex directory structure
    workspace = tmp_path / "workspace"
    archive = tmp_path / "archive"
    shared = tmp_path / "shared"

    for d in [workspace, archive, shared]:
        d.mkdir()

    project1 = workspace / "project1"
    project2 = workspace / "project2"
    project1.mkdir()
    project2.mkdir()

    # Copy video files to various locations
    shutil.copy(test_video, project1 / "video_a.mp4")
    shutil.copy(test_video, project2 / "video_b.mp4")
    shutil.copy(test_video, shared / "video_a.mp4")  # Duplicate basename in shared
    shutil.copy(test_video, shared / "video_c.mp4")  # Only in shared
    shutil.copy(test_video, archive / "video_b.mp4")  # Duplicate basename in archive

    skel = Skeleton(nodes=["head", "tail"])

    # Create Labels1 with multiple videos
    labels1 = Labels()
    labels1.skeletons.append(skel)

    # Video in project1
    video1a = Video(filename=str(project1 / "video_a.mp4"))
    labels1.videos.append(video1a)

    # Video that doesn't exist locally but might be in shared
    # Create the file so save/load works
    shutil.copy(test_video, project1 / "video_c.mp4")
    video1c = Video(filename=str(project1 / "video_c.mp4"))
    labels1.videos.append(video1c)

    frame1 = LabeledFrame(
        video=video1a,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skel)],
    )
    labels1.append(frame1)

    # Create Labels2 with overlapping videos
    labels2 = Labels()
    labels2.skeletons.append(skel)

    # Video with same basename as video1a but different location
    video2a = Video(filename=str(shared / "video_a.mp4"))
    labels2.videos.append(video2a)

    # Video in project2
    video2b = Video(filename=str(project2 / "video_b.mp4"))
    labels2.videos.append(video2b)

    # Video that matches video1c but from different path
    # Create the file so save/load works
    shutil.copy(test_video, archive / "video_c.mp4")
    video2c = Video(filename=str(archive / "video_c.mp4"))
    labels2.videos.append(video2c)

    frame2 = LabeledFrame(
        video=video2a,
        frame_idx=1,
        instances=[Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skel)],
    )
    labels2.append(frame2)

    frame3 = LabeledFrame(
        video=video2b,
        frame_idx=2,
        instances=[Instance.from_numpy(np.array([[50, 50], [60, 60]]), skeleton=skel)],
    )
    labels2.append(frame3)

    # Save and reload to simulate real usage
    labels1.save(project1 / "labels1.slp")
    labels2.save(project2 / "labels2.slp")

    loaded1 = load_slp(project1 / "labels1.slp")
    loaded2 = load_slp(project2 / "labels2.slp")

    # Create a video matcher using BASENAME (filename-based matching)
    video_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

    result = loaded1.merge(loaded2, video=video_matcher)

    assert result.successful
    assert result.frames_merged == 2  # frame2 and frame3

    # Check video handling:
    # - video1a and video2a: Same basename but different paths, both exist
    # - video1c: Doesn't exist at original path but exists in shared (fallback)
    # - video2b: Exists at its path
    # - video2c: Doesn't exist at original path, should check fallbacks

    # After merge, we should have all unique videos
    # Check all unique videos are present
    [v.filename for v in loaded1.videos]

    # Test with edge cases

    # Test 4: Test with non-existent fallback directories (should not crash)
    video_matcher_bad = VideoMatcher(method=VideoMatchMethod.BASENAME)

    labels3 = Labels()
    labels3.skeletons.append(skel)
    video3 = Video(filename="orphan.mp4")
    labels3.videos.append(video3)

    result2 = loaded1.merge(labels3, video=video_matcher_bad)
    assert result2.successful  # Should not crash even with bad paths

    # Test 5: Test priority order - direct match should take precedence
    labels5 = Labels()
    labels5.skeletons.append(skel)
    labels5.videos.append(video1a)  # Exact same video object as in loaded1

    labels6 = Labels()
    labels6.skeletons.append(skel)
    # Same filename but different object
    video6 = Video(filename=str(project1 / "video_a.mp4"))
    labels6.videos.append(video6)

    # Should match based on path even before trying resolve strategies
    result4 = labels5.merge(labels6, video=video_matcher)
    assert result4.successful
    assert len(labels5.videos) == 1  # Should match and not duplicate


def test_labels_merge_video_basename_edge_cases(tmp_path):
    """Test VideoMatchMethod.BASENAME edge cases.

    Tests various edge cases for the BASENAME matching method.
    """
    import shutil

    # Get path to test video
    test_video = Path("tests/data/videos/centered_pair_low_quality.mp4")

    # Create directory structure
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    fallback = tmp_path / "fallback"
    base = tmp_path / "base"

    dir1.mkdir()
    dir2.mkdir()
    fallback.mkdir()
    base.mkdir()

    # Test 1: Fallback directories with matching basename
    # The key is that the videos must:
    # 1. Have the same basename
    # 2. NOT match via matches_path (different directories)
    # 3. Have the file exist in fallback directory

    video1_path = dir1 / "recording.mp4"
    video2_path = dir2 / "recording.mp4"  # Same basename, different directory
    fallback_video = fallback / "recording.mp4"

    shutil.copy(test_video, video1_path)
    # Copy to video2_path as well so Video() can open it
    shutil.copy(test_video, video2_path)
    shutil.copy(test_video, fallback_video)  # Also in fallback

    # Create videos - the key is they must not match via matches_path
    # but should match via fallback when basename matches
    video1 = Video(filename=str(video1_path))
    video2 = Video(filename=str(video2_path))

    # First verify they don't match without fallback
    matcher_no_fallback = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
    assert not matcher_no_fallback.match(video1, video2)

    # Now test with simplified RESOLVE (basename matching)
    video_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)

    # They should match because they have the same basename (recording.mp4)
    assert video_matcher.match(video1, video2)

    # Test 2: Base path with basename lookup
    # Create videos with same basename that exists in base_path
    base_video = base / "shared.mp4"
    shutil.copy(test_video, base_video)

    # Create videos that reference files with same basename but different paths
    # These files don't need to exist
    video3_path = dir1 / "shared.mp4"
    video4_path = dir2 / "shared.mp4"

    shutil.copy(test_video, video3_path)
    shutil.copy(test_video, video4_path)

    video3 = Video(filename=str(video3_path))
    video4 = Video(filename=str(video4_path))

    # Without base_path, they shouldn't match (different paths)
    matcher_no_base = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
    assert not matcher_no_base.match(video3, video4)

    # With simplified RESOLVE (basename matching), they should match
    video_matcher2 = VideoMatcher(method=VideoMatchMethod.BASENAME)

    # Should match because they have the same basename (shared.mp4)
    assert video_matcher2.match(video3, video4)

    # Test 3: Non-matching basenames don't match
    # RESOLVE should NOT match videos with different basenames
    # (simplified RESOLVE only does basename matching)

    # Create videos with different basenames
    diff_video1_path = dir1 / "unique1.mp4"
    diff_video2_path = dir2 / "unique2.mp4"  # Different basename

    shutil.copy(test_video, diff_video1_path)
    shutil.copy(test_video, diff_video2_path)

    video5 = Video(filename=str(diff_video1_path))
    video6 = Video(filename=str(diff_video2_path))

    video_matcher4 = VideoMatcher(method=VideoMatchMethod.BASENAME)

    # Should not match because basenames differ (unique1.mp4 vs unique2.mp4)
    result = video_matcher4.match(video5, video6)
    assert not result  # Should not match

    # Additional coverage tests
    # Test same object matching (line 228)
    video_same = Video(filename=str(diff_video1_path))
    matcher_same = VideoMatcher(method=VideoMatchMethod.BASENAME)
    assert matcher_same.match(video_same, video_same)

    # Test additional edge case: ensure consistent behavior
    # All simplified RESOLVE matchers should behave the same (basename matching)
    matcher_simple = VideoMatcher(method=VideoMatchMethod.BASENAME)
    assert matcher_simple.match(video1, video2)  # Same basenames match
    assert matcher_simple.match(video3, video4)  # Same basenames match

    # Test exception handling in relative_to (lines 273-274)
    # Create videos with relative paths that will fail relative_to
    video_rel1 = Video(filename="relative/path1.mp4")
    video_rel2 = Video(filename="relative/path2.mp4")
    matcher_exc = VideoMatcher(method=VideoMatchMethod.BASENAME)
    # This should trigger the exception handler
    assert not matcher_exc.match(video_rel1, video_rel2)

    # Test non-matching basenames with path check (line 278)
    # These have different basenames, so should go through the else branch
    assert not matcher_same.match(video5, video6)

    # Test 4: Test with both fallback and base_path
    # Ensure fallback is checked before base_path
    combo_video = fallback / "combo.mp4"
    base_combo = base / "combo.mp4"

    shutil.copy(test_video, combo_video)  # In fallback
    shutil.copy(test_video, base_combo)  # Also in base

    video5_path = dir1 / "combo.mp4"
    video6_path = dir2 / "combo.mp4"

    shutil.copy(test_video, video5_path)
    shutil.copy(test_video, video6_path)

    video5 = Video(filename=str(video5_path))
    video6 = Video(filename=str(video6_path))

    video_matcher4 = VideoMatcher(method=VideoMatchMethod.BASENAME)

    # Should match via fallback (checked before base_path)
    assert video_matcher4.match(video5, video6)


def test_labels_merge_suggestion_frame_duplication():
    """Test Labels.merge handling of duplicate suggestion frames."""
    from sleap_io import Labels, SuggestionFrame, Video

    # Create videos
    video1 = Video(filename="test1.mp4")
    video2 = Video(filename="test2.mp4")

    # Create labels with suggestion frames
    labels1 = Labels()
    labels1.videos.append(video1)
    labels1.suggestions.append(SuggestionFrame(video=video1, frame_idx=10))
    labels1.suggestions.append(SuggestionFrame(video=video1, frame_idx=20))

    labels2 = Labels()
    labels2.videos.append(video1)  # Same video
    # Add duplicate suggestion frame
    labels2.suggestions.append(SuggestionFrame(video=video1, frame_idx=10))
    # Add new suggestion frame
    labels2.suggestions.append(SuggestionFrame(video=video1, frame_idx=30))

    # Merge labels
    labels1.merge(labels2)

    # Should have only unique suggestions (lines 1855-1860 check for duplicates)
    assert len(labels1.suggestions) == 3  # 10, 20, 30 (no duplicate of 10)

    # Check that we have the expected frame indices
    suggestion_indices = {(s.video.filename, s.frame_idx) for s in labels1.suggestions}
    assert ("test1.mp4", 10) in suggestion_indices
    assert ("test1.mp4", 20) in suggestion_indices
    assert ("test1.mp4", 30) in suggestion_indices

    # Test with different videos
    labels3 = Labels()
    labels3.videos.append(video2)
    labels3.suggestions.append(SuggestionFrame(video=video2, frame_idx=10))

    # Note: The merge won't add the suggestion if the video wasn't mapped
    # This is because video2 is different from video1
    # Merge with different video
    labels1.merge(labels3)

    # If video2 was added to labels1.videos, we should see the suggestion
    if video2 in labels1.videos:
        assert any(s.video == video2 and s.frame_idx == 10 for s in labels1.suggestions)


def test_labels_merge_skeleton_remapping_for_existing_frames(tmp_path):
    """Test skeleton references are correctly remapped when merging overlapping frames.

    This test reproduces the bug where instances from merged frames retain references
    to the original skeleton objects instead of being remapped to the matching skeleton
    in the target Labels object.
    """
    import numpy as np

    from sleap_io import Instance, LabeledFrame, Labels, Skeleton, Video

    # Create skeleton and video (same structure, different objects)
    skeleton1 = Skeleton(["head", "tail"])
    skeleton2 = Skeleton(["head", "tail"])  # Same structure, different object
    video = Video(filename="test.mp4", open_backend=False)

    # Create first Labels with one frame
    labels1 = Labels()
    labels1.skeletons = [skeleton1]
    labels1.videos = [video]
    frame1 = LabeledFrame(video=video, frame_idx=0)
    inst1 = Instance.from_numpy(np.array([[10, 10], [20, 20]]), skeleton=skeleton1)
    frame1.instances = [inst1]
    labels1.append(frame1)

    # Create second Labels with overlapping frame (same frame_idx)
    labels2 = Labels()
    labels2.skeletons = [skeleton2]
    labels2.videos = [video]
    frame2 = LabeledFrame(video=video, frame_idx=0)
    inst2 = Instance.from_numpy(np.array([[30, 30], [40, 40]]), skeleton=skeleton2)
    frame2.instances = [inst2]
    labels2.append(frame2)

    # Verify skeletons are different objects but same structure
    assert skeleton1 is not skeleton2
    assert skeleton1.matches(skeleton2)

    # Merge labels2 into labels1
    result = labels1.merge(labels2, frame="keep_both")

    assert result.successful
    assert len(labels1.skeletons) == 1  # Should reuse existing skeleton
    assert labels1.skeletons[0] is skeleton1  # Should be the original skeleton

    # Check that all instances reference the correct skeleton
    for frame_idx, frame in enumerate(labels1.labeled_frames):
        for inst_idx, instance in enumerate(frame.instances):
            assert instance.skeleton is skeleton1, (
                f"Frame {frame_idx}, Instance {inst_idx}: "
                f"skeleton {id(instance.skeleton)} != expected {id(skeleton1)}"
            )

    # The main test is that the skeleton references are correct.
    # Previously, saving would fail with ValueError because instances had
    # skeleton references that weren't in labels1.skeletons.
    # Now test that save would work (skip actual save/load due to video backend)

    # Simulate what save does - check all instance skeletons are in labels.skeletons
    for frame in labels1.labeled_frames:
        for instance in frame.instances:
            assert instance.skeleton in labels1.skeletons, (
                "Instance skeleton not in labels.skeletons list"
            )


def test_labels_match_basic():
    """Test basic Labels.match functionality."""
    # Create ground truth labels
    skeleton_gt = Skeleton(["head", "tail"])
    video_gt = Video(filename="/data/experiment/video.mp4", open_backend=False)
    gt_labels = Labels(videos=[video_gt], skeletons=[skeleton_gt])

    # Create prediction labels with different path but same basename
    skeleton_pred = Skeleton(["head", "tail"])
    video_pred = Video(filename="/output/model/video.mp4", open_backend=False)
    pred_labels = Labels(videos=[video_pred], skeletons=[skeleton_pred])

    # Match predictions to ground truth
    result = gt_labels.match(pred_labels)

    # Check video matching (should match by basename)
    assert len(result.video_map) == 1
    assert video_pred in result.video_map
    assert result.video_map[video_pred] is video_gt

    # Check skeleton matching (should match by structure)
    assert len(result.skeleton_map) == 1
    assert skeleton_pred in result.skeleton_map
    assert result.skeleton_map[skeleton_pred] is skeleton_gt

    # Check convenience properties
    assert result.all_videos_matched
    assert result.all_skeletons_matched
    assert result.n_videos_matched == 1
    assert result.n_skeletons_matched == 1
    assert len(result.unmatched_videos) == 0
    assert len(result.unmatched_skeletons) == 0


def test_labels_match_unmatched_videos():
    """Test Labels.match with videos that don't match."""
    skeleton = Skeleton(["head", "tail"])
    video_gt = Video(filename="/data/video_a.mp4", open_backend=False)
    video_pred = Video(filename="/data/video_b.mp4", open_backend=False)

    gt_labels = Labels(videos=[video_gt], skeletons=[skeleton])
    pred_labels = Labels(videos=[video_pred], skeletons=[skeleton])

    result = gt_labels.match(pred_labels)

    # Video should not match (different basenames)
    assert not result.all_videos_matched
    assert len(result.unmatched_videos) == 1
    assert result.video_map[video_pred] is None

    # Skeleton should still match
    assert result.all_skeletons_matched


def test_labels_match_multiple_videos():
    """Test Labels.match with multiple videos."""
    skeleton = Skeleton(["head", "tail"])

    # Ground truth with 2 videos
    video_gt_1 = Video(filename="/data/video1.mp4", open_backend=False)
    video_gt_2 = Video(filename="/data/video2.mp4", open_backend=False)
    gt_labels = Labels(videos=[video_gt_1, video_gt_2], skeletons=[skeleton])

    # Predictions with 3 videos (2 match, 1 doesn't)
    video_pred_1 = Video(filename="/output/video1.mp4", open_backend=False)
    video_pred_2 = Video(filename="/output/video2.mp4", open_backend=False)
    video_pred_3 = Video(filename="/output/video3.mp4", open_backend=False)
    pred_labels = Labels(
        videos=[video_pred_1, video_pred_2, video_pred_3], skeletons=[skeleton]
    )

    result = gt_labels.match(pred_labels)

    # Check matches
    assert result.n_videos_matched == 2
    assert len(result.unmatched_videos) == 1
    assert video_pred_3 in result.unmatched_videos
    assert result.video_map[video_pred_1] is video_gt_1
    assert result.video_map[video_pred_2] is video_gt_2
    assert result.video_map[video_pred_3] is None


def test_labels_match_track_matching():
    """Test Labels.match with track matching."""
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="/data/video.mp4", open_backend=False)

    track_gt = Track("animal_1")
    track_pred = Track("animal_1")  # Same name, different object

    gt_labels = Labels(videos=[video], skeletons=[skeleton], tracks=[track_gt])
    pred_labels = Labels(videos=[video], skeletons=[skeleton], tracks=[track_pred])

    # Opt in to name-based matching: these are two distinct Track objects with
    # the same name, which only coalesce under track="name" (the identity
    # default would keep them as separate tracks).
    result = gt_labels.match(pred_labels, track="name")

    # Track should match by name
    assert result.all_tracks_matched
    assert result.track_map[track_pred] is track_gt


def test_labels_match_string_method():
    """Test Labels.match with string method arguments."""
    skeleton = Skeleton(["head", "tail"])
    video_gt = Video(filename="/data/video.mp4", open_backend=False)
    video_pred = Video(filename="/output/video.mp4", open_backend=False)

    gt_labels = Labels(videos=[video_gt], skeletons=[skeleton])
    pred_labels = Labels(videos=[video_pred], skeletons=[skeleton])

    # Test with string methods
    result = gt_labels.match(pred_labels, video="basename", skeleton="structure")

    assert result.all_videos_matched
    assert result.all_skeletons_matched


def test_labels_match_custom_matchers():
    """Test Labels.match with custom matcher objects."""
    # For SUBSET matching, skeleton1 must be a subset of skeleton2
    # Since match() is called with (self_skel, other_skel) = (gt, pred),
    # gt nodes must be subset of pred nodes for match to succeed
    skeleton_gt = Skeleton(["head", "tail"])
    skeleton_pred = Skeleton(["head", "body", "tail"])  # GT nodes are subset of pred
    video = Video(filename="/data/video.mp4", open_backend=False)

    gt_labels = Labels(videos=[video], skeletons=[skeleton_gt])
    pred_labels = Labels(videos=[video], skeletons=[skeleton_pred])

    # Structure matching should fail (different nodes)
    result_struct = gt_labels.match(pred_labels, skeleton="structure")
    assert not result_struct.all_skeletons_matched

    # Subset matching should succeed (gt nodes are subset of pred nodes)
    result_subset = gt_labels.match(pred_labels, skeleton="subset")
    assert result_subset.all_skeletons_matched


def test_labels_match_empty():
    """Test Labels.match with empty Labels."""
    gt_labels = Labels()
    pred_labels = Labels()

    result = gt_labels.match(pred_labels)

    assert len(result.video_map) == 0
    assert len(result.skeleton_map) == 0
    assert len(result.track_map) == 0
    assert result.all_videos_matched  # Vacuously true
    assert result.all_skeletons_matched
    assert result.all_tracks_matched


def test_labels_match_summary():
    """Test MatchResult.summary() output."""
    skeleton = Skeleton(["head", "tail"])
    video_gt = Video(filename="/data/video.mp4", open_backend=False)
    video_pred = Video(filename="/output/different.mp4", open_backend=False)

    gt_labels = Labels(videos=[video_gt], skeletons=[skeleton])
    pred_labels = Labels(videos=[video_pred], skeletons=[skeleton])

    result = gt_labels.match(pred_labels)

    summary = result.summary()
    assert "Videos: 0/1 matched" in summary
    assert "Skeletons: 1/1 matched" in summary
    assert "Unmatched videos:" in summary


def test_labels_match_result_import():
    """Test that MatchResult can be imported from sleap_io."""
    import sleap_io as sio

    # Should be importable directly
    assert hasattr(sio, "MatchResult")

    # Should be usable for type hints
    result = sio.MatchResult()
    assert isinstance(result.video_map, dict)
    assert isinstance(result.skeleton_map, dict)
    assert isinstance(result.track_map, dict)


def _synthetic_mosaic_labels(tmp_path):
    """Build labels with a 2-tile mosaic of one source plus an uncropped video.

    Returns ``(labels, src, n_tiles, inst_points)`` where ``src`` is the
    uncropped source video for the mosaic, ``n_tiles`` is the number of cropped
    tiles, and ``inst_points`` is the point array placed on the first tile.
    """
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(4, 32, 64, 3), dtype=np.uint8)
    src_path = tmp_path / "src.mp4"
    sleap_io.save_video(frames, src_path)

    src = Video.from_filename(src_path.as_posix())
    left = src.crop((0, 0, 32, 32))
    right = src.crop((32, 0, 64, 32))

    other_frames = rng.integers(0, 255, size=(3, 16, 16, 3), dtype=np.uint8)
    other_path = tmp_path / "other.mp4"
    sleap_io.save_video(other_frames, other_path)
    other = Video.from_filename(other_path.as_posix())

    skel = Skeleton(["A", "B"])
    inst_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    lf = LabeledFrame(
        video=left,
        frame_idx=0,
        instances=[Instance.from_numpy(inst_points.copy(), skeleton=skel)],
    )
    labels = Labels(videos=[left, right, other], labeled_frames=[lf], skeletons=[skel])
    return labels, src, 2, inst_points


def test_apply_crops_mosaic_bakes_unique_files(tmp_path):
    """apply_crops bakes each mosaic tile to a unique file in video_dir."""
    labels, src, n_tiles, _ = _synthetic_mosaic_labels(tmp_path)
    out_dir = tmp_path / "baked"

    labels.apply_crops(video_dir=out_dir)

    baked_files = sorted(out_dir.glob("*.mp4"))
    # Both tiles share the stem "src", so they must be disambiguated.
    assert len(baked_files) == n_tiles
    names = [p.name for p in baked_files]
    assert len(set(names)) == n_tiles  # unique names
    assert all(p.exists() for p in baked_files)
    assert all(name.startswith("src_crop") for name in names)


def test_apply_crops_updates_references_to_baked_files(tmp_path):
    """apply_crops rewires labels.videos to baked files, dropping crop backends."""
    labels, src, n_tiles, _ = _synthetic_mosaic_labels(tmp_path)
    out_dir = tmp_path / "baked"

    labels.apply_crops(video_dir=out_dir)

    cropped_videos = [v for v in labels.videos if v.filename.startswith(str(out_dir))]
    assert len(cropped_videos) == n_tiles
    for v in cropped_videos:
        # No longer a virtual crop: backend is a plain media backend.
        assert not isinstance(v.backend, CropVideoBackend)
        assert v._crop_tuple() is None
        assert Path(v.filename).parent == out_dir
        # The labeled frame now points at the baked file.
    # The labeled frame's video reference was updated to a baked file.
    lf_video = labels[0].video
    assert lf_video in labels.videos
    assert not isinstance(lf_video.backend, CropVideoBackend)


def test_apply_crops_preserves_source_provenance(tmp_path):
    """Baked videos keep source_video pointing at the uncropped original."""
    labels, src, n_tiles, _ = _synthetic_mosaic_labels(tmp_path)
    out_dir = tmp_path / "baked"

    labels.apply_crops(video_dir=out_dir)

    baked = [v for v in labels.videos if v.filename.startswith(str(out_dir))]
    for v in baked:
        assert v.source_video is not None
        # Cropped shape (32x32) differs from uncropped source shape (32x64).
        assert v.shape[1:3] == (32, 32)
        assert v.source_video.shape[1:3] == (32, 64)


def test_apply_crops_leaves_uncropped_video_untouched(tmp_path):
    """Uncropped videos are not rewritten or replaced by apply_crops."""
    labels, src, n_tiles, _ = _synthetic_mosaic_labels(tmp_path)
    other_before = labels.videos[2]
    out_dir = tmp_path / "baked"

    labels.apply_crops(video_dir=out_dir)

    # The uncropped video object is the same instance, untouched.
    assert other_before in labels.videos
    assert labels.videos[2] is other_before
    assert other_before.filename.endswith("other.mp4")
    # No baked file was created for the uncropped video.
    assert not (out_dir / "other_crop.mp4").exists()


def test_apply_crops_coordinates_unchanged(tmp_path):
    """Instance point coordinates are unchanged after baking (coord-neutral)."""
    labels, src, n_tiles, inst_points = _synthetic_mosaic_labels(tmp_path)
    out_dir = tmp_path / "baked"

    labels.apply_crops(video_dir=out_dir)

    baked_points = labels[0].instances[0].numpy()
    assert_allclose(baked_points, inst_points)


def test_apply_crops_default_dir_beside_source(tmp_path):
    """Without video_dir, baked files are written next to the source video."""
    labels, src, n_tiles, _ = _synthetic_mosaic_labels(tmp_path)

    labels.apply_crops()

    baked = [v for v in labels.videos if "_crop" in Path(v.filename).name]
    assert len(baked) == n_tiles
    for v in baked:
        assert Path(v.filename).parent == tmp_path
        assert Path(v.filename).exists()


def test_apply_crops_empty_suffix_refuses_to_overwrite_source(tmp_path):
    """An empty suffix next to the source would overwrite it; apply_crops raises."""
    # Single cropped tile over one source so the baked name is "{stem}{suffix}.mp4"
    # with no index disambiguation; an empty suffix collides with the source file.
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(4, 32, 64, 3), dtype=np.uint8)
    src_path = tmp_path / "src.mp4"
    sleap_io.save_video(frames, src_path)
    src = Video.from_filename(src_path.as_posix())
    tile = src.crop((0, 0, 32, 32))
    labels = Labels(videos=[tile])

    with pytest.raises(ValueError, match="overwrite a source video"):
        labels.apply_crops(suffix="")
    # The source file is left intact and no replacement happened.
    assert isinstance(labels.videos[0].backend, CropVideoBackend)


def test_labels_match_with_matcher_objects():
    """Test Labels.match with actual Matcher objects (not strings)."""
    skeleton = Skeleton(["head", "tail"])
    video_gt = Video(filename="/data/video.mp4", open_backend=False)
    video_pred = Video(filename="/output/video.mp4", open_backend=False)
    track_gt = Track("animal_1")
    track_pred = Track("animal_1")

    gt_labels = Labels(videos=[video_gt], skeletons=[skeleton], tracks=[track_gt])
    pred_labels = Labels(videos=[video_pred], skeletons=[skeleton], tracks=[track_pred])

    # Pass actual Matcher objects instead of strings
    from sleap_io.model.matching import (
        SkeletonMatcher,
        SkeletonMatchMethod,
        TrackMatcher,
        TrackMatchMethod,
        VideoMatcher,
        VideoMatchMethod,
    )

    skeleton_matcher = SkeletonMatcher(method=SkeletonMatchMethod.STRUCTURE)
    video_matcher = VideoMatcher(method=VideoMatchMethod.BASENAME)
    track_matcher = TrackMatcher(method=TrackMatchMethod.NAME)

    result = gt_labels.match(
        pred_labels,
        skeleton=skeleton_matcher,
        video=video_matcher,
        track=track_matcher,
    )

    assert result.all_videos_matched
    assert result.all_skeletons_matched
    assert result.all_tracks_matched


def test_labels_match_summary_many_unmatched():
    """Test MatchResult.summary() with more than 5 unmatched videos."""
    skeleton = Skeleton(["head", "tail"])

    # Create GT with one video
    video_gt = Video(filename="/data/video_gt.mp4", open_backend=False)
    gt_labels = Labels(videos=[video_gt], skeletons=[skeleton])

    # Create predictions with 7 videos that don't match (different basenames)
    pred_videos = [
        Video(filename=f"/output/pred_{i}.mp4", open_backend=False) for i in range(7)
    ]
    pred_labels = Labels(videos=pred_videos, skeletons=[skeleton])

    result = gt_labels.match(pred_labels)

    # All 7 should be unmatched (different basenames from GT)
    assert len(result.unmatched_videos) == 7

    summary = result.summary()
    assert "Videos: 0/7 matched" in summary
    assert "Unmatched videos:" in summary
    assert "... and 2 more" in summary  # 7 - 5 = 2 more


def test_labels_match_summary_image_video():
    """Test MatchResult.summary() with ImageVideo (filename is a list)."""
    skeleton = Skeleton(["head", "tail"])

    # Create GT with a regular video
    video_gt = Video(filename="/data/video.mp4", open_backend=False)
    gt_labels = Labels(videos=[video_gt], skeletons=[skeleton])

    # Create predictions with an ImageVideo (list of filenames)
    video_pred = Video(
        filename=["/output/frame001.png", "/output/frame002.png"],
        open_backend=False,
    )
    pred_labels = Labels(videos=[video_pred], skeletons=[skeleton])

    result = gt_labels.match(pred_labels)

    # Should not match (different type)
    assert len(result.unmatched_videos) == 1

    # Summary should handle list filename
    summary = result.summary()
    assert "Unmatched videos:" in summary
    assert "/output/frame001.png" in summary  # First filename from list


def test_labels_copy_basic(slp_minimal):
    """Test basic copy creates independent object."""
    labels = load_slp(slp_minimal)
    labels_copy = labels.copy()

    # Assert independence
    assert labels_copy is not labels
    assert labels_copy.labeled_frames is not labels.labeled_frames
    assert labels_copy.videos is not labels.videos
    assert labels_copy.skeletons is not labels.skeletons
    assert labels_copy.tracks is not labels.tracks

    # Assert equivalence
    assert len(labels_copy) == len(labels)
    assert len(labels_copy.videos) == len(labels.videos)
    assert len(labels_copy.skeletons) == len(labels.skeletons)
    assert len(labels_copy.tracks) == len(labels.tracks)


def test_labels_copy_independence(slp_real_data):
    """Test modifications to copy don't affect original."""
    labels = load_slp(slp_real_data)
    labels_copy = labels.copy()

    # Store original value
    original_point = labels.labeled_frames[0].instances[0].points["xy"][0].copy()

    # Modify copy
    labels_copy.labeled_frames[0].instances[0].points["xy"][0] = [999.0, 999.0]

    # Assert original unchanged
    assert np.array_equal(
        labels.labeled_frames[0].instances[0].points["xy"][0], original_point
    )
    assert not np.array_equal(
        labels.labeled_frames[0].instances[0].points["xy"][0], [999.0, 999.0]
    )


def test_labels_copy_preserves_video_references(slp_real_data):
    """Test object references are preserved within copy."""
    labels = load_slp(slp_real_data)
    labels_copy = labels.copy()

    # Get video reference from first frame
    video_from_frame = labels_copy.labeled_frames[0].video
    # Same video should be in videos list (by identity)
    assert video_from_frame in labels_copy.videos

    # But should not be same object as original
    original_video = labels.labeled_frames[0].video
    assert video_from_frame is not original_video


def test_labels_copy_preserves_skeleton_references(slp_minimal):
    """Test skeleton references are preserved within copy."""
    labels = load_slp(slp_minimal)
    labels_copy = labels.copy()

    # Check skeleton references
    skeleton_from_inst = labels_copy.labeled_frames[0].instances[0].skeleton
    assert skeleton_from_inst in labels_copy.skeletons

    # Skeleton should be copied
    original_skeleton = labels.labeled_frames[0].instances[0].skeleton
    assert skeleton_from_inst is not original_skeleton


def test_labels_copy_video_backend(centered_pair_low_quality_video):
    """Test video backends are properly handled in copy."""
    labels = Labels([LabeledFrame(video=centered_pair_low_quality_video, frame_idx=0)])

    # Open backend
    _ = labels.videos[0][0]
    assert labels.videos[0].backend is not None

    # Copy
    labels_copy = labels.copy()

    # Backend should work in copy
    frame = labels_copy.videos[0][0]
    assert frame is not None
    assert frame.shape[0] > 0  # Should have valid frame data


def test_labels_copy_tracks(centered_pair):
    """Test track references are preserved."""
    labels = load_slp(centered_pair)
    labels_copy = labels.copy()

    # Find instance with track in copy
    inst_with_track = None
    for lf in labels_copy.labeled_frames:
        for inst in lf.instances:
            if inst.track is not None:
                inst_with_track = inst
                break
        if inst_with_track:
            break

    # Should have found at least one
    assert inst_with_track is not None

    # Track should be in tracks list
    assert inst_with_track.track in labels_copy.tracks

    # But should be copied
    original_inst = None
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            if inst.track is not None:
                original_inst = inst
                break
        if original_inst:
            break

    assert inst_with_track.track is not original_inst.track


def test_labels_copy_suggestions(slp_real_data):
    """Test suggestion frames are copied."""
    labels = load_slp(slp_real_data)
    labels_copy = labels.copy()

    # Independent lists
    assert labels_copy.suggestions is not labels.suggestions
    assert len(labels_copy.suggestions) == len(labels.suggestions)

    # Video references preserved within copy
    for sf in labels_copy.suggestions:
        assert sf.video in labels_copy.videos

    # But should be independent from original
    for sf_copy, sf_orig in zip(labels_copy.suggestions, labels.suggestions):
        assert sf_copy is not sf_orig
        assert sf_copy.video is not sf_orig.video


def test_labels_copy_sessions(slp_multiview):
    """Test recording sessions are copied."""
    labels = load_slp(slp_multiview)
    labels_copy = labels.copy()

    assert labels_copy.sessions is not labels.sessions
    assert len(labels_copy.sessions) == len(labels.sessions)

    # Sessions should be independent
    for session_copy, session_orig in zip(labels_copy.sessions, labels.sessions):
        assert session_copy is not session_orig

    # Check camera-video mappings preserved in copy
    if len(labels_copy.sessions) > 0:
        session = labels_copy.sessions[0]
        for camera, video in session._video_by_camera.items():
            assert video in labels_copy.videos


def test_labels_copy_provenance(slp_real_data):
    """Test provenance dict is copied."""
    labels = load_slp(slp_real_data)
    labels.provenance["test_key"] = "test_value"

    labels_copy = labels.copy()

    # Independent dict
    assert labels_copy.provenance is not labels.provenance
    assert labels_copy.provenance["test_key"] == "test_value"

    # Mutation test
    labels_copy.provenance["new_key"] = "new_value"
    assert "new_key" not in labels.provenance


def test_labels_copy_numpy_arrays(slp_minimal):
    """Test numpy arrays are properly copied."""
    labels = load_slp(slp_minimal)
    labels_copy = labels.copy()

    # Get points from both
    orig_points = labels.labeled_frames[0].instances[0].points
    copy_points = labels_copy.labeled_frames[0].instances[0].points

    # Different arrays
    assert orig_points is not copy_points

    # Same values
    assert np.array_equal(orig_points["xy"], copy_points["xy"])
    assert np.array_equal(orig_points["visible"], copy_points["visible"])

    # Independence - modify copy
    original_xy = orig_points["xy"][0].copy()
    copy_points["xy"][0] = [999.0, 999.0]
    assert np.array_equal(orig_points["xy"][0], original_xy)
    assert not np.array_equal(orig_points["xy"][0], [999.0, 999.0])


def test_labels_copy_skeleton_structure(slp_real_data):
    """Test skeleton edges and symmetries are copied."""
    labels = load_slp(slp_real_data)
    labels_copy = labels.copy()

    # Independent objects
    assert labels_copy.skeletons[0] is not labels.skeletons[0]

    # Same structure
    assert len(labels_copy.skeletons[0].nodes) == len(labels.skeletons[0].nodes)
    assert len(labels_copy.skeletons[0].edges) == len(labels.skeletons[0].edges)


def test_labels_copy_empty():
    """Test copying empty labels."""
    labels = Labels()
    labels_copy = labels.copy()

    assert len(labels_copy) == 0
    assert labels_copy is not labels


def test_labels_copy_backend_metadata():
    """Test backend_metadata dict is copied (PR #243 regression test)."""
    video = Video(filename="test.mp4", backend_metadata={"key": "value"})
    labels = Labels(videos=[video])
    labels_copy = labels.copy()

    # Modify copy metadata
    labels_copy.videos[0].backend_metadata["new_key"] = "new_value"

    # Original unchanged
    assert "new_key" not in labels.videos[0].backend_metadata
    assert labels.videos[0].backend_metadata == {"key": "value"}


def test_labels_copy_open_videos_default(centered_pair_low_quality_video):
    """Test default open_videos=None preserves original settings."""
    # Create labels with mixed open_backend settings
    video1 = Video(filename="test1.mp4", open_backend=True)
    video2 = Video(filename="test2.mp4", open_backend=False)
    labels = Labels(videos=[video1, video2])

    labels_copy = labels.copy()

    # Settings preserved per-video
    assert labels_copy.videos[0].open_backend is True
    assert labels_copy.videos[1].open_backend is False


def test_labels_copy_open_videos_true():
    """Test open_videos=True enables auto-opening for all videos."""
    video1 = Video(filename="test1.mp4", open_backend=True)
    video2 = Video(filename="test2.mp4", open_backend=False)
    labels = Labels(videos=[video1, video2])

    labels_copy = labels.copy(open_videos=True)

    # All videos have open_backend=True
    assert labels_copy.videos[0].open_backend is True
    assert labels_copy.videos[1].open_backend is True

    # Originals unchanged
    assert labels.videos[0].open_backend is True
    assert labels.videos[1].open_backend is False


def test_labels_copy_open_videos_false(centered_pair_low_quality_video):
    """Test open_videos=False disables auto-opening for all videos."""
    labels = Labels([LabeledFrame(video=centered_pair_low_quality_video, frame_idx=0)])

    # Verify original has auto-open enabled
    assert labels.videos[0].open_backend is True

    labels_copy = labels.copy(open_videos=False)

    # Copy has auto-open disabled
    assert labels_copy.videos[0].open_backend is False
    # Backend should not open on copy
    assert labels_copy.videos[0].backend is None

    # Original unchanged
    assert labels.videos[0].open_backend is True


def test_labels_copy_performance_profile(
    slp_minimal, slp_real_data, centered_pair, slp_multiview
):
    """Profile copy performance on various fixture sizes.

    This test measures and reports performance characteristics of the copy operation
    across different dataset sizes. It does not enforce strict performance limits,
    but documents expected performance ranges.
    """
    import time

    fixtures = {
        "minimal": load_slp(slp_minimal),
        "real_data": load_slp(slp_real_data),
        "centered_pair": load_slp(centered_pair),
        "multiview": load_slp(slp_multiview),
    }

    results = {}
    for name, labels in fixtures.items():
        # Gather metadata
        n_frames = len(labels.labeled_frames)
        n_instances = sum(len(lf.instances) for lf in labels.labeled_frames)
        n_points = sum(
            inst.points.shape[0]
            for lf in labels.labeled_frames
            for inst in lf.instances
        )

        # Warm up (first copy may be slower due to caching)
        _ = labels.copy()

        # Benchmark with multiple runs for stability
        times = []
        for _ in range(5):
            start = time.perf_counter()
            labels_copy = labels.copy()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Verify copy worked
            assert labels_copy is not labels

        # Use median time for stability
        median_time = sorted(times)[len(times) // 2]

        results[name] = {
            "frames": n_frames,
            "instances": n_instances,
            "points": n_points,
            "time_ms": median_time * 1000,
            "time_per_instance_us": (median_time * 1e6) / max(n_instances, 1),
        }

    # Print report (helpful for development and CI logs)
    print("\n\nLabels.copy() Performance Profile")
    print("=" * 80)
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Frames: {stats['frames']}")
        print(f"  Instances: {stats['instances']}")
        print(f"  Total points: {stats['points']}")
        print(f"  Copy time: {stats['time_ms']:.2f} ms")
        print(f"  Time per instance: {stats['time_per_instance_us']:.2f} µs")
    print("=" * 80)

    # Sanity check: copy should complete in reasonable time
    # These are very generous limits just to catch major performance regressions
    assert results["minimal"]["time_ms"] < 100, "Minimal dataset copy too slow"
    assert results["centered_pair"]["time_ms"] < 5000, "Large dataset copy too slow"

    # Performance should scale roughly linearly with instance count
    # (this is a loose check, not a strict requirement)
    cp_instances = results["centered_pair"]["instances"]
    min_instances = results["minimal"]["instances"]
    if cp_instances > 0 and min_instances > 0:
        scale_factor = cp_instances / min_instances
        time_ratio = results["centered_pair"]["time_ms"] / max(
            results["minimal"]["time_ms"], 0.001
        )

        # Time ratio shouldn't be more than 100x the scale factor
        # (allows for some overhead that's not purely linear)
        assert time_ratio < scale_factor * 100, (
            f"Copy performance doesn't scale well: "
            f"{scale_factor:.1f}x more instances but {time_ratio:.1f}x slower"
        )


# =======================
# Fast Stats Tests
# =======================


def test_n_user_instances_basic():
    """Test n_user_instances returns correct count for eager-loaded Labels."""
    skel = Skeleton(["A", "B"])
    vid = Video(filename="test")
    labels = Labels(
        [
            LabeledFrame(
                video=vid,
                frame_idx=0,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel),
                    PredictedInstance([[4, 5], [6, 7]], skeleton=skel),
                ],
            ),
            LabeledFrame(
                video=vid,
                frame_idx=1,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel),
                    Instance([[4, 5], [6, 7]], skeleton=skel),
                    PredictedInstance([[8, 9], [10, 11]], skeleton=skel),
                ],
            ),
        ]
    )

    # 1 user instance in frame 0 + 2 user instances in frame 1 = 3 total
    assert labels.n_user_instances == 3


def test_n_pred_instances_basic():
    """Test n_pred_instances returns correct count for eager-loaded Labels."""
    skel = Skeleton(["A", "B"])
    vid = Video(filename="test")
    labels = Labels(
        [
            LabeledFrame(
                video=vid,
                frame_idx=0,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel),
                    PredictedInstance([[4, 5], [6, 7]], skeleton=skel),
                ],
            ),
            LabeledFrame(
                video=vid,
                frame_idx=1,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel),
                    PredictedInstance([[4, 5], [6, 7]], skeleton=skel),
                    PredictedInstance([[8, 9], [10, 11]], skeleton=skel),
                ],
            ),
        ]
    )

    # 1 predicted instance in frame 0 + 2 predicted instances in frame 1 = 3 total
    assert labels.n_pred_instances == 3


def test_n_user_instances_empty():
    """Test n_user_instances returns 0 for empty Labels."""
    labels = Labels()
    assert labels.n_user_instances == 0


def test_n_pred_instances_empty():
    """Test n_pred_instances returns 0 for empty Labels."""
    labels = Labels()
    assert labels.n_pred_instances == 0


def test_n_user_instances_lazy(centered_pair):
    """Test n_user_instances with lazy-loaded Labels matches eager."""
    lazy = load_slp(centered_pair, lazy=True)
    eager = load_slp(centered_pair, lazy=False)

    assert lazy.n_user_instances == eager.n_user_instances
    # centered_pair has 0 user instances
    assert lazy.n_user_instances == 0


def test_n_pred_instances_lazy(centered_pair):
    """Test n_pred_instances with lazy-loaded Labels matches eager."""
    lazy = load_slp(centered_pair, lazy=True)
    eager = load_slp(centered_pair, lazy=False)

    assert lazy.n_pred_instances == eager.n_pred_instances
    # centered_pair has 2274 predicted instances
    assert lazy.n_pred_instances == 2274


def test_n_frames_per_video_basic():
    """Test n_frames_per_video returns correct counts."""
    vid1 = Video(filename="video1")
    vid2 = Video(filename="video2")
    labels = Labels(
        [
            LabeledFrame(video=vid1, frame_idx=0, instances=[]),
            LabeledFrame(video=vid1, frame_idx=1, instances=[]),
            LabeledFrame(video=vid1, frame_idx=2, instances=[]),
            LabeledFrame(video=vid2, frame_idx=0, instances=[]),
        ]
    )

    frame_counts = labels.n_frames_per_video()
    assert frame_counts[vid1] == 3
    assert frame_counts[vid2] == 1


def test_n_frames_per_video_empty():
    """Test n_frames_per_video returns empty dict for empty Labels."""
    labels = Labels()
    assert labels.n_frames_per_video() == {}


def test_n_frames_per_video_lazy(centered_pair):
    """Test n_frames_per_video with lazy-loaded Labels matches eager."""
    lazy = load_slp(centered_pair, lazy=True)
    eager = load_slp(centered_pair, lazy=False)

    lazy_counts = lazy.n_frames_per_video()
    eager_counts = eager.n_frames_per_video()

    # Should have same videos and counts
    for vid in eager.videos:
        lazy_vid = lazy.videos[eager.videos.index(vid)]
        assert lazy_counts[lazy_vid] == eager_counts[vid]


def test_n_instances_per_track_basic():
    """Test n_instances_per_track returns correct counts."""
    skel = Skeleton(["A", "B"])
    vid = Video(filename="test")
    track1 = Track("track1")
    track2 = Track("track2")
    labels = Labels(
        [
            LabeledFrame(
                video=vid,
                frame_idx=0,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel, track=track1),
                    Instance([[4, 5], [6, 7]], skeleton=skel, track=track2),
                ],
            ),
            LabeledFrame(
                video=vid,
                frame_idx=1,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=skel, track=track1),
                    Instance([[4, 5], [6, 7]], skeleton=skel),  # No track
                ],
            ),
        ],
        tracks=[track1, track2],
    )

    track_counts = labels.n_instances_per_track()
    assert track_counts[track1] == 2  # 2 instances with track1
    assert track_counts[track2] == 1  # 1 instance with track2


def test_n_instances_per_track_empty():
    """Test n_instances_per_track returns empty dict for empty Labels."""
    labels = Labels()
    assert labels.n_instances_per_track() == {}


def test_n_instances_per_track_no_tracks():
    """Test n_instances_per_track with Labels that have no tracks defined."""
    skel = Skeleton(["A", "B"])
    vid = Video(filename="test")
    labels = Labels(
        [
            LabeledFrame(
                video=vid,
                frame_idx=0,
                instances=[Instance([[0, 1], [2, 3]], skeleton=skel)],
            ),
        ]
    )

    # No tracks defined, should return empty dict
    assert labels.n_instances_per_track() == {}


def test_n_instances_per_track_lazy(centered_pair):
    """Test n_instances_per_track with lazy-loaded Labels matches eager."""
    lazy = load_slp(centered_pair, lazy=True)
    eager = load_slp(centered_pair, lazy=False)

    lazy_counts = lazy.n_instances_per_track()
    eager_counts = eager.n_instances_per_track()

    # Should have same tracks and counts
    for track in eager.tracks:
        lazy_track = lazy.tracks[eager.tracks.index(track)]
        assert lazy_counts[lazy_track] == eager_counts[track]


def test_n_user_frames_basic():
    """Test n_user_frames returns correct count for eager-loaded Labels."""
    video = Video(filename="test")
    skeleton = Skeleton(["A", "B"])

    # Create a mix of user and predicted frames
    user_inst = Instance.from_numpy(
        np.array([[10, 20], [30, 40]]),
        skeleton=skeleton,
    )
    pred_inst = PredictedInstance.from_numpy(
        np.array([[10, 20], [30, 40]]),
        skeleton=skeleton,
        point_scores=np.array([0.9, 0.8]),
        score=0.85,
    )

    labels = Labels(
        [
            # Frame 0: user instance only
            LabeledFrame(video=video, frame_idx=0, instances=[user_inst]),
            # Frame 1: predicted instance only
            LabeledFrame(video=video, frame_idx=1, instances=[pred_inst]),
            # Frame 2: user instance only
            LabeledFrame(video=video, frame_idx=2, instances=[user_inst]),
            # Frame 3: both user and predicted
            LabeledFrame(video=video, frame_idx=3, instances=[user_inst, pred_inst]),
        ]
    )

    # Frames 0, 2, and 3 have user instances
    assert labels.n_user_frames == 3
    # Total frames is 4
    assert len(labels.labeled_frames) == 4


def test_n_user_frames_empty():
    """Test n_user_frames returns 0 for empty Labels."""
    labels = Labels()
    assert labels.n_user_frames == 0


def test_n_user_frames_no_user_instances():
    """Test n_user_frames returns 0 when only predicted instances exist."""
    video = Video(filename="test")
    skeleton = Skeleton(["A", "B"])
    pred_inst = PredictedInstance.from_numpy(
        np.array([[10, 20], [30, 40]]),
        skeleton=skeleton,
        point_scores=np.array([0.9, 0.8]),
        score=0.85,
    )

    labels = Labels(
        [
            LabeledFrame(video=video, frame_idx=0, instances=[pred_inst]),
            LabeledFrame(video=video, frame_idx=1, instances=[pred_inst]),
        ]
    )

    assert labels.n_user_frames == 0


def test_n_user_frames_lazy(centered_pair):
    """Test n_user_frames with lazy-loaded Labels matches eager."""
    lazy = load_slp(centered_pair, lazy=True)
    eager = load_slp(centered_pair, lazy=False)

    assert lazy.n_user_frames == eager.n_user_frames
    # centered_pair has 0 user instances, so 0 user frames
    assert lazy.n_user_frames == 0


def test_n_user_frames_lazy_with_user_instances(slp_real_data):
    """Test n_user_frames with lazy-loaded Labels that has user instances."""
    lazy = load_slp(slp_real_data, lazy=True)
    eager = load_slp(slp_real_data, lazy=False)

    # Should match eager
    assert lazy.n_user_frames == eager.n_user_frames
    # slp_real_data has user instances
    assert lazy.n_user_frames > 0


# =============================================================================
# Tests for Labels.add_video() - PROPOSED NEW METHOD
# =============================================================================
# These tests document expected behavior for a new method that safely adds
# videos while preventing duplicates.
# Current implementation: Method does not exist (tests will fail with AttributeError).


class TestLabelsAddVideo:
    """Tests for Labels.add_video() method.

    This method is needed to prevent duplicate video creation when adding
    videos to Labels. The current pattern of `labels.videos.append(video)`
    or `video not in labels.videos` fails to detect duplicates because
    Video uses identity comparison (eq=False).

    Background (from merge investigation):
    The GUI uses `video not in context.labels.videos` to check for duplicates,
    but this always returns True for new Video objects even if they point to
    the same file. This leads to:
    1. Duplicate videos in the Labels.videos list
    2. Downstream deletion bugs (remove_video deletes frames from ALL matching videos)
    3. Data corruption when merging predictions
    """

    def test_add_video_new_video(self):
        """Adding a new video should append it to the list.

        Basic case: Adding a video that doesn't exist in Labels should
        add it and return the same video object.
        """
        labels = Labels()
        video = Video(filename="/data/video.mp4", open_backend=False)

        result = labels.add_video(video)

        assert result is video
        assert len(labels.videos) == 1
        assert labels.videos[0] is video

    def test_add_video_prevents_duplicate_same_path(self):
        """Adding video with same path should return existing video.

        Use case: User accidentally adds the same video twice via GUI.
        Instead of creating a duplicate, add_video should recognize the
        existing video and return it.
        """
        labels = Labels()
        video1 = Video(filename="/data/video.mp4", open_backend=False)
        labels.add_video(video1)

        # Create a NEW Video object with the SAME path
        video2 = Video(filename="/data/video.mp4", open_backend=False)
        result = labels.add_video(video2)

        # Should return the existing video, not add a duplicate
        assert result is video1
        assert len(labels.videos) == 1
        assert labels.videos[0] is video1

    def test_add_video_source_video_match(self):
        """Adding embedded video should match external video with same source.

        Use case (UC3): After loading predictions from PKG.SLP, the prediction
        video has source_video pointing to the original. When iterating through
        frames and adding videos, we should recognize the source_video match.
        """
        labels = Labels()
        external = Video(filename="/data/recordings/video.mp4", open_backend=False)
        labels.add_video(external)

        # Embedded video from PKG.SLP with source_video pointing to external
        source = Video(filename="/data/recordings/video.mp4", open_backend=False)
        embedded = Video(
            filename="predictions.pkg.slp", source_video=source, open_backend=False
        )
        result = labels.add_video(embedded)

        # Should recognize as same file via source_video and return external
        assert result is external
        assert len(labels.videos) == 1

    def test_add_video_different_videos(self):
        """Adding different videos should add all of them."""
        labels = Labels()
        video1 = Video(filename="/data/video_a.mp4", open_backend=False)
        video2 = Video(filename="/data/video_b.mp4", open_backend=False)

        result1 = labels.add_video(video1)
        result2 = labels.add_video(video2)

        assert result1 is video1
        assert result2 is video2
        assert len(labels.videos) == 2

    def test_add_video_same_basename_different_dir(self):
        """Videos with same basename but different directories are different.

        This tests that add_video doesn't incorrectly deduplicate videos
        that happen to have the same filename but are in different directories.

        Real-world example: Multiple experiments may have `fly.mp4` in
        different directories (exp1/fly.mp4, exp2/fly.mp4).
        """
        labels = Labels()
        video1 = Video(filename="/data/exp1/fly.mp4", open_backend=False)
        video2 = Video(filename="/data/exp2/fly.mp4", open_backend=False)

        labels.add_video(video1)
        result = labels.add_video(video2)

        assert result is video2
        assert len(labels.videos) == 2

    def test_add_video_imagevideo(self):
        """Adding ImageVideo with same image list should detect duplicate."""
        labels = Labels()
        paths = ["/data/img_000.jpg", "/data/img_001.jpg", "/data/img_002.jpg"]
        video1 = Video(filename=paths.copy(), open_backend=False)
        labels.add_video(video1)

        # Create a new ImageVideo with the same paths
        video2 = Video(filename=paths.copy(), open_backend=False)
        result = labels.add_video(video2)

        assert result is video1
        assert len(labels.videos) == 1


def test_labels_with_rois_and_masks():
    video = Video(filename="test.mp4")
    roi1 = UserROI.from_bbox(0, 0, 10, 10, video=video)
    roi2 = UserROI.from_bbox(5, 5, 20, 20, video=video)
    mask1 = UserSegmentationMask.from_numpy(np.zeros((10, 10), dtype=bool))

    lf1 = LabeledFrame(video=video, frame_idx=1, masks=[mask1])
    lf3 = LabeledFrame(video=video, frame_idx=3, rois=[roi2])
    labels = Labels(labeled_frames=[lf1, lf3], videos=[video], rois=[roi1])
    assert len(labels.rois) == 2
    assert len(labels.masks) == 1


def test_labels_static_and_temporal_rois():
    video = Video(filename="test.mp4")
    static_roi = UserROI.from_bbox(0, 0, 100, 100, video=video)
    temporal_roi = UserROI.from_bbox(10, 10, 20, 20, video=video)

    lf5 = LabeledFrame(video=video, frame_idx=5, rois=[temporal_roi])
    labels = Labels(labeled_frames=[lf5], videos=[video], rois=[static_roi])
    assert len(labels.static_rois) == 1
    assert labels.static_rois[0] is static_roi
    assert len(labels.temporal_rois) == 1
    assert labels.temporal_rois[0] is temporal_roi


def test_labels_get_rois():
    video1 = Video(filename="v1.mp4")
    video2 = Video(filename="v2.mp4")
    roi1 = UserROI.from_bbox(0, 0, 10, 10, video=video1, category="cat")
    roi2 = UserROI.from_bbox(5, 5, 10, 10, video=video2, category="dog")
    roi3 = UserROI.from_polygon(
        [(0, 0), (10, 0), (10, 10)],
        category="arena",
    )

    lf0_v1 = LabeledFrame(video=video1, frame_idx=0, rois=[roi1, roi3])
    lf1_v2 = LabeledFrame(video=video2, frame_idx=1, rois=[roi2])
    labels = Labels(labeled_frames=[lf0_v1, lf1_v2], videos=[video1, video2])

    assert len(labels.get_rois(video=video1)) == 2
    assert len(labels.get_rois(video=video2)) == 1
    assert len(labels.get_rois(frame_idx=0)) == 2
    assert len(labels.get_rois(category="cat")) == 1
    assert len(labels.get_rois(category="arena")) == 1


def test_labels_get_masks():
    video = Video(filename="test.mp4")
    mask1 = UserSegmentationMask.from_numpy(np.ones((5, 5), dtype=bool), category="cat")
    mask2 = UserSegmentationMask.from_numpy(np.ones((5, 5), dtype=bool), category="dog")

    lf0 = LabeledFrame(video=video, frame_idx=0, masks=[mask1])
    lf1 = LabeledFrame(video=video, frame_idx=1, masks=[mask2])
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video])
    assert len(labels.get_masks(frame_idx=0)) == 1
    assert len(labels.get_masks(category="dog")) == 1
    assert len(labels.get_masks(video=video)) == 2


def test_labels_get_rois_by_track_and_instance():
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    track1 = Track(name="t1")
    track2 = Track(name="t2")
    inst = Instance(np.array([[0, 0]]), skeleton=skeleton)

    roi1 = UserROI.from_bbox(0, 0, 10, 10, video=video, track=track1)
    roi2 = UserROI.from_bbox(5, 5, 10, 10, video=video, track=track2, instance=inst)
    roi3 = UserROI.from_bbox(1, 1, 5, 5, video=video)

    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
        tracks=[track1, track2],
        rois=[roi1, roi2, roi3],
    )
    assert len(labels.get_rois(track=track1)) == 1
    assert labels.get_rois(track=track1)[0] is roi1
    assert len(labels.get_rois(track=track2)) == 1
    assert len(labels.get_rois(instance=inst)) == 1
    assert labels.get_rois(instance=inst)[0] is roi2


def test_labels_get_masks_by_track():
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    mask1 = UserSegmentationMask.from_numpy(np.ones((5, 5), dtype=bool), track=track)
    mask2 = UserSegmentationMask.from_numpy(np.ones((5, 5), dtype=bool))

    lf = LabeledFrame(video=video, frame_idx=0, masks=[mask1, mask2])
    labels = Labels(labeled_frames=[lf], videos=[video], tracks=[track])
    assert len(labels.get_masks(track=track)) == 1
    assert labels.get_masks(track=track)[0] is mask1


def test_labels_replace_videos_updates_rois_and_masks():
    old_video = Video(filename="old.mp4")
    new_video = Video(filename="new.mp4")
    roi = UserROI.from_bbox(0, 0, 10, 10, video=old_video)
    mask = UserSegmentationMask.from_numpy(np.zeros((5, 5), dtype=bool))

    lf = LabeledFrame(video=old_video, frame_idx=0, rois=[roi], masks=[mask])
    labels = Labels(labeled_frames=[lf], videos=[old_video])
    labels.replace_videos(old_videos=[old_video], new_videos=[new_video])

    assert roi.video is new_video
    assert labels.videos[0] is new_video


def test_labels_materialize_rois_and_masks(tmp_path):
    """materialize() deep copies ROIs and masks, relinking video/track refs."""
    video = Video(filename="test.mp4")
    track = Track(name="track0")
    skeleton = Skeleton(["A"])
    roi_with_refs = UserROI.from_bbox(0, 0, 10, 10, video=video)
    roi_with_refs.track = track
    roi_no_refs = UserROI.from_bbox(5, 5, 10, 10)  # No video or track
    mask_with_refs = UserSegmentationMask.from_numpy(np.ones((5, 5), dtype=bool))
    mask_with_refs.track = track
    mask_no_refs = UserSegmentationMask.from_numpy(np.ones((3, 3), dtype=bool))

    lf = LabeledFrame(
        video=video,
        frame_idx=0,
        rois=[roi_with_refs, roi_no_refs],
        masks=[mask_with_refs, mask_no_refs],
    )
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        tracks=[track],
        skeletons=[skeleton],
    )
    path = str(tmp_path / "test.slp")
    sleap_io.save_slp(labels, path)

    lazy = sleap_io.load_slp(path, lazy=True)
    materialized = lazy.materialize()

    # ROIs and masks present after materialization
    assert len(materialized.rois) == 2
    assert len(materialized.masks) == 2

    # Video/track references point to the materialized copies
    assert materialized.rois[0].video is materialized.videos[0]
    assert materialized.rois[0].track is materialized.tracks[0]
    assert materialized.masks[0].track is materialized.tracks[0]

    # ROI/mask without track refs stay None
    assert materialized.rois[1].track is None
    assert materialized.masks[1].track is None


def test_labels_copy_preserves_rois_and_masks(slp_minimal, tmp_path):
    """Test that lazy copy preserves ROIs and masks."""
    labels = load_slp(slp_minimal)
    video = labels.videos[0]

    roi = UserROI.from_bbox(10, 20, 30, 40, video=video, category="test")
    mask = UserSegmentationMask.from_numpy(np.ones((5, 5), dtype=bool), category="fg")
    lf = labels._find_or_create_frame(video, 0)
    lf.rois.append(roi)
    lf.masks.append(mask)
    labels._invalidate_indices()

    # Save and reload lazily
    out_path = tmp_path / "with_rois.slp"
    labels.save(str(out_path))
    lazy_labels = load_slp(str(out_path), lazy=True)

    assert len(lazy_labels.rois) == 1
    assert len(lazy_labels.masks) == 1

    # Copy the lazy labels
    labels_copy = lazy_labels.copy()

    assert len(labels_copy.rois) == 1
    assert labels_copy.rois[0].category == "test"
    assert len(labels_copy.masks) == 1
    assert labels_copy.masks[0].category == "fg"

    # Verify independence (properties return new lists)
    assert len(lazy_labels.rois) == 1


def test_labels_copy_preserves_label_images(slp_minimal, tmp_path):
    """Test that lazy copy preserves label_images."""
    labels = load_slp(slp_minimal)
    video = labels.videos[0]
    track = Track(name="t1")

    li = UserLabelImage(
        data=np.array([[0, 1], [2, 0]], dtype=np.int32),
        objects={
            1: LabelImage.Info(track=track, category="neuron"),
            2: LabelImage.Info(category="glia"),
        },
    )
    lf = labels._find_or_create_frame(video, 0)
    lf.label_images.append(li)
    labels._invalidate_indices()
    labels.tracks.append(track)

    # Save and reload lazily
    out_path = tmp_path / "with_li.slp"
    labels.save(str(out_path))
    lazy_labels = load_slp(str(out_path), lazy=True)

    assert len(lazy_labels.label_images) == 1

    # Copy the lazy labels
    labels_copy = lazy_labels.copy()
    assert len(labels_copy.label_images) == 1
    assert labels_copy.label_images[0].n_objects == 2

    # Verify independence (properties return new lists)
    assert len(lazy_labels.label_images) == 1


def test_labels_materialize_label_images(tmp_path):
    """materialize() deep copies label_images, relinking video/track/instance refs."""
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    skeleton = Skeleton(["A"])
    instance = Instance.from_numpy(
        np.array([[10, 20]], dtype=np.float32), skeleton=skeleton
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])

    li = UserLabelImage(
        data=np.array([[0, 1], [2, 0]], dtype=np.int32),
        objects={
            1: LabelImage.Info(track=track, category="neuron", instance=instance),
            2: LabelImage.Info(category="glia"),
        },
    )
    lf.label_images.append(li)

    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        tracks=[track],
        skeletons=[skeleton],
    )
    path = str(tmp_path / "test.slp")
    sleap_io.save_slp(labels, path)

    lazy = sleap_io.load_slp(path, lazy=True)
    materialized = lazy.materialize()

    assert len(materialized.label_images) == 1
    mat_li = materialized.label_images[0]

    # Track in objects points to new copy
    for info in mat_li.objects.values():
        if info.track is not None:
            assert info.track is materialized.tracks[0]
            assert info.track is not lazy.tracks[0]

    # Instance in objects resolved via _instance_idx
    assert mat_li.objects[1].instance is materialized.labeled_frames[0].instances[0]
    assert mat_li.objects[1]._instance_idx == -1
    assert mat_li.objects[2].instance is None


def test_labels_get_masks_predicted():
    """get_masks filters by predicted flag."""
    video = Video(filename="test.mp4")
    user_mask = UserSegmentationMask.from_numpy(
        np.ones((5, 5), dtype=bool), category="a"
    )
    pred_mask = PredictedSegmentationMask.from_numpy(
        np.ones((5, 5), dtype=bool), category="b", score=0.9
    )
    lf = LabeledFrame(video=video, frame_idx=0, masks=[user_mask, pred_mask])
    labels = Labels(labeled_frames=[lf])

    assert len(labels.get_masks()) == 2
    assert labels.get_masks(predicted=True) == [pred_mask]
    assert labels.get_masks(predicted=False) == [user_mask]


def test_labels_get_rois_predicted():
    """get_rois filters by predicted flag."""
    user_roi = UserROI(geometry=box(0, 0, 5, 5), category="a")
    pred_roi = PredictedROI(geometry=box(0, 0, 5, 5), category="b", score=0.8)
    labels = Labels(rois=[user_roi, pred_roi])

    assert len(labels.get_rois()) == 2
    assert labels.get_rois(predicted=True) == [pred_roi]
    assert labels.get_rois(predicted=False) == [user_roi]


def test_labels_get_label_images_predicted():
    """get_label_images filters by predicted flag."""
    video = Video(filename="test.mp4")
    data = np.array([[0, 1]], dtype=np.int32)
    user_li = UserLabelImage(data=data)
    pred_li = PredictedLabelImage(data=data, score=0.9)
    lf = LabeledFrame(video=video, frame_idx=0, label_images=[user_li, pred_li])
    labels = Labels(labeled_frames=[lf])

    assert len(labels.get_label_images()) == 2
    assert labels.get_label_images(predicted=True) == [pred_li]
    assert labels.get_label_images(predicted=False) == [user_li]


def test_labels_get_bboxes():
    """get_bboxes filters by video, frame, category, and predicted."""
    video = Video(filename="test.mp4")
    bbox1 = UserBoundingBox(x1=45, y1=45, x2=55, y2=55, category="mouse")
    bbox2 = PredictedBoundingBox(x1=50, y1=50, x2=70, y2=70, category="fly", score=0.9)
    bbox3 = UserBoundingBox(x1=62.5, y1=62.5, x2=77.5, y2=77.5, category="mouse")

    lf0 = LabeledFrame(video=video, frame_idx=0, bboxes=[bbox1, bbox2])
    lf1 = LabeledFrame(video=video, frame_idx=1, bboxes=[bbox3])
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video])

    assert len(labels.get_bboxes(video=video)) == 3
    assert len(labels.get_bboxes(frame_idx=0)) == 2
    assert len(labels.get_bboxes(category="mouse")) == 2
    assert len(labels.get_bboxes(predicted=True)) == 1
    assert labels.get_bboxes(predicted=True)[0] is bbox2
    assert len(labels.get_bboxes(predicted=False)) == 2


def test_labels_replace_videos_updates_bboxes():
    """replace_videos should update labeled frame video references."""
    old_video = Video(filename="old.mp4")
    new_video = Video(filename="new.mp4")
    bbox = UserBoundingBox(x1=45, y1=45, x2=55, y2=55)
    lf = LabeledFrame(video=old_video, frame_idx=0, bboxes=[bbox])
    labels = Labels(labeled_frames=[lf], videos=[old_video])
    labels.replace_videos(old_videos=[old_video], new_videos=[new_video])
    assert lf.video is new_video


def test_get_rois_query():
    """get_rois should filter by video, frame, and other attributes."""
    from shapely.geometry import box

    video1 = Video.from_filename("v1.mp4")
    video2 = Video.from_filename("v2.mp4")

    rois = [
        UserROI(geometry=box(0, 0, 10, 10), video=video1),
        UserROI(geometry=box(0, 0, 10, 10), video=video1),
        UserROI(geometry=box(0, 0, 10, 10), video=video2),
    ]
    lf0_v1 = LabeledFrame(video=video1, frame_idx=0, rois=[rois[0]])
    lf1_v1 = LabeledFrame(video=video1, frame_idx=1, rois=[rois[1]])
    lf0_v2 = LabeledFrame(video=video2, frame_idx=0, rois=[rois[2]])
    labels = Labels(labeled_frames=[lf0_v1, lf1_v1, lf0_v2])

    # Query by video + frame_idx
    result = labels.get_rois(video=video1, frame_idx=0)
    assert len(result) == 1
    assert result[0] is rois[0]

    # Query by video only
    result = labels.get_rois(video=video1)
    assert len(result) == 2

    # Query by frame_idx only
    result = labels.get_rois(frame_idx=0)
    assert len(result) == 2

    # Mutation via direct append is immediately visible
    new_roi = UserROI(geometry=box(0, 0, 5, 5), video=video1)
    lf0_v1.rois.append(new_roi)
    labels._invalidate_indices()
    result = labels.get_rois(video=video1, frame_idx=0)
    assert len(result) == 2
    assert new_roi in result


def test_get_masks_query():
    """get_masks should filter by video, frame, and other attributes."""
    video = Video.from_filename("v.mp4")
    mask_data = np.zeros((10, 10), dtype=bool)
    mask_data[2:8, 2:8] = True

    masks = [
        UserSegmentationMask.from_numpy(mask_data),
        UserSegmentationMask.from_numpy(mask_data),
    ]
    lf0 = LabeledFrame(video=video, frame_idx=0, masks=[masks[0]])
    lf1 = LabeledFrame(video=video, frame_idx=1, masks=[masks[1]])
    labels = Labels(labeled_frames=[lf0, lf1])

    result = labels.get_masks(video=video, frame_idx=0)
    assert len(result) == 1
    assert result[0] is masks[0]

    result = labels.get_masks(video=video)
    assert len(result) == 2

    # Mutation via direct append is immediately visible
    new_mask = UserSegmentationMask.from_numpy(mask_data)
    lf0.masks.append(new_mask)
    labels._invalidate_indices()
    result = labels.get_masks(video=video, frame_idx=0)
    assert len(result) == 2
    assert new_mask in result


def test_labels_copy_with_annotation_refs(labels_all_annotations, tmp_path):
    """Test lazy copy path relinks label_image video/track/instance refs."""
    # Save and reload to create file-backed (lazy) Labels
    path = str(tmp_path / "all_annots.slp")
    save_slp(labels_all_annotations, path)
    loaded = load_slp(path, open_videos=False)

    # Exercise the lazy copy path
    labels_copy = loaded.copy()

    # All 6 label_images (3 User + 3 Predicted) are present
    assert len(labels_copy.label_images) == 6

    # Label images are deep copies, not the same objects
    assert labels_copy.label_images[0] is not loaded.label_images[0]

    # UserLabelImage and PredictedLabelImage types are preserved
    user_lis = [li for li in labels_copy.label_images if isinstance(li, UserLabelImage)]
    pred_lis = [
        li for li in labels_copy.label_images if isinstance(li, PredictedLabelImage)
    ]
    assert len(user_lis) == 3
    assert len(pred_lis) == 3

    # Track references in objects are valid tracks from the copy
    for li in labels_copy.label_images:
        for info in li.objects.values():
            if info.track is not None:
                assert info.track in labels_copy.tracks

    # Mutating the copy does not affect the original
    labels_copy.label_images.clear()
    assert len(loaded.label_images) == 6


def test_labels_close():
    """Labels.close() closes the h5py file handle and is idempotent."""
    labels = Labels()

    # No file handle — should not raise
    labels.close()

    # With a mock file handle
    class FakeFile:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake = FakeFile()
    labels._label_image_file = fake
    labels.close()
    assert fake.closed
    assert labels._label_image_file is None

    # Idempotent — calling again should not raise
    labels.close()


def test_labels_close_handles_exception():
    """Labels.close() suppresses exceptions from the file handle."""

    class BadFile:
        def close(self):
            raise OSError("simulated close failure")

    labels = Labels()
    labels._label_image_file = BadFile()
    labels.close()  # should not raise
    assert labels._label_image_file is None


def test_labels_del_drops_reference_without_force_close():
    """Labels.__del__() drops the file reference but does NOT forcibly close.

    Forcibly closing on GC would invalidate LabelImage lazy loaders whose
    h5py Dataset references rely on the file staying open. By just dropping
    the reference, h5py's own C-level refcounting keeps the file open as
    long as any Dataset (held by a LabelImage closure) still uses it.
    """
    labels = Labels()

    class FakeFile:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake = FakeFile()
    labels._label_image_file = fake
    labels.__del__()
    # Reference is dropped
    assert labels._label_image_file is None
    # But the file was NOT forcibly closed
    assert not fake.closed


def test_labels_get_centroids():
    """get_centroids filters by video, frame, category, track, and predicted."""
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    c1 = UserCentroid(x=10.0, y=20.0, category="cell", track=track)
    c2 = PredictedCentroid(x=30.0, y=40.0, category="lysosome", score=0.9)
    c3 = UserCentroid(x=50.0, y=60.0, category="cell")

    lf0 = LabeledFrame(video=video, frame_idx=0, centroids=[c1, c2])
    lf1 = LabeledFrame(video=video, frame_idx=1, centroids=[c3])
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video], tracks=[track])

    assert len(labels.get_centroids(video=video)) == 3
    assert len(labels.get_centroids(frame_idx=0)) == 2
    assert len(labels.get_centroids(category="cell")) == 2
    assert len(labels.get_centroids(track=track)) == 1
    assert labels.get_centroids(track=track)[0] is c1
    assert len(labels.get_centroids(predicted=True)) == 1
    assert labels.get_centroids(predicted=True)[0] is c2
    assert len(labels.get_centroids(predicted=False)) == 2


def test_labels_materialize_centroids(tmp_path):
    """materialize() deep copies centroids, relinking track refs."""
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    skeleton = Skeleton(["A"])
    c_with_refs = UserCentroid(x=1.0, y=2.0, track=track)
    c_no_refs = PredictedCentroid(x=3.0, y=4.0, score=0.8)

    lf = LabeledFrame(video=video, frame_idx=0, centroids=[c_with_refs, c_no_refs])
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        tracks=[track],
        skeletons=[skeleton],
    )
    path = str(tmp_path / "test.slp")
    sleap_io.save_slp(labels, path)

    lazy = sleap_io.load_slp(path, lazy=True)
    materialized = lazy.materialize()

    assert len(materialized.centroids) == 2

    # Find centroids by coordinates (ordering may differ after round-trip)
    c_ref = [c for c in materialized.centroids if c.x == 1.0][0]
    c_noref = [c for c in materialized.centroids if c.x == 3.0][0]

    # Track references point to the NEW copies
    assert c_ref.track is materialized.tracks[0]

    # Centroid without refs stays None
    assert c_noref.track is None


def test_labels_replace_videos_updates_centroids():
    """replace_videos should update labeled frame video references."""
    old_video = Video(filename="old.mp4")
    new_video = Video(filename="new.mp4")
    c = UserCentroid(x=1.0, y=2.0)
    lf = LabeledFrame(video=old_video, frame_idx=0, centroids=[c])
    labels = Labels(labeled_frames=[lf], videos=[old_video])
    labels.replace_videos(old_videos=[old_video], new_videos=[new_video])
    assert lf.video is new_video


def test_labels_replace_videos_updates_label_images():
    """replace_videos should update labeled frame video references."""
    old_video = Video(filename="old.mp4")
    new_video = Video(filename="new.mp4")
    other_video = Video(filename="other.mp4")
    li_old = UserLabelImage(data=np.zeros((4, 4), dtype=np.int32))
    li_other = UserLabelImage(data=np.zeros((4, 4), dtype=np.int32))
    lf_old = LabeledFrame(video=old_video, frame_idx=0, label_images=[li_old])
    lf_other = LabeledFrame(video=other_video, frame_idx=0, label_images=[li_other])
    labels = Labels(labeled_frames=[lf_old, lf_other], videos=[old_video, other_video])
    labels.replace_videos(old_videos=[old_video], new_videos=[new_video])
    assert lf_old.video is new_video
    assert lf_other.video is other_video  # Unchanged — not in video_map


def test_labels_copy_lazy_preserves_centroids(tmp_path):
    """Lazy copy should preserve centroids (regression test for missing kwarg)."""
    old_video = Video(filename="test.mp4")
    c = UserCentroid(x=1.0, y=2.0)
    lf = LabeledFrame(video=old_video, frame_idx=0, centroids=[c])
    labels = Labels(labeled_frames=[lf], videos=[old_video])

    # Save and reload as lazy to get a lazy Labels
    path = str(tmp_path / "centroids.slp")
    save_slp(labels, path)
    lazy_labels = load_slp(path, open_videos=False)

    assert len(lazy_labels.centroids) == 1

    # Exercise the lazy copy path
    labels_copy = lazy_labels.copy()
    assert len(labels_copy.centroids) == 1
    assert labels_copy.centroids[0].x == 1.0
    assert labels_copy.centroids[0].y == 2.0


def test_labels_distribute_annotations():
    """Annotations are placed directly on LabeledFrames."""
    video = Video(filename="test.mp4", open_backend=False)
    c1 = UserCentroid(x=1.0, y=2.0)
    c2 = UserCentroid(x=3.0, y=4.0)
    b1 = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)

    lf0 = LabeledFrame(video=video, frame_idx=0, centroids=[c1], bboxes=[b1])
    lf1 = LabeledFrame(video=video, frame_idx=1, centroids=[c2])
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video])

    # Two frames exist
    assert len(labels.labeled_frames) == 2

    # Centroids on correct frames
    assert len(lf0.centroids) == 1
    assert lf0.centroids[0] is c1
    assert len(lf1.centroids) == 1
    assert lf1.centroids[0] is c2

    # Bbox on frame 0
    assert len(lf0.bboxes) == 1
    assert lf0.bboxes[0] is b1

    # Property returns flat view
    assert len(labels.centroids) == 2
    assert len(labels.bboxes) == 1


def test_labels_distribute_to_existing_frames():
    """Annotations can be placed on pre-existing LabeledFrames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton)
    c = UserCentroid(x=1.0, y=2.0)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst], centroids=[c])

    labels = Labels(labeled_frames=[lf])

    # Centroid on the existing frame
    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].centroids) == 1
    assert labels.labeled_frames[0].centroids[0] is c
    assert len(labels.labeled_frames[0].instances) == 1


def test_labels_add_centroid():
    """Centroids can be added to frames via _find_or_create_frame."""
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")
    c = UserCentroid(x=1.0, y=2.0, track=track)

    labels = Labels(videos=[video])
    lf = labels._find_or_create_frame(video, 0)
    lf.centroids.append(c)
    labels._invalidate_indices()
    labels.update()

    assert len(labels.labeled_frames) == 1
    assert labels.labeled_frames[0].centroids[0] is c
    assert track in labels.tracks

    # Adding to same frame reuses it
    c2 = UserCentroid(x=3.0, y=4.0)
    lf = labels._find_or_create_frame(video, 0)
    lf.centroids.append(c2)
    assert len(labels.labeled_frames) == 1
    assert len(labels.labeled_frames[0].centroids) == 2


def test_labels_add_centroid_requires_video_and_frame():
    """Centroids are added to LabeledFrames which require video and frame_idx."""
    video = Video(filename="test.mp4", open_backend=False)
    labels = Labels(videos=[video])
    c = UserCentroid(x=1.0, y=2.0)
    # LabeledFrame requires video and frame_idx
    lf = labels._find_or_create_frame(video, 0)
    lf.centroids.append(c)
    assert len(lf.centroids) == 1


def test_labels_centroids_property_flat_view():
    """labels.centroids returns flat list across all frames."""
    video = Video(filename="test.mp4", open_backend=False)
    c1 = UserCentroid(x=1.0, y=2.0)
    c2 = UserCentroid(x=3.0, y=4.0)
    c3 = UserCentroid(x=5.0, y=6.0)

    lf0 = LabeledFrame(video=video, frame_idx=0, centroids=[c1, c3])
    lf1 = LabeledFrame(video=video, frame_idx=1, centroids=[c2])
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video])

    flat = labels.centroids
    assert len(flat) == 3
    assert c1 in flat
    assert c2 in flat
    assert c3 in flat


def test_labels_multi_video_distribution():
    """Annotations from different videos go to different frames."""
    v1 = Video(filename="v1.mp4", open_backend=False)
    v2 = Video(filename="v2.mp4", open_backend=False)
    c1 = UserCentroid(x=1.0, y=2.0)
    c2 = UserCentroid(x=3.0, y=4.0)

    lf_v1 = LabeledFrame(video=v1, frame_idx=0, centroids=[c1])
    lf_v2 = LabeledFrame(video=v2, frame_idx=0, centroids=[c2])
    labels = Labels(labeled_frames=[lf_v1, lf_v2], videos=[v1, v2])

    assert len(labels.labeled_frames) == 2
    assert len(lf_v1.centroids) == 1
    assert lf_v1.centroids[0] is c1
    assert len(lf_v2.centroids) == 1
    assert lf_v2.centroids[0] is c2


def test_labels_update_collects_annotation_tracks():
    """update() collects tracks from nested annotations."""
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")
    c = UserCentroid(x=1.0, y=2.0, track=track)

    lf = LabeledFrame(video=video, frame_idx=0, centroids=[c])
    labels = Labels(labeled_frames=[lf], videos=[video])

    assert track in labels.tracks


def test_labels_slp_roundtrip_with_annotations(tmp_path):
    """Save and reload Labels with annotations on frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track = Track(name="t1")
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton, track=track)
    c = UserCentroid(x=1.0, y=2.0, track=track)
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)
    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[inst], centroids=[c], bboxes=[b]
    )

    labels = Labels(labeled_frames=[lf])

    path = str(tmp_path / "test.slp")
    save_slp(labels, path)
    loaded = load_slp(path, open_videos=False)

    assert len(loaded.centroids) == 1
    assert loaded.centroids[0].x == 1.0
    assert len(loaded.bboxes) == 1

    # Annotations are on the correct frame
    lf0 = loaded.labeled_frames[0]
    assert len(lf0.centroids) == 1
    assert len(lf0.bboxes) == 1
    assert len(lf0.instances) == 1


def test_labels_centroid_only_roundtrip(tmp_path):
    """Centroid-only data (e.g., TrackMate) round-trips through SLP."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track = Track(name="t1")

    lfs = []
    for i in range(5):
        c = PredictedCentroid(x=float(i), y=float(i * 2), track=track, score=0.9)
        lfs.append(LabeledFrame(video=video, frame_idx=i, centroids=[c]))

    labels = Labels(
        labeled_frames=lfs, videos=[video], skeletons=[skeleton], tracks=[track]
    )
    assert len(labels.labeled_frames) == 5

    path = str(tmp_path / "centroids.slp")
    save_slp(labels, path)
    loaded = load_slp(path, open_videos=False)

    assert len(loaded.centroids) == 5
    assert len(loaded.labeled_frames) == 5
    for i, c in enumerate(sorted(loaded.centroids, key=lambda c: c.x)):
        assert c.x == float(i)
        assert c.y == float(i * 2)


def test_labels_add_bbox_auto_populates():
    """Adding bboxes to frames auto-populates videos and tracks lists."""
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10, track=track)

    lf = LabeledFrame(video=video, frame_idx=0, bboxes=[b])
    labels = Labels(labeled_frames=[lf])

    assert video in labels.videos
    assert track in labels.tracks
    assert len(labels.labeled_frames) == 1
    assert labels.labeled_frames[0].bboxes[0] is b


def test_labels_add_label_image_collects_tracks():
    """Adding label images to frames collects tracks."""
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")
    li = UserLabelImage(
        data=np.array([[0, 1], [2, 0]], dtype=np.int32),
        objects={1: LabelImage.Info(track=track, category="neuron")},
    )

    lf = LabeledFrame(video=video, frame_idx=0, label_images=[li])
    labels = Labels(labeled_frames=[lf])

    assert track in labels.tracks
    assert len(labels.labeled_frames) == 1


def test_labels_collect_annotation_tracks_on_append():
    """Appending frames with annotation tracks collects them."""
    video = Video(filename="test.mp4", open_backend=False)
    t_bbox = Track(name="t_bbox")
    t_mask = Track(name="t_mask")
    t_roi = Track(name="t_roi")

    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10, track=t_bbox)
    mask_data = np.zeros((10, 10), dtype=bool)
    mask_data[2:8, 2:8] = True
    m = UserSegmentationMask.from_numpy(mask_data, track=t_mask)
    r = UserROI.from_bbox(0, 0, 10, 10, video=video)
    r.track = t_roi

    lf = LabeledFrame(video=video, frame_idx=0, bboxes=[b], masks=[m], rois=[r])
    labels = Labels(videos=[video])
    labels.append(lf)

    assert t_bbox in labels.tracks
    assert t_mask in labels.tracks
    assert t_roi in labels.tracks


def test_labels_materialize_with_annotations(tmp_path):
    """materialize() relinks annotations on frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track = Track(name="t1")
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton, track=track)
    c = UserCentroid(x=1.0, y=2.0, track=track, instance=inst)
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10, track=track)
    li = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=track)},
    )

    lf = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[inst],
        centroids=[c],
        bboxes=[b],
        label_images=[li],
    )
    labels = Labels(labeled_frames=[lf])

    path = str(tmp_path / "test.slp")
    save_slp(labels, path)
    lazy = load_slp(path, lazy=True)

    # Verify lazy property returns annotations without materializing
    assert len(lazy.centroids) >= 1
    assert len(lazy.bboxes) >= 1

    materialized = lazy.materialize()

    # Annotations are on frames with relinked references
    assert len(materialized.centroids) >= 1
    mat_c = [c for c in materialized.centroids if c.x == 1.0][0]
    assert mat_c.track is materialized.tracks[0]

    # Label image track references relinked
    assert len(materialized.label_images) >= 1
    mat_li = materialized.label_images[0]
    for info in mat_li.objects.values():
        if info.track is not None:
            assert info.track is materialized.tracks[0]


def test_labels_lazy_copy_supplementary_frames(tmp_path):
    """Lazy copy preserves supplementary (annotation-only) frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track = Track(name="t1")

    # Create centroid-only labels (no instances — will create annotation-only frames)
    lfs = []
    for i in range(3):
        c = PredictedCentroid(x=float(i), y=float(i), track=track, score=0.9)
        lfs.append(LabeledFrame(video=video, frame_idx=i, centroids=[c]))
    labels = Labels(
        labeled_frames=lfs,
        videos=[video],
        skeletons=[skeleton],
        tracks=[track],
    )

    path = str(tmp_path / "test.slp")
    save_slp(labels, path)
    lazy = load_slp(path, lazy=True)

    # Access all frames (includes supplementary)
    n = len(lazy.labeled_frames)
    assert n >= 3

    # Copy preserves everything
    copied = lazy.copy()
    assert len(copied.centroids) == 3
    assert len(copied.labeled_frames) >= 3

    # Supplementary frames accessible by index
    for i in range(len(lazy.labeled_frames)):
        lf = lazy.labeled_frames[i]
        assert lf.video is not None


def test_labels_find_or_create_frame_reuse():
    """_find_or_create_frame reuses existing frames."""
    video = Video(filename="test.mp4", open_backend=False)
    labels = Labels(
        labeled_frames=[LabeledFrame(video=video, frame_idx=0)],
        videos=[video],
    )

    # Should find existing frame
    lf = labels._find_or_create_frame(video, 0)
    assert lf is labels.labeled_frames[0]
    assert len(labels.labeled_frames) == 1

    # Should create new frame
    lf2 = labels._find_or_create_frame(video, 1)
    assert len(labels.labeled_frames) == 2
    assert lf2.frame_idx == 1


def test_labels_materialize_annotations_no_track(tmp_path):
    """materialize() handles annotations without track references."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton)
    # Centroids without track
    c = UserCentroid(x=1.0, y=2.0)
    li = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(category="cell")},
    )

    lf = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[inst],
        centroids=[c],
        label_images=[li],
    )
    labels = Labels(labeled_frames=[lf])

    path = str(tmp_path / "test.slp")
    save_slp(labels, path)
    lazy = load_slp(path, lazy=True)
    materialized = lazy.materialize()

    # Centroid without track still works
    mat_c = [x for x in materialized.centroids if x.x == 1.0][0]
    assert mat_c.track is None

    # Label image without track still works
    assert len(materialized.label_images) >= 1
    mat_li = materialized.label_images[0]
    for info in mat_li.objects.values():
        assert info.track is None


def test_labels_materialize_undistributed_label_images(tmp_path):
    """materialize() preserves label images on frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])

    li = UserLabelImage(
        data=np.array([[1, 2]], dtype=np.int32),
        objects={1: LabelImage.Info(category="bg")},
    )

    lf = LabeledFrame(video=video, frame_idx=0, label_images=[li])
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])
    path = str(tmp_path / "test.slp")
    save_slp(labels, path)

    lazy = load_slp(path, lazy=True)
    materialized = lazy.materialize()

    # Label image survives round-trip
    assert len(materialized.label_images) >= 1


def test_frame_index_build_and_lookup():
    """Frame index builds on demand and provides O(1) lookups."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    lf0 = LabeledFrame(video=video, frame_idx=0)
    lf5 = LabeledFrame(video=video, frame_idx=5)
    labels = Labels(labeled_frames=[lf0, lf5], videos=[video], skeletons=[skeleton])

    # get_frame returns correct frame by identity
    assert labels.get_frame(video, 0) is lf0
    assert labels.get_frame(video, 5) is lf5
    assert labels.get_frame(video, 99) is None


def test_frame_index_multi_video():
    """Frame index correctly separates frames from different videos."""
    v1 = Video(filename="v1.mp4", open_backend=False)
    v2 = Video(filename="v2.mp4", open_backend=False)
    lf_v1 = LabeledFrame(video=v1, frame_idx=0)
    lf_v2 = LabeledFrame(video=v2, frame_idx=0)
    labels = Labels(labeled_frames=[lf_v1, lf_v2], videos=[v1, v2])

    assert labels.get_frame(v1, 0) is lf_v1
    assert labels.get_frame(v2, 0) is lf_v2
    assert labels.get_frame(v1, 0) is not labels.get_frame(v2, 0)


def test_frame_index_staleness():
    """Frame index auto-rebuilds when labeled_frames length changes."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    lf0 = LabeledFrame(video=video, frame_idx=0)
    labels = Labels(labeled_frames=[lf0], videos=[video], skeletons=[skeleton])

    assert labels.get_frame(video, 0) is lf0
    assert labels.get_frame(video, 1) is None

    # Append a new frame — index should auto-rebuild
    lf1 = LabeledFrame(video=video, frame_idx=1)
    labels.labeled_frames.append(lf1)
    assert labels.get_frame(video, 1) is lf1


def test_frame_index_duplicate_warning():
    """Duplicate (video, frame_idx) triggers a warning."""
    video = Video(filename="test.mp4", open_backend=False)
    lf0a = LabeledFrame(video=video, frame_idx=0)
    lf0b = LabeledFrame(video=video, frame_idx=0)
    labels = Labels(labeled_frames=[lf0a, lf0b], videos=[video])

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        labels._ensure_frame_index()
        assert len(w) == 1
        assert "Duplicate" in str(w[0].message)

    # Last-wins semantics
    assert labels.get_frame(video, 0) is lf0b


def test_track_index_build_and_lookup():
    """Track index provides O(1) lookup by (video, track)."""
    video = Video(filename="test.mp4", open_backend=False)
    t1 = Track(name="t1")
    t2 = Track(name="t2")
    c1 = UserCentroid(x=1.0, y=2.0, track=t1)
    c2 = UserCentroid(x=3.0, y=4.0, track=t1)
    c3 = UserCentroid(x=5.0, y=6.0, track=t2)

    lf0 = LabeledFrame(video=video, frame_idx=0, centroids=[c1])
    lf2 = LabeledFrame(video=video, frame_idx=2, centroids=[c3])
    lf5 = LabeledFrame(video=video, frame_idx=5, centroids=[c2])
    labels = Labels(labeled_frames=[lf0, lf2, lf5], videos=[video], tracks=[t1, t2])

    # Track 1 has 2 annotations, sorted by frame_idx
    t1_anns = labels.get_track_annotations(video, t1)
    assert len(t1_anns) == 2
    assert t1_anns[0] is c1  # frame_idx=0
    assert t1_anns[1] is c2  # frame_idx=5

    # Track 2 has 1 annotation
    t2_anns = labels.get_track_annotations(video, t2)
    assert len(t2_anns) == 1
    assert t2_anns[0] is c3

    # Unknown track returns empty list
    t3 = Track(name="t3")
    assert labels.get_track_annotations(video, t3) == []


def test_track_index_includes_instances():
    """Track index includes instances, not just annotations."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track = Track(name="t1")
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton, track=track)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels(labeled_frames=[lf], videos=[video], tracks=[track])

    anns = labels.get_track_annotations(video, track)
    assert len(anns) == 1
    assert anns[0] is inst


def test_reindex():
    """reindex() forces index rebuild on next access."""
    video = Video(filename="test.mp4", open_backend=False)
    lf = LabeledFrame(video=video, frame_idx=0)
    labels = Labels(labeled_frames=[lf], videos=[video])

    # Build index
    labels.get_frame(video, 0)
    assert labels._frame_index is not None

    # Reindex clears it
    labels.reindex()
    assert labels._frame_index is None
    assert labels._track_index is None

    # Rebuilds on next access
    assert labels.get_frame(video, 0) is lf


def test_find_uses_frame_index():
    """find() uses frame index for O(1) eager lookups."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    frames = [LabeledFrame(video=video, frame_idx=i) for i in range(10)]
    labels = Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton])

    # Single frame lookup
    result = labels.find(video, 5)
    assert len(result) == 1
    assert result[0] is frames[5]

    # Multiple frame lookup
    result = labels.find(video, [2, 7])
    assert len(result) == 2
    assert result[0] is frames[2]
    assert result[1] is frames[7]

    # Missing frame with return_new
    result = labels.find(video, 99, return_new=True)
    assert len(result) == 1
    assert result[0].frame_idx == 99

    # Missing frame without return_new
    result = labels.find(video, 99)
    assert len(result) == 0


def test_get_centroids_fast_path():
    """get_centroids uses O(1) frame lookup when video+frame_idx given."""
    video = Video(filename="test.mp4", open_backend=False)
    c0 = UserCentroid(x=1.0, y=2.0)
    c1 = UserCentroid(x=3.0, y=4.0)
    lf0 = LabeledFrame(video=video, frame_idx=0, centroids=[c0])
    lf1 = LabeledFrame(video=video, frame_idx=1, centroids=[c1])
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video])

    # Fast path: both video and frame_idx
    result = labels.get_centroids(video=video, frame_idx=0)
    assert len(result) == 1
    assert result[0] is c0

    # Video only: iterates frames for that video
    result = labels.get_centroids(video=video)
    assert len(result) == 2

    # Frame only: uses property
    result = labels.get_centroids(frame_idx=0)
    assert len(result) == 1

    # No match
    result = labels.get_centroids(video=video, frame_idx=99)
    assert len(result) == 0


def test_replace_videos_reindexes():
    """replace_videos() invalidates indices after updating references."""
    old_video = Video(filename="old.mp4", open_backend=False)
    new_video = Video(filename="new.mp4", open_backend=False)
    lf = LabeledFrame(video=old_video, frame_idx=0)
    labels = Labels(labeled_frames=[lf], videos=[old_video])

    # Build index with old video
    assert labels.get_frame(old_video, 0) is lf

    # Replace video — index should be invalidated
    labels.replace_videos(old_videos=[old_video], new_videos=[new_video])

    # Old video no longer in index
    assert labels.get_frame(old_video, 0) is None

    # New video found via rebuilt index
    assert labels.get_frame(new_video, 0) is lf


def test_track_index_includes_label_images():
    """Track index includes label images via their objects' tracks."""
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")
    li = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=track, category="cell")},
    )
    lf = LabeledFrame(video=video, frame_idx=0, label_images=[li])
    labels = Labels(labeled_frames=[lf], videos=[video], tracks=[track])

    anns = labels.get_track_annotations(video, track)
    assert len(anns) == 1
    assert anns[0] is li


def test_get_bboxes_fast_path():
    """get_bboxes uses O(1) frame lookup and filters by track/instance."""
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton)
    b1 = UserBoundingBox(x1=0, y1=0, x2=10, y2=10, track=track, instance=inst)
    b2 = UserBoundingBox(x1=5, y1=5, x2=15, y2=15)
    lf = LabeledFrame(video=video, frame_idx=0, bboxes=[b1, b2])
    labels = Labels(labeled_frames=[lf], videos=[video])

    # Fast path with video+frame_idx
    result = labels.get_bboxes(video=video, frame_idx=0)
    assert len(result) == 2

    # Filter by track
    result = labels.get_bboxes(video=video, frame_idx=0, track=track)
    assert len(result) == 1
    assert result[0] is b1

    # Filter by instance
    result = labels.get_bboxes(video=video, frame_idx=0, instance=inst)
    assert len(result) == 1
    assert result[0] is b1

    # No match
    result = labels.get_bboxes(video=video, frame_idx=99)
    assert len(result) == 0


def test_get_masks_fast_path():
    """get_masks uses O(1) frame lookup when video+frame_idx provided."""
    video = Video(filename="test.mp4", open_backend=False)
    mask_data = np.zeros((10, 10), dtype=bool)
    mask_data[2:8, 2:8] = True
    m = UserSegmentationMask.from_numpy(mask_data)
    lf = LabeledFrame(video=video, frame_idx=0, masks=[m])
    labels = Labels(labeled_frames=[lf], videos=[video])

    result = labels.get_masks(video=video, frame_idx=0)
    assert len(result) == 1
    assert result[0] is m

    result = labels.get_masks(video=video, frame_idx=99)
    assert len(result) == 0


def test_get_label_images_fast_path():
    """get_label_images uses O(1) frame lookup when video+frame_idx given."""
    video = Video(filename="test.mp4", open_backend=False)
    li = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(category="cell")},
    )
    lf = LabeledFrame(video=video, frame_idx=0, label_images=[li])
    labels = Labels(labeled_frames=[lf], videos=[video])

    result = labels.get_label_images(video=video, frame_idx=0)
    assert len(result) == 1
    assert result[0] is li

    result = labels.get_label_images(video=video, frame_idx=99)
    assert len(result) == 0


def test_append_invalidates_indices():
    """append() invalidates cached indices."""
    video = Video(filename="test.mp4", open_backend=False)
    lf0 = LabeledFrame(video=video, frame_idx=0)
    labels = Labels(labeled_frames=[lf0], videos=[video])

    # Build index
    assert labels.get_frame(video, 0) is lf0
    assert labels._frame_index is not None

    # Append new frame — index should be invalidated
    lf1 = LabeledFrame(video=video, frame_idx=1)
    labels.append(lf1)
    assert labels._frame_index is None

    # Rebuilt on next access
    assert labels.get_frame(video, 1) is lf1


def test_extend_invalidates_indices():
    """extend() invalidates cached indices."""
    video = Video(filename="test.mp4", open_backend=False)
    lf0 = LabeledFrame(video=video, frame_idx=0)
    labels = Labels(labeled_frames=[lf0], videos=[video])

    # Build index
    assert labels.get_frame(video, 0) is lf0

    # Extend — index should be invalidated
    lf1 = LabeledFrame(video=video, frame_idx=1)
    lf2 = LabeledFrame(video=video, frame_idx=2)
    labels.extend([lf1, lf2])
    assert labels._frame_index is None

    # Rebuilt on next access
    assert labels.get_frame(video, 2) is lf2


def test_clean_invalidates_indices():
    """clean() invalidates cached indices."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton)
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    lf1 = LabeledFrame(video=video, frame_idx=1)  # Empty, will be removed
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video], skeletons=[skeleton])

    # Build index
    assert labels.get_frame(video, 1) is lf1

    # Clean removes empty frame and invalidates index
    labels.clean()
    assert labels._frame_index is None
    assert labels.get_frame(video, 1) is None
    assert labels.get_frame(video, 0) is lf0


def test_clean_preserves_frames_with_any_annotations():
    """clean() preserves frames with any annotation type."""
    video = Video(filename="test.mp4", open_backend=False)
    c = UserCentroid(x=1.0, y=2.0)
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)
    mask_data = np.zeros((10, 10), dtype=bool)
    mask_data[2:8, 2:8] = True
    m = UserSegmentationMask.from_numpy(mask_data)
    roi = UserROI(geometry=box(0, 0, 10, 10), video=video)
    li = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(category="cell")},
    )

    lf0 = LabeledFrame(video=video, frame_idx=0, centroids=[c])
    lf1 = LabeledFrame(video=video, frame_idx=1, bboxes=[b])
    lf2 = LabeledFrame(video=video, frame_idx=2, masks=[m])
    lf3 = LabeledFrame(video=video, frame_idx=3, rois=[roi])
    lf4 = LabeledFrame(video=video, frame_idx=4, label_images=[li])
    labels = Labels(
        labeled_frames=[lf0, lf1, lf2, lf3, lf4],
        videos=[video],
    )

    # All 5 frames exist (one per annotation type), none have instances
    assert len(labels.labeled_frames) == 5

    # Clean should preserve all — each has at least one annotation
    labels.clean()
    assert len(labels.labeled_frames) == 5

    # Verify each frame's annotations survived
    for lf in labels.labeled_frames:
        has_any = lf.centroids or lf.bboxes or lf.masks or lf.label_images or lf.rois
        assert has_any, f"Frame {lf.frame_idx} lost its annotations"


def test_clean_removes_truly_empty_frames_only():
    """clean() removes empty frames but keeps annotation-only and instance frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton)

    # Frame 0: has instance
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    # Frame 1: has centroid only
    c = UserCentroid(x=1.0, y=2.0)
    lf1 = LabeledFrame(video=video, frame_idx=1, centroids=[c])
    # Frame 2: truly empty
    lf2 = LabeledFrame(video=video, frame_idx=2)

    labels = Labels(
        labeled_frames=[lf0, lf1, lf2], videos=[video], skeletons=[skeleton]
    )
    assert len(labels.labeled_frames) == 3

    labels.clean()

    # Only the truly empty frame (2) should be removed
    assert len(labels.labeled_frames) == 2
    frame_idxs = {lf.frame_idx for lf in labels.labeled_frames}
    assert frame_idxs == {0, 1}


def test_get_rois_fast_path():
    """get_rois uses O(1) frame lookup when video+frame_idx provided."""
    video = Video(filename="test.mp4", open_backend=False)
    roi1 = UserROI(geometry=box(0, 0, 10, 10), video=video)
    roi2 = UserROI(geometry=box(5, 5, 15, 15), video=video)
    lf0 = LabeledFrame(video=video, frame_idx=0, rois=[roi1])
    lf1 = LabeledFrame(video=video, frame_idx=1, rois=[roi2])
    labels = Labels(labeled_frames=[lf0, lf1], videos=[video])

    # Fast path with video+frame_idx
    result = labels.get_rois(video=video, frame_idx=0)
    assert len(result) == 1
    assert result[0] is roi1

    # Video only: iterates frames for that video
    result = labels.get_rois(video=video)
    assert len(result) == 2

    # No match
    result = labels.get_rois(video=video, frame_idx=99)
    assert len(result) == 0


def test_merge_copies_annotations_new_frame():
    """merge() copies annotations when creating new frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track = Track(name="t1")

    # Self labels: frame 0 with instance
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skeleton)
    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    self_labels = Labels(
        labeled_frames=[lf0], videos=[video], skeletons=[skeleton], tracks=[track]
    )

    # Other labels: frame 1 with centroid + bbox (not in self)
    c = UserCentroid(x=5.0, y=10.0, track=track)
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)
    lf1 = LabeledFrame(video=video, frame_idx=1, centroids=[c], bboxes=[b])
    other_labels = Labels(
        labeled_frames=[lf1], videos=[video], skeletons=[skeleton], tracks=[track]
    )

    result = self_labels.merge(other_labels)
    assert result.frames_merged == 1

    # New frame should have been created with annotations
    merged_lf = self_labels.find(video, 1)
    assert len(merged_lf) == 1
    assert len(merged_lf[0].centroids) == 1
    assert len(merged_lf[0].bboxes) == 1


def test_merge_remaps_annotation_tracks():
    """merge() remaps track references on annotations."""
    video = Video(filename="v.mp4", open_backend=False)
    skeleton = Skeleton(["A"])

    # Two separate track objects with the same name (will be matched)
    track_self = Track(name="animal")
    track_other = Track(name="animal")

    # Self labels: empty
    self_labels = Labels(videos=[video], skeletons=[skeleton], tracks=[track_self])

    # Other labels: centroid referencing track_other
    c = UserCentroid(x=1.0, y=2.0, track=track_other)
    lf = LabeledFrame(video=video, frame_idx=0, centroids=[c])
    other_labels = Labels(
        labeled_frames=[lf], videos=[video], skeletons=[skeleton], tracks=[track_other]
    )

    # track="name" so the two distinct same-named Track objects coalesce and the
    # annotation track is remapped (the identity default would keep them separate).
    self_labels.merge(other_labels, track="name")

    # The centroid's track should now reference track_self (the matched track)
    merged_lf = self_labels.find(video, 0)[0]
    assert len(merged_lf.centroids) == 1
    assert merged_lf.centroids[0].track is track_self


def test_merge_remaps_annotations_existing_frame():
    """merge() remaps annotation tracks when merging into existing frame.

    Also covers the False branch of video remapping: self_frame already has
    a centroid with video_self, which is NOT a key in video_map. Only the
    incoming centroid (with video_other) should have its video remapped.
    """
    video_self = Video(filename="v.mp4", open_backend=False)
    video_other = Video(filename="v.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track_self = Track(name="animal")
    track_other = Track(name="animal")

    # Self labels: frame 0 with instance + pre-existing centroid
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skeleton)
    c_self = UserCentroid(x=0.0, y=0.0, track=track_self)
    lf0 = LabeledFrame(
        video=video_self, frame_idx=0, instances=[inst], centroids=[c_self]
    )
    self_labels = Labels(
        labeled_frames=[lf0],
        videos=[video_self],
        skeletons=[skeleton],
        tracks=[track_self],
    )

    # Other labels: same frame with centroid referencing different video/track
    c_other = UserCentroid(x=5.0, y=10.0, track=track_other)
    other_lf = LabeledFrame(video=video_other, frame_idx=0, centroids=[c_other])
    other_labels = Labels(
        labeled_frames=[other_lf],
        videos=[video_other],
        skeletons=[skeleton],
        tracks=[track_other],
    )

    # track="name" so the two distinct same-named Track objects coalesce and the
    # incoming centroid's track is remapped (the identity default would not).
    self_labels.merge(other_labels, frame="keep_both", track="name")

    # Both centroids should be on the frame
    assert len(lf0.centroids) == 2
    # Self's centroid: track unchanged
    assert c_self.track is track_self
    # Other's centroid: track remapped to track_self
    merged_other = [c for c in lf0.centroids if c is not c_self][0]
    assert merged_other.track is track_self


def test_merge_remaps_label_image_tracks():
    """merge() remaps track references in label_image objects."""
    video = Video(filename="v.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track_self = Track(name="cell")
    track_other = Track(name="cell")

    self_labels = Labels(videos=[video], skeletons=[skeleton], tracks=[track_self])

    li = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=track_other, category="cell")},
    )
    lf = LabeledFrame(video=video, frame_idx=0, label_images=[li])
    other_labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_other],
    )

    # track="name" so the two distinct same-named Track objects coalesce and the
    # label_image track reference is remapped (the identity default would not).
    self_labels.merge(other_labels, track="name")

    merged_lf = self_labels.find(video, 0)[0]
    assert len(merged_lf.label_images) == 1
    assert merged_lf.label_images[0].objects[1].track is track_self


def test_merge_default_keeps_same_named_distinct_tracks_separate():
    """Default merge (identity) keeps distinct same-named tracks as separate tracks.

    This locks in the correctness-first identity default: two independently loaded
    files often have positionally-named tracks (e.g. "track_0") that are *different*
    animals, so the default must NOT collapse them by name.
    """
    skeleton = Skeleton(["A"])
    video = Video(filename="v.mp4", open_backend=False)

    track_self = Track(name="track_0")
    track_other = Track(name="track_0")  # Same name, distinct object.

    inst_self = Instance.from_numpy(
        np.array([[1.0, 2.0]]), skeleton=skeleton, track=track_self
    )
    self_labels = Labels(
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, instances=[inst_self])],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_self],
    )

    inst_other = Instance.from_numpy(
        np.array([[3.0, 4.0]]), skeleton=skeleton, track=track_other
    )
    other_labels = Labels(
        labeled_frames=[LabeledFrame(video=video, frame_idx=1, instances=[inst_other])],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_other],
    )

    self_labels.merge(other_labels)

    # No silent collapse: both distinct tracks survive as the original objects.
    assert len(self_labels.tracks) == 2
    assert self_labels.tracks[0] is track_self
    assert self_labels.tracks[1] is track_other
    # The incoming instance keeps its own (distinct) track; nothing was rebound.
    assert inst_other.track is track_other


def test_merge_track_name_opt_in_collapses_and_warns():
    """track="name" still collapses same-named tracks and still warns on divergence.

    Mirrors the divergence-warning fixture: two distinct ``track_0`` tracks whose
    instances are spatially far apart. The explicit name opt-in must (a) coalesce
    them into a single track and (b) emit the spatial-divergence warning.
    """
    skeleton = Skeleton(["head", "tail"])
    video = Video(filename="v.mp4", open_backend=False)

    track_self = Track(name="track_0")
    track_other = Track(name="track_0")

    inst_self = Instance.from_numpy(
        np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, track=track_self
    )
    self_labels = Labels(
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, instances=[inst_self])],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_self],
    )

    inst_other = Instance.from_numpy(
        np.array([[500.0, 500.0], [510.0, 510.0]]), skeleton=skeleton, track=track_other
    )
    other_labels = Labels(
        labeled_frames=[LabeledFrame(video=video, frame_idx=0, instances=[inst_other])],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_other],
    )

    with pytest.warns(UserWarning, match=r"track_0.*diverge spatially"):
        self_labels.merge(other_labels, frame="keep_both", track="name")

    # Opt-in name matching collapses the two same-named tracks into one.
    assert len(self_labels.tracks) == 1


def test_match_default_keeps_same_named_distinct_tracks_unmatched():
    """match() default (identity) does not match distinct same-named tracks.

    Locks match()/merge() consistency: both resolve ``track=None`` to a bare
    ``TrackMatcher()`` and therefore share the identity default. ``track="name"``
    is the opt-in that matches by name.
    """
    skeleton = Skeleton(["A"])
    video = Video(filename="v.mp4", open_backend=False)

    track_gt = Track(name="track_0")
    track_pred = Track(name="track_0")  # Same name, distinct object.

    gt_labels = Labels(videos=[video], skeletons=[skeleton], tracks=[track_gt])
    pred_labels = Labels(videos=[video], skeletons=[skeleton], tracks=[track_pred])

    # Default identity: distinct objects do not match (track_map maps to None).
    result = gt_labels.match(pred_labels)
    assert result.track_map[track_pred] is not track_gt
    assert result.track_map[track_pred] is None

    # Opt-in name matching: they match.
    result_named = gt_labels.match(pred_labels, track="name")
    assert result_named.track_map[track_pred] is track_gt


def test_clean_removes_orphaned_annotation_tracks():
    """clean() removes annotations whose tracks are no longer in self.tracks."""
    video = Video(filename="test.mp4", open_backend=False)
    track_keep = Track(name="keep")
    track_remove = Track(name="remove")
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(
        np.array([[1.0, 2.0]]), skeleton=skeleton, track=track_keep
    )

    c_keep = UserCentroid(x=1.0, y=2.0, track=track_keep)
    c_remove = UserCentroid(x=3.0, y=4.0, track=track_remove)
    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[inst], centroids=[c_keep, c_remove]
    )

    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_keep, track_remove],
    )

    # Remove track_remove externally, then clean
    labels.tracks.remove(track_remove)
    labels.clean()

    # Only the centroid with track_keep should remain
    assert len(lf.centroids) == 1
    assert lf.centroids[0].track is track_keep


def test_clean_preserves_trackless_annotations():
    """clean() preserves annotations without tracks even during track cleanup."""
    video = Video(filename="test.mp4", open_backend=False)
    c_no_track = UserCentroid(x=1.0, y=2.0)
    lf = LabeledFrame(video=video, frame_idx=0, centroids=[c_no_track])
    labels = Labels(labeled_frames=[lf], videos=[video])

    labels.clean()
    assert len(lf.centroids) == 1
    assert lf.centroids[0] is c_no_track


def test_extract_preserves_annotations():
    """extract() includes annotations nested in extracted frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skeleton)
    c = UserCentroid(x=5.0, y=10.0)
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)

    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[inst], centroids=[c], bboxes=[b]
    )
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    extracted = labels.extract([0])
    assert len(extracted.labeled_frames) == 1
    ex_lf = extracted.labeled_frames[0]
    assert len(ex_lf.centroids) == 1
    assert len(ex_lf.bboxes) == 1


def test_split_preserves_annotations():
    """split() includes annotations in both splits."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])

    lfs = []
    for i in range(4):
        inst = Instance.from_numpy(np.array([[float(i), 0.0]]), skeleton=skeleton)
        c = UserCentroid(x=float(i), y=0.0)
        lf = LabeledFrame(video=video, frame_idx=i, instances=[inst], centroids=[c])
        lfs.append(lf)

    labels = Labels(labeled_frames=lfs, videos=[video], skeletons=[skeleton])

    splits = labels.split(0.5)
    for split_labels in splits.values():
        for lf in split_labels.labeled_frames:
            assert len(lf.centroids) == 1


def test_remove_predictions_clears_predicted_annotations():
    """remove_predictions() removes predicted annotations from frames."""
    video = Video(filename="test.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skeleton)
    pred_inst = PredictedInstance.from_numpy(
        np.array([[3.0, 4.0]]), skeleton=skeleton, score=0.9
    )

    user_c = UserCentroid(x=1.0, y=2.0)
    pred_c = PredictedCentroid(x=3.0, y=4.0, score=0.8)
    user_b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)
    pred_b = PredictedBoundingBox(x1=5, y1=5, x2=15, y2=15, score=0.7)

    lf = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[inst, pred_inst],
        centroids=[user_c, pred_c],
        bboxes=[user_b, pred_b],
    )
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    labels.remove_predictions(clean=False)

    assert len(lf.instances) == 1
    assert type(lf.instances[0]) is Instance
    assert len(lf.centroids) == 1
    assert type(lf.centroids[0]) is UserCentroid
    assert len(lf.bboxes) == 1
    assert type(lf.bboxes[0]) is UserBoundingBox


def test_merge_remaps_annotations_with_different_videos():
    """merge() remaps video references on annotations when videos differ.

    Covers the existing-frame path where self already has label_images, exercising
    the False branch of `li.video in video_map` (self's label_image video is NOT
    in video_map, only other's is).
    """
    video_self = Video(filename="v.mp4", open_backend=False)
    video_other = Video(filename="v.mp4", open_backend=False)
    skeleton = Skeleton(["A"])
    track = Track(name="t")
    track_other = Track(name="t")

    # Self labels: frame 0 with instance + label_image
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skeleton)
    li_self = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(track=track, category="cell")},
    )
    lf0 = LabeledFrame(
        video=video_self, frame_idx=0, instances=[inst], label_images=[li_self]
    )
    self_labels = Labels(
        labeled_frames=[lf0],
        videos=[video_self],
        skeletons=[skeleton],
        tracks=[track],
    )

    # Other labels: same frame with label_image referencing other video/track
    li_other = UserLabelImage(
        data=np.array([[0, 2]], dtype=np.int32),
        objects={2: LabelImage.Info(track=track_other, category="neuron")},
    )
    other_lf = LabeledFrame(video=video_other, frame_idx=0, label_images=[li_other])
    other_labels = Labels(
        labeled_frames=[other_lf],
        videos=[video_other],
        skeletons=[skeleton],
        tracks=[track_other],
    )

    # track="name" so the two distinct same-named Track objects coalesce and the
    # incoming label_image track is remapped (the identity default would not).
    self_labels.merge(other_labels, frame="keep_both", track="name")

    # Both label_images should be on the frame
    assert len(lf0.label_images) == 2
    # Self's label_image: track unchanged
    assert li_self.objects[1].track is track
    # Other's label_image: track remapped
    merged_li = [li for li in lf0.label_images if li is not li_self][0]
    assert merged_li.objects[2].track is track


def test_merge_annotation_trackless_and_label_images_with_different_video():
    """merge() handles trackless annotations and label_images with video remapping."""
    video_self = Video(filename="base.mp4", open_backend=False)
    video_other = Video(filename="base.mp4", open_backend=False)
    skeleton = Skeleton(["A"])

    self_labels = Labels(videos=[video_self], skeletons=[skeleton])

    # Trackless centroid + label_image with no track on info
    c = UserCentroid(x=1.0, y=2.0)
    li = UserLabelImage(
        data=np.array([[0, 1]], dtype=np.int32),
        objects={1: LabelImage.Info(category="cell")},
    )
    lf = LabeledFrame(video=video_other, frame_idx=0, centroids=[c], label_images=[li])
    other_labels = Labels(
        labeled_frames=[lf], videos=[video_other], skeletons=[skeleton]
    )

    self_labels.merge(other_labels)

    merged_lf = self_labels.find(video_self, 0)[0]
    # Trackless centroid should survive with track=None
    assert len(merged_lf.centroids) == 1
    assert merged_lf.centroids[0].track is None
    # Label image should be merged
    assert len(merged_lf.label_images) == 1
    # Label image info track should remain None
    assert merged_lf.label_images[0].objects[1].track is None


def test_clean_removes_label_image_orphaned_tracks():
    """clean() removes label_image object entries with orphaned tracks."""
    video = Video(filename="test.mp4", open_backend=False)
    track_keep = Track(name="keep")
    track_remove = Track(name="remove")
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(
        np.array([[1.0, 2.0]]), skeleton=skeleton, track=track_keep
    )

    li = UserLabelImage(
        data=np.array([[0, 1, 2]], dtype=np.int32),
        objects={
            1: LabelImage.Info(track=track_keep, category="cell"),
            2: LabelImage.Info(track=track_remove, category="cell"),
        },
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst], label_images=[li])

    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_keep, track_remove],
    )

    # Remove track_remove externally, then clean
    labels.tracks.remove(track_remove)
    labels.clean()

    # Only the object entry with track_keep should remain
    assert len(lf.label_images[0].objects) == 1
    assert 1 in lf.label_images[0].objects
    assert lf.label_images[0].objects[1].track is track_keep


def test_clean_orphaned_annotations_without_frame_removal():
    """clean(frames=False, tracks=True) removes orphaned annotations."""
    video = Video(filename="test.mp4", open_backend=False)
    track_keep = Track(name="keep")
    track_remove = Track(name="remove")
    skeleton = Skeleton(["A"])
    inst = Instance.from_numpy(
        np.array([[1.0, 2.0]]), skeleton=skeleton, track=track_keep
    )

    c_keep = UserCentroid(x=1.0, y=2.0, track=track_keep)
    c_remove = UserCentroid(x=3.0, y=4.0, track=track_remove)
    lf = LabeledFrame(
        video=video, frame_idx=0, instances=[inst], centroids=[c_keep, c_remove]
    )

    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_keep, track_remove],
    )

    # Remove track, then clean with frames=False
    labels.tracks.remove(track_remove)
    labels.clean(frames=False, tracks=True)

    assert len(lf.centroids) == 1
    assert lf.centroids[0].track is track_keep


def test_get_frame_lazy_raises(centered_pair):
    """get_frame raises RuntimeError on lazy Labels."""
    lazy = load_slp(centered_pair, lazy=True)
    video = lazy.videos[0]
    with pytest.raises(RuntimeError, match="get_frame"):
        lazy.get_frame(video, 0)


def test_get_track_annotations_lazy_raises(centered_pair):
    """get_track_annotations raises RuntimeError on lazy Labels."""
    lazy = load_slp(centered_pair, lazy=True)
    video = lazy.videos[0]
    track = lazy.tracks[0]
    with pytest.raises(RuntimeError, match="get_track_annotations"):
        lazy.get_track_annotations(video, track)


def test_add_annotation_invalidates_track_index():
    """After adding centroid to frame, get_track_annotations sees it."""
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")
    c1 = UserCentroid(x=1.0, y=2.0, track=track)
    lf = LabeledFrame(video=video, frame_idx=0, centroids=[c1])
    labels = Labels(labeled_frames=[lf], videos=[video], tracks=[track])

    # Force index build
    result = labels.get_track_annotations(video, track)
    assert len(result) == 1

    # Add a second centroid to the same frame and track
    c2 = UserCentroid(x=3.0, y=4.0, track=track)
    lf.centroids.append(c2)
    labels._invalidate_indices()

    # Index should have been invalidated; new centroid visible
    result = labels.get_track_annotations(video, track)
    assert len(result) == 2


def test_remove_predictions_no_clean_invalidates_indices():
    """After remove_predictions(clean=False), track index reflects removal."""
    skel = Skeleton(nodes=["head", "tail"])
    video = Video(filename="test.mp4", open_backend=False)
    track = Track(name="t1")

    pred = PredictedInstance.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]), skeleton=skel, track=track
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[pred])
    c = UserCentroid(x=5.0, y=6.0, track=track)
    lf.centroids.append(c)

    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skel],
        tracks=[track],
    )

    # Force index build — track index should include centroid + predicted instance
    result = labels.get_track_annotations(video, track)
    assert len(result) == 2

    # Remove predictions without cleaning
    labels.remove_predictions(clean=False)

    # Track index should be invalidated — only centroid remains
    result = labels.get_track_annotations(video, track)
    assert len(result) == 1
    assert lf.centroids[0] is c


def test_merge_does_not_corrupt_source_annotations():
    """After merging, source labels' centroids keep original video/track."""
    skel = Skeleton(nodes=["head", "tail"])

    video_a = Video(filename="a.mp4")
    video_b = Video(filename="b.mp4")
    track_b = Track(name="track_b")

    # labels_a: empty frame at idx 0
    labels_a = Labels(skeletons=[skel], videos=[video_a])
    lf_a = LabeledFrame(video=video_a, frame_idx=0, instances=[])
    labels_a.append(lf_a)

    # labels_b: frame at idx 0 with a centroid
    labels_b = Labels(skeletons=[skel], videos=[video_b], tracks=[track_b])
    c = UserCentroid(x=1.0, y=2.0, track=track_b)
    lf_b = LabeledFrame(video=video_b, frame_idx=0, instances=[])
    lf_b.centroids.append(c)
    labels_b.append(lf_b)

    # Merge b into a (videos differ, so b's frame is added as new)
    video_matcher = VideoMatcher(method=VideoMatchMethod.PATH, strict=True)
    labels_a.merge(labels_b, video=video_matcher)

    # Source centroid in labels_b must still reference the original track
    assert lf_b.centroids[0].track is track_b


def test_labels_merge_keep_original_discards_other_annotations():
    """Labels.merge with frame='keep_original' does not add other's centroids."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="shared.mp4", open_backend=False)
    track_a = Track(name="track_a")
    track_b = Track(name="track_b")

    inst_a = Instance([[0, 0], [1, 1]], skeleton=skel, track=track_a)
    inst_b = Instance([[2, 2], [3, 3]], skeleton=skel, track=track_b)
    c_a = UserCentroid(x=0.5, y=0.5, track=track_a)
    c_b = UserCentroid(x=2.5, y=2.5, track=track_b)

    labels_a = Labels(
        skeletons=[skel],
        videos=[video],
        tracks=[track_a],
        labeled_frames=[
            LabeledFrame(video=video, frame_idx=0, instances=[inst_a], centroids=[c_a])
        ],
    )
    labels_b = Labels(
        skeletons=[skel],
        videos=[video],
        tracks=[track_b],
        labeled_frames=[
            LabeledFrame(video=video, frame_idx=0, instances=[inst_b], centroids=[c_b])
        ],
    )

    labels_a.merge(labels_b, frame="keep_original")

    # Only the original centroid should remain
    merged_lf = labels_a.labeled_frames[0]
    assert len(merged_lf.centroids) == 1
    assert merged_lf.centroids[0].x == 0.5


def test_labels_merge_replace_predictions_filters_annotations():
    """Labels.merge with frame='replace_predictions' keeps user, replaces predicted."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="shared.mp4", open_backend=False)
    track = Track(name="t")

    inst_user = Instance([[0, 0], [1, 1]], skeleton=skel, track=track)
    inst_pred = PredictedInstance([[4, 4], [5, 5]], skeleton=skel, track=track)
    c_user = UserCentroid(x=0.5, y=0.5, track=track)
    c_pred_self = PredictedCentroid(x=4.5, y=4.5, track=track, score=0.9)
    c_pred_other = PredictedCentroid(x=9.0, y=9.0, track=track, score=0.7)

    labels_a = Labels(
        skeletons=[skel],
        videos=[video],
        tracks=[track],
        labeled_frames=[
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[inst_user],
                centroids=[c_user, c_pred_self],
            )
        ],
    )
    labels_b = Labels(
        skeletons=[skel],
        videos=[video],
        tracks=[track],
        labeled_frames=[
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[inst_pred],
                centroids=[c_pred_other],
            )
        ],
    )

    labels_a.merge(labels_b, frame="replace_predictions")

    merged_lf = labels_a.labeled_frames[0]
    # User centroid kept, self's predicted removed, other's predicted added
    assert len(merged_lf.centroids) == 2
    user_centroids = [c for c in merged_lf.centroids if not c.is_predicted]
    pred_centroids = [c for c in merged_lf.centroids if c.is_predicted]
    assert len(user_centroids) == 1
    assert user_centroids[0].x == 0.5
    assert len(pred_centroids) == 1
    assert pred_centroids[0].x == 9.0


def test_labels_merge_auto_annotations_spatial():
    """Labels.merge with auto adds user centroids from other when unmatched."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="shared.mp4", open_backend=False)
    track = Track(name="t")

    inst_a = Instance([[0, 0], [1, 1]], skeleton=skel, track=track)
    inst_b = Instance([[50, 50], [51, 51]], skeleton=skel, track=track)

    # Self: user instance + user centroid at (0.5, 0.5)
    c_self = UserCentroid(x=0.5, y=0.5, track=track)

    # Other: user instance far away + user centroid at (50.5, 50.5) — unmatched
    c_other = UserCentroid(x=50.5, y=50.5, track=track)

    labels_a = Labels(
        skeletons=[skel],
        videos=[video],
        tracks=[track],
        labeled_frames=[
            LabeledFrame(
                video=video, frame_idx=0, instances=[inst_a], centroids=[c_self]
            )
        ],
    )
    labels_b = Labels(
        skeletons=[skel],
        videos=[video],
        tracks=[track],
        labeled_frames=[
            LabeledFrame(
                video=video, frame_idx=0, instances=[inst_b], centroids=[c_other]
            )
        ],
    )

    labels_a.merge(labels_b)  # default frame="auto"

    merged_lf = labels_a.labeled_frames[0]
    # Both user centroids should be present (unmatched — distance >> 5px)
    assert len(merged_lf.centroids) == 2
    xs = {c.x for c in merged_lf.centroids}
    assert xs == {0.5, 50.5}


def test_labels_get_label_images_by_frame_idx_only():
    """get_label_images(frame_idx=N) without video filters by frame index."""
    video = Video(filename="test.mp4", open_backend=False)
    li0 = UserLabelImage(data=np.zeros((2, 2), dtype=np.int32))
    li1 = UserLabelImage(data=np.ones((2, 2), dtype=np.int32))
    lf0 = LabeledFrame(video=video, frame_idx=0, label_images=[li0])
    lf1 = LabeledFrame(video=video, frame_idx=1, label_images=[li1])
    labels = Labels(labeled_frames=[lf0, lf1])

    assert labels.get_label_images(frame_idx=0) == [li0]
    assert labels.get_label_images(frame_idx=1) == [li1]
    assert labels.get_label_images(frame_idx=99) == []


def test_labels_replace_videos_updates_static_rois():
    """replace_videos should update video references on static ROIs."""
    old_video = Video(filename="old.mp4")
    new_video = Video(filename="new.mp4")
    static_roi = UserROI.from_bbox(0, 0, 100, 100, video=old_video)

    labels = Labels(videos=[old_video], rois=[static_roi])
    labels.replace_videos(old_videos=[old_video], new_videos=[new_video])

    assert static_roi.video is new_video


def test_labels_materialize_static_rois_with_refs(tmp_path):
    """materialize() deep copies static ROIs and relinks video/track refs."""
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    skeleton = Skeleton(["A"])
    static_roi = UserROI.from_bbox(0, 0, 50, 50, video=video)
    static_roi.track = track

    lf = LabeledFrame(video=video, frame_idx=0)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        tracks=[track],
        skeletons=[skeleton],
        rois=[static_roi],
    )
    path = str(tmp_path / "test.slp")
    save_slp(labels, path)

    lazy = load_slp(path, lazy=True)
    materialized = lazy.materialize()

    assert len(materialized.static_rois) == 1
    mat_roi = materialized.static_rois[0]
    assert mat_roi.video is materialized.videos[0]
    assert mat_roi.track is materialized.tracks[0]


def test_remap_frame_annotations_with_rois():
    """_remap_frame_annotations remaps ROI video and track references."""
    old_video = Video(filename="old.mp4")
    new_video = Video(filename="new.mp4")
    old_track = Track(name="old")
    new_track = Track(name="new")

    roi = UserROI.from_bbox(0, 0, 10, 10, video=old_video)
    roi.track = old_track
    lf = LabeledFrame(video=old_video, frame_idx=0, rois=[roi])

    Labels._remap_frame_annotations(
        lf,
        video_map={old_video: new_video},
        track_map={old_track: new_track},
    )

    assert roi.video is new_video
    assert roi.track is new_track
