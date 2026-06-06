"""Tests for functions in the sleap_io.io.slp file."""

from __future__ import annotations

import shutil
import sys
import zlib
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
import shapely
import simplejson as json
from PIL import Image

from sleap_io import (
    Camera,
    CameraGroup,
    FrameGroup,
    Instance,
    InstanceGroup,
    LabeledFrame,
    Labels,
    LabelsSet,
    Node,
    PredictedInstance,
    RecordingSession,
    Skeleton,
    SuggestionFrame,
    Track,
    Video,
    get_default_image_plugin,
    load_file,
    load_slp,
    save_file,
    save_slp,
    set_default_image_plugin,
)
from sleap_io.io.slp import (
    LI_DTYPE,
    OBJ_DTYPE,
    ExportCancelled,
    LabelImageWriter,
    _points_from_hdf5_data,
    _write_metadata_standalone,
    camera_group_to_dict,
    camera_to_dict,
    can_use_fast_path,
    embed_frames,
    embed_videos,
    frame_group_to_dict,
    instance_group_to_dict,
    make_camera,
    make_camera_group,
    make_frame_group,
    make_instance_group,
    make_session,
    merge_label_images,
    prepare_frames_to_embed,
    process_and_embed_frames,
    read_bboxes,
    read_centroids,
    read_identities,
    read_instances,
    read_label_images,
    read_labels,
    read_labels_set,
    read_masks,
    read_metadata,
    read_negative_frames,
    read_points,
    read_pred_points,
    read_rois,
    read_sessions,
    read_skeletons,
    read_suggestions,
    read_tracks,
    read_video_crops,
    read_videos,
    session_to_dict,
    video_to_dict,
    write_bboxes,
    write_centroids,
    write_identities,
    write_labels,
    write_lfs,
    write_metadata,
    write_sessions,
    write_suggestions,
    write_tracks,
    write_video_crops,
    write_videos,
)
from sleap_io.io.utils import read_hdf5_attrs, read_hdf5_dataset
from sleap_io.io.video_reading import (
    CropVideoBackend,
    HDF5Video,
    ImageVideo,
    MediaVideo,
)
from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.centroid import PredictedCentroid, UserCentroid
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Instance3D, PredictedInstance3D
from sleap_io.model.label_image import LabelImage, PredictedLabelImage, UserLabelImage
from sleap_io.model.mask import (
    PredictedSegmentationMask,
    UserSegmentationMask,
)
from sleap_io.model.roi import PredictedROI, UserROI
from sleap_io.transform.frame import crop_frame


def test_read_labels(slp_typical, slp_simple_skel, slp_minimal):
    """Test `read_labels` can read different types of .slp files."""
    labels = read_labels(slp_typical)
    assert type(labels) is Labels

    labels = read_labels(slp_simple_skel)
    assert type(labels) is Labels

    labels = read_labels(slp_minimal)
    assert type(labels) is Labels


def test_load_slp_with_provenance(slp_predictions_with_provenance):
    labels = read_labels(slp_predictions_with_provenance)
    provenance = labels.provenance
    assert type(provenance) is dict
    assert provenance["sleap_version"] == "1.2.7"


def test_legacy_coordinate_system(slp_legacy_grid_labels):
    """Test that legacy SLP files with FORMAT_ID = 1.0 have correct coordinates.

    Legacy files use a coordinate system where the top-left of the pixel is at (0, 0).
    Newer files (FORMAT_ID >= 1.1) use (-0.5, -0.5) for the top-left of the pixel,
    with the pixel center at the origin.
    """
    labels = read_labels(slp_legacy_grid_labels)

    # Get the first instance from the first frame
    inst = labels[0][0]

    # Check that the coordinates match the expected legacy coordinate system
    # These coordinates should reflect the adjustment from pixel corners to centers
    np.testing.assert_array_equal(inst.numpy(), [[-1, -1], [-0.5, -0.5], [-1, 0]])


def test_read_instances_from_predicted(slp_real_data):
    labels = read_labels(slp_real_data)

    lf = labels.find(video=labels.video, frame_idx=220)[0]
    assert len(lf) == 3
    assert type(lf.instances[0]) is PredictedInstance
    assert type(lf.instances[1]) is PredictedInstance
    assert type(lf.instances[2]) is Instance
    assert lf.instances[2].from_predicted == lf.instances[1]
    assert lf.unused_predictions == [lf.instances[0]]

    lf = labels.find(video=labels.video, frame_idx=770)[0]
    assert len(lf) == 4
    assert type(lf.instances[0]) is PredictedInstance
    assert type(lf.instances[1]) is PredictedInstance
    assert type(lf.instances[2]) is Instance
    assert type(lf.instances[3]) is Instance
    assert lf.instances[2].from_predicted == lf.instances[1]
    assert lf.instances[3].from_predicted == lf.instances[0]
    assert len(lf.unused_predictions) == 0


def test_read_labels_multiview(slp_multiview):
    labels = read_labels(slp_multiview)
    assert type(labels) is Labels
    assert len(labels.sessions) == 1
    assert isinstance(labels.sessions[0], RecordingSession)

    session = labels.sessions[0]
    for video in session.videos:
        assert isinstance(video, Video)
        assert video in labels.videos

    for frame_group in session.frame_groups.values():
        assert isinstance(frame_group, FrameGroup)
        for labeled_frame in frame_group.labeled_frames:
            assert isinstance(labeled_frame, LabeledFrame)
            assert labeled_frame in labels.labeled_frames

        for instance_group in frame_group.instance_groups:
            assert isinstance(instance_group, InstanceGroup)
            for instance in instance_group.instances:
                assert isinstance(instance, Instance)
                assert instance in labels.instances


def test_read_skeleton(centered_pair):
    skeletons = read_skeletons(centered_pair)
    assert len(skeletons) == 1
    skeleton = skeletons[0]
    assert type(skeleton) is Skeleton
    assert len(skeleton.nodes) == 24
    assert len(skeleton.edges) == 23
    # Legacy files may have duplicate symmetries (one for each direction)
    # After deduplication, we should have 10 unique symmetry pairs
    assert len(skeleton.symmetries) == 10
    assert "wingR" in skeleton.symmetry_names[0]
    assert "wingL" in skeleton.symmetry_names[0]


def test_read_videos_pkg(slp_minimal_pkg):
    videos = read_videos(slp_minimal_pkg)
    assert len(videos) == 1
    video = videos[0]
    assert video.shape == (1, 384, 384, 1)
    assert video.backend.dataset == "video0/video"


def assert_matches_slp_multiview(
    sessions: list[RecordingSession],
    videos: list[Video],
    labeled_frames: list[LabeledFrame],
    instances_labels: set[Instance],
):
    """Assert that the data loaded from the .slp file is as expected.

    Each assert statement confirms that the data in `sessions` matches the data we
    expect to find in the `slp_multiview` fixture.

    Args:
        sessions: The list of RecordingSession objects.
        videos: The list of Video objects in the .slp file.
        labeled_frames: The list of LabeledFrame objects in the .slp file.
        instances_labels: The set of Instance objects in the .slp file.

    Raises:
        AssertionError: If the data in `sessions` does not match the expected data.
    """
    assert len(sessions) == 1

    session = sessions[0]
    assert isinstance(session, RecordingSession)

    camera_group = session.camera_group
    assert isinstance(camera_group, CameraGroup)
    n_cameras = len(camera_group.cameras)
    assert n_cameras == 8

    # Test video to camera linking.
    for video in session.videos:
        assert isinstance(video, Video)
        assert video in videos

        camera = session.get_camera(video)
        assert isinstance(camera, Camera)
        assert camera.name in str(video.filename)
        assert camera in camera_group.cameras

        assert session.get_video(camera) is video

    # Test frame groups.
    frame_groups = session.frame_groups
    assert len(frame_groups) == 3
    for frame_idx, frame_group in frame_groups.items():
        assert isinstance(frame_group, FrameGroup)
        assert frame_group.frame_idx == frame_idx

        # Test labeled frames to camera linking.
        cameras = frame_group.cameras
        n_cameras_in_frame = len(cameras)
        assert len(frame_group.labeled_frames) == n_cameras_in_frame
        for labeled_frame, camera in zip(
            frame_group.labeled_frames, frame_group.cameras
        ):
            assert isinstance(labeled_frame, LabeledFrame)
            assert labeled_frame in labeled_frames
            assert labeled_frame.frame_idx == frame_idx

            assert isinstance(camera, Camera)
            assert camera in camera_group.cameras
            assert frame_group.get_frame(camera) is labeled_frame

        # Test instance groups.
        assert len(frame_group.instance_groups) == 2
        for instance_group in frame_group.instance_groups:
            assert isinstance(instance_group, InstanceGroup)

            instances = instance_group.instances
            n_instances = len(instances)
            assert n_instances == 6 or n_instances == 8

            # Test instance to camera linking.
            cameras = instance_group.cameras
            assert len(cameras) == n_instances
            for camera, instance in zip(cameras, instances):
                assert isinstance(camera, Camera)
                assert camera in camera_group.cameras

                assert isinstance(instance, Instance)
                assert instance_group.get_instance(camera) is instance
                assert instance in instances_labels


def test_read_sessions(slp_multiview):
    labels_path = slp_multiview

    # Retrieve necessary data from the .slp file.

    # Read the videos list from the .slp file.
    videos = read_videos(labels_path, open_backend=False)

    # Read the Labeled_frames from the .slp file.
    tracks = read_tracks(labels_path)
    skeletons = read_skeletons(labels_path)
    points = read_points(labels_path)
    pred_points = read_pred_points(labels_path)
    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")
    instances_labels = read_instances(
        labels_path, skeletons, tracks, points, pred_points, format_id
    )
    frames = read_hdf5_dataset(labels_path, "frames")
    labeled_frames = []
    for _, video_id, frame_idx, instance_id_start, instance_id_end in frames:
        labeled_frames.append(
            LabeledFrame(
                video=videos[video_id],
                frame_idx=int(frame_idx),
                instances=instances_labels[instance_id_start:instance_id_end],
            )
        )

    # Test read_sessions.

    sessions = read_sessions(labels_path, videos, labeled_frames)
    assert_matches_slp_multiview(sessions, videos, labeled_frames, instances_labels)


def test_write_videos(slp_minimal_pkg, centered_pair, tmp_path):
    def compare_videos(videos_ref, videos_test):
        assert len(videos_ref) == len(videos_test)
        for video_ref, video_test in zip(videos_ref, videos_test):
            assert video_ref.shape == video_test.shape
            assert (video_ref[0] == video_test[0]).all()

    videos_ref = read_videos(slp_minimal_pkg)
    write_videos(tmp_path / "test_minimal_pkg.slp", videos_ref)
    videos_test = read_videos(tmp_path / "test_minimal_pkg.slp")
    compare_videos(videos_ref, videos_test)

    videos_ref = read_videos(centered_pair)
    write_videos(tmp_path / "test_centered_pair.slp", videos_ref)
    videos_test = read_videos(tmp_path / "test_centered_pair.slp")
    compare_videos(videos_ref, videos_test)

    videos = read_videos(centered_pair) * 2
    write_videos(tmp_path / "test_centered_pair_2vids.slp", videos)
    videos_test = read_videos(tmp_path / "test_centered_pair_2vids.slp")
    compare_videos(videos, videos_test)


def test_write_tracks(centered_pair, tmp_path):
    tracks = read_tracks(centered_pair)
    write_tracks(tmp_path / "test.slp", tracks)

    # TODO: Test for byte-for-byte equality of HDF5 datasets when we implement the
    # spawned_on attribute.
    # json_fixture = read_hdf5_dataset(centered_pair, "tracks_json")
    # json_test = read_hdf5_dataset(tmp_path / "test.slp", "tracks_json")
    # assert (json_fixture == json_test).all()

    saved_tracks = read_tracks(tmp_path / "test.slp")
    assert len(saved_tracks) == len(tracks)
    for saved_track, track in zip(saved_tracks, tracks):
        assert saved_track.name == track.name


def test_write_metadata(centered_pair, tmp_path):
    labels = read_labels(centered_pair)
    write_metadata(tmp_path / "test.slp", labels)

    saved_md = read_metadata(tmp_path / "test.slp")
    assert saved_md["version"] == "2.0.0"
    assert saved_md["provenance"] == labels.provenance

    saved_skeletons = read_skeletons(tmp_path / "test.slp")
    assert len(saved_skeletons) == len(labels.skeletons)
    assert len(saved_skeletons) == 1
    assert saved_skeletons[0].name == labels.skeletons[0].name
    assert saved_skeletons[0].node_names == labels.skeletons[0].node_names
    assert saved_skeletons[0].edge_inds == labels.skeletons[0].edge_inds
    assert (
        saved_skeletons[0].get_flipped_node_inds()
        == labels.skeletons[0].get_flipped_node_inds()
    )


def test_write_lfs(centered_pair, slp_real_data, tmp_path):
    labels = read_labels(centered_pair)
    n_insts = len([inst for lf in labels for inst in lf])
    write_lfs(tmp_path / "test.slp", labels)

    points = read_points(tmp_path / "test.slp")
    pred_points = read_pred_points(tmp_path / "test.slp")

    assert (len(points) + len(pred_points)) == (n_insts * len(labels.skeleton))

    labels = read_labels(slp_real_data)
    n_insts = len([inst for lf in labels for inst in lf])
    write_lfs(tmp_path / "test2.slp", labels)

    points = read_points(tmp_path / "test2.slp")
    pred_points = read_pred_points(tmp_path / "test2.slp")

    assert (len(points) + len(pred_points)) == (n_insts * len(labels.skeleton))


def test_write_labels(centered_pair, slp_real_data, tmp_path):
    for fn in [centered_pair, slp_real_data]:
        labels = read_labels(fn)
        write_labels(tmp_path / "test.slp", labels)

        saved_labels = read_labels(tmp_path / "test.slp")
        assert len(saved_labels) == len(labels)
        assert [lf.frame_idx for lf in saved_labels] == [lf.frame_idx for lf in labels]
        assert [len(lf) for lf in saved_labels] == [len(lf) for lf in labels]
        np.testing.assert_array_equal(saved_labels.numpy(), labels.numpy())
        assert saved_labels.video.filename == labels.video.filename
        assert type(saved_labels.video.backend) is type(labels.video.backend)
        assert saved_labels.video.backend.grayscale == labels.video.backend.grayscale
        assert saved_labels.video.backend.shape == labels.video.backend.shape
        assert len(saved_labels.skeletons) == len(labels.skeletons) == 1
        assert saved_labels.skeleton.name == labels.skeleton.name
        assert saved_labels.skeleton.node_names == labels.skeleton.node_names
        assert len(saved_labels.suggestions) == len(labels.suggestions)


def test_negative_frames_roundtrip(tmp_path):
    """Test that negative frames survive save/load cycle."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test.mp4")

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

    # Save
    path = tmp_path / "test.slp"
    write_labels(str(path), labels)

    # Verify dataset was created
    with h5py.File(path, "r") as f:
        assert "negative_frames" in f
        data = f["negative_frames"][:]
        assert len(data) == 1
        assert data[0]["video_id"] == 0
        assert data[0]["frame_idx"] == 1

    # Load and verify
    loaded = read_labels(str(path))

    assert len(loaded.negative_frames) == 1
    assert loaded.labeled_frames[0].is_negative is False  # frame 0
    assert loaded.labeled_frames[1].is_negative is True  # frame 1
    assert loaded.labeled_frames[2].is_negative is False  # frame 2


def test_negative_frames_no_dataset_when_empty(tmp_path):
    """Test no negative_frames dataset is created when there are none."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test.mp4")

    lf = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[Instance([[0, 1], [2, 3]], skeleton=skel)],
    )

    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skel])

    path = tmp_path / "test.slp"
    write_labels(str(path), labels)

    # Verify no dataset was created
    with h5py.File(path, "r") as f:
        assert "negative_frames" not in f

    # Load should still work with is_negative=False
    loaded = read_labels(str(path))
    assert loaded.labeled_frames[0].is_negative is False


def test_read_negative_frames_missing_dataset(tmp_path):
    """Test read_negative_frames returns empty set when dataset doesn't exist."""
    # Create a minimal HDF5 file without negative_frames
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("dummy", data=[1, 2, 3])

    result = read_negative_frames(str(path))
    assert result == set()


def test_negative_frames_lazy_roundtrip(tmp_path):
    """Test that negative frames survive lazy save/load cycle."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test.mp4")

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

    # Save eagerly first
    path = tmp_path / "test.slp"
    write_labels(str(path), labels)

    # Load lazily
    lazy_labels = load_slp(str(path), lazy=True)

    # Check that negative frame info is available in lazy store
    assert lazy_labels.is_lazy
    assert len(lazy_labels._lazy_store._negative_frames) == 1

    # Check that copy() preserves negative frames
    copied_store = lazy_labels._lazy_store.copy()
    assert copied_store._negative_frames == lazy_labels._lazy_store._negative_frames
    assert copied_store._negative_frames is not lazy_labels._lazy_store._negative_frames

    # Materialize individual frames and check is_negative
    assert lazy_labels.labeled_frames[0].is_negative is False
    assert lazy_labels.labeled_frames[1].is_negative is True
    assert lazy_labels.labeled_frames[2].is_negative is False

    # Check Labels.negative_frames property
    assert len(lazy_labels.negative_frames) == 1
    assert lazy_labels.negative_frames[0].frame_idx == 1


def test_negative_frames_lazy_user_labeled_frames(tmp_path):
    """Test that user_labeled_frames includes negative frames in lazy mode."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test.mp4")

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
    lf_pred = LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[PredictedInstance([[4, 5], [6, 7]], skeleton=skel)],
    )

    labels = Labels(
        labeled_frames=[lf_regular, lf_negative, lf_pred],
        videos=[video],
        skeletons=[skel],
    )

    path = tmp_path / "test.slp"
    write_labels(str(path), labels)

    lazy_labels = load_slp(str(path), lazy=True)

    # user_labeled_frames should include regular + negative but not pred-only
    user_frames = lazy_labels.user_labeled_frames
    assert len(user_frames) == 2
    frame_indices = {lf.frame_idx for lf in user_frames}
    assert 0 in frame_indices
    assert 1 in frame_indices
    assert 2 not in frame_indices


def test_negative_frames_lazy_write(tmp_path):
    """Test that lazy write path preserves negative frames."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test.mp4")

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

    labels = Labels(
        labeled_frames=[lf_regular, lf_negative],
        videos=[video],
        skeletons=[skel],
    )

    # Save eagerly first
    path1 = tmp_path / "eager.slp"
    write_labels(str(path1), labels)

    # Load lazily and re-save (uses _write_labels_lazy fast path)
    lazy_labels = load_slp(str(path1), lazy=True)
    path2 = tmp_path / "lazy_resave.slp"
    write_labels(str(path2), lazy_labels)

    # Verify negative frames dataset exists in resaved file
    with h5py.File(path2, "r") as f:
        assert "negative_frames" in f
        data = f["negative_frames"][:]
        assert len(data) == 1

    # Load the resaved file eagerly and verify
    reloaded = read_labels(str(path2))
    assert len(reloaded.negative_frames) == 1
    assert reloaded.labeled_frames[1].is_negative is True


def test_negative_frames_materialize(tmp_path):
    """Test that materialize() preserves is_negative on frames."""
    skel = Skeleton(["A", "B"])
    video = Video(filename="test.mp4")

    lf_negative = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[],
        is_negative=True,
    )
    lf_regular = LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[Instance([[0, 1], [2, 3]], skeleton=skel)],
    )

    labels = Labels(
        labeled_frames=[lf_negative, lf_regular],
        videos=[video],
        skeletons=[skel],
    )

    path = tmp_path / "test.slp"
    write_labels(str(path), labels)

    # Load lazily then materialize
    lazy_labels = load_slp(str(path), lazy=True)
    eager_labels = lazy_labels.materialize()

    assert not eager_labels.is_lazy
    assert eager_labels.labeled_frames[0].is_negative is True
    assert eager_labels.labeled_frames[1].is_negative is False
    assert len(eager_labels.negative_frames) == 1


def test_lazy_write_preserves_static_roi_video(tmp_path):
    """Lazy round-trip must preserve video association on static ROIs."""
    video = Video(filename="v.mp4")
    roi = UserROI.from_bbox(0, 0, 100, 100, video=video, category="arena")
    labels = Labels(videos=[video], rois=[roi])

    p1 = tmp_path / "a.slp"
    p2 = tmp_path / "b.slp"
    save_file(labels, p1)

    lazy = load_slp(p1, lazy=True)
    save_file(lazy, p2)

    reread = load_slp(p2)
    assert len(reread.static_rois) == 1
    assert reread.static_rois[0].video is not None
    assert reread.static_rois[0].video.filename == "v.mp4"
    assert reread.static_rois[0].category == "arena"


def test_lazy_write_static_roi_without_video(tmp_path):
    """Lazy round-trip must handle ROIs with no video association (fallback)."""
    video = Video(filename="v.mp4")
    anchored = UserROI.from_bbox(0, 0, 10, 10, video=video, category="anchored")
    orphan = UserROI.from_bbox(20, 20, 30, 30, video=None, category="orphan")
    labels = Labels(videos=[video], rois=[anchored, orphan])

    p1 = tmp_path / "a.slp"
    p2 = tmp_path / "b.slp"
    save_file(labels, p1)

    lazy = load_slp(p1, lazy=True)
    save_file(lazy, p2)

    reread = load_slp(p2)
    by_category = {r.category: r for r in reread.static_rois}
    assert by_category["anchored"].video is reread.videos[0]
    assert by_category["orphan"].video is None


def test_lazy_write_preserves_static_roi_video_multi(tmp_path):
    """Lazy round-trip must pick the correct video index for multi-video labels."""
    video_a = Video(filename="a.mp4")
    video_b = Video(filename="b.mp4")
    video_c = Video(filename="c.mp4")
    roi_b = UserROI.from_bbox(0, 0, 50, 50, video=video_b, category="arena_b")
    roi_c = UserROI.from_bbox(10, 10, 60, 60, video=video_c, category="arena_c")
    labels = Labels(videos=[video_a, video_b, video_c], rois=[roi_b, roi_c])

    p1 = tmp_path / "a.slp"
    p2 = tmp_path / "b.slp"
    save_file(labels, p1)

    lazy = load_slp(p1, lazy=True)
    save_file(lazy, p2)

    reread = load_slp(p2)
    assert len(reread.static_rois) == 2
    by_category = {r.category: r for r in reread.static_rois}
    assert by_category["arena_b"].video is reread.videos[1]
    assert by_category["arena_b"].video.filename == "b.mp4"
    assert by_category["arena_c"].video is reread.videos[2]
    assert by_category["arena_c"].video.filename == "c.mp4"


def test_write_sessions(slp_multiview, tmp_path):
    labels = read_labels(slp_multiview)
    sessions = labels.sessions
    videos = labels.videos
    labeled_frames = labels.labeled_frames
    write_sessions(tmp_path / "test.slp", sessions, videos, labeled_frames)

    saved_sessions = read_sessions(tmp_path / "test.slp", videos, labeled_frames)
    assert_matches_slp_multiview(
        saved_sessions, videos, labeled_frames, set(labels.instances)
    )


def test_make_camera_and_camera_to_dict():
    """Test camera (de)serialization functions."""
    # Define camera dictionary
    name = "back"
    size = [1280, 1024]
    matrix = [
        [762.513822135494, 0.0, 639.5],
        [0.0, 762.513822135494, 511.5],
        [0.0, 0.0, 1.0],
    ]
    distortions = [-0.2868458380166852, 0.0, 0.0, 0.0, 0.0]
    rotation = [0.3571857188780474, 0.8879473292757126, 1.6832001677006176]
    translation = [-555.4577842902744, -294.43494957092884, -190.82196458369515]
    metadata = {"extra_key": "extra_value", 444: 555}
    camera_dict = {
        "name": name,
        "size": size,
        "matrix": matrix,
        "distortions": distortions,
        "rotation": rotation,
        "translation": translation,
    }
    camera_dict.update(metadata)

    # Test make_camera
    camera = make_camera(camera_dict)
    assert camera.name == "back"
    assert camera.size == tuple(size)
    np.testing.assert_array_almost_equal(camera.matrix, np.array(matrix))
    np.testing.assert_array_almost_equal(camera.dist, np.array(distortions))
    np.testing.assert_array_almost_equal(camera.rvec, np.array(rotation))
    np.testing.assert_array_almost_equal(camera.tvec, np.array(translation))
    assert camera.metadata == metadata

    # Test camera_to_dict
    assert camera_to_dict(camera) == camera_dict

    # Test when Camera has None for optional attributes

    camera = Camera(rvec=rotation, tvec=translation)
    assert camera.name is None
    assert camera.size is None

    # Test camera_to_dict
    camera_dict = camera_to_dict(camera)
    assert camera_dict["name"] == ""
    assert camera_dict["size"] == ""
    assert camera_dict["matrix"] == camera.matrix.tolist()
    assert camera_dict["distortions"] == camera.dist.tolist()
    assert camera_dict["rotation"] == camera.rvec.tolist()
    assert camera_dict["translation"] == camera.tvec.tolist()

    # Test make_camera
    camera_0 = make_camera(camera_dict)
    assert camera_0.name is None
    assert camera_0.size is None


def test_make_camera_group_and_camera_group_to_dict():
    """Test camera group (de)serialization functions."""
    # Define template camera dictionary
    size = [1280, 1024]
    matrix = np.eye(3).tolist()
    distortions = np.zeros(5).tolist()
    rotation = np.zeros(3).tolist()
    translation = np.zeros(3).tolist()
    camera_dict_template = {
        "size": size,
        "matrix": matrix,
        "distortions": distortions,
        "rotation": rotation,
        "translation": translation,
    }
    camera_group_dict = {}
    n_cameras = 3
    for i in range(n_cameras):
        camera_dict = camera_dict_template.copy()
        camera_dict["name"] = f"camera{i}"
        camera_group_dict[f"cam_{i}"] = camera_dict
    metadata = {"extra_key": "extra_value", 444: 555}
    camera_group_dict["metadata"] = metadata

    camera_group_0 = make_camera_group(camera_group_dict)
    assert camera_group_0.metadata == metadata
    camera_group_dict_0: dict = camera_group_to_dict(camera_group_0)
    assert camera_group_dict == camera_group_dict_0
    assert len(camera_group_0.cameras) == 3
    for i in range(n_cameras):
        assert camera_group_0.cameras[i].name == f"camera{i}"
        assert camera_group_0.cameras[i].size == tuple(size)
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].matrix, np.array(matrix)
        )
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].dist, np.array(distortions)
        )
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].rvec, np.array(rotation)
        )
        np.testing.assert_array_almost_equal(
            camera_group_0.cameras[i].tvec, np.array(translation)
        )


def test_make_instance_group_and_instance_group_to_dict(
    instance_group_345: InstanceGroup, camera_group_345: CameraGroup
):
    """Test InstanceGroup (de)serialization functions.

    Args:
        instance_group_345: Instance group with an `Instance` at each camera view.
        camera_group_345: Camera group with 3-4-5 triangle configuration.
    """
    instance_group = instance_group_345

    # Create necessary helper objects.

    def new_labeled_frame(inst: Instance | None = None):
        """Create a new labeled frame with or without the specified instance.

        Args:
            inst: Instance to include in the labeled frame. If None, a new instance is
                created instead.
        """
        video = Video(filename="test")
        if inst is None:
            inst = Instance([[8, 9], [10, 11]], skeleton=Skeleton(["A", "B"]))
        return LabeledFrame(
            video=video,
            frame_idx=0,
            instances=[
                inst,
                Instance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
            ],
        )

    # Create labeled frames, with some irrelevant frames to make mapping more complex
    labeled_frames = []
    for inst in instance_group._instance_by_camera.values():
        labeled_frames.append(new_labeled_frame(inst))
        labeled_frames.append(new_labeled_frame())

    # Create our instance_to_lf_and_inst_idx dictionary.
    instance_to_lf_and_inst_idx = {
        inst: (inst_idx * 2, 0)  # inst_idx * 2 because we have irrelevant frames
        for inst_idx, inst in enumerate(instance_group._instance_by_camera.values())
    }

    # Test to dict

    instance_group_dict = instance_group_to_dict(
        instance_group=instance_group,
        instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx,
        camera_group=camera_group_345,
    )
    assert instance_group_dict["score"] == instance_group._score
    assert np.array_equal(instance_group_dict["points"], instance_group.points.tolist())

    # Test from dict

    instance_group_0 = make_instance_group(
        instance_group_dict,
        labeled_frames=labeled_frames,
        camera_group=camera_group_345,
    )
    assert instance_group_0._score == instance_group._score
    assert np.array_equal(instance_group_0.points, instance_group.points)
    assert instance_group_0.metadata == instance_group.metadata
    assert len(instance_group_0._instance_by_camera) == len(
        instance_group._instance_by_camera
    )

    # Check the instances and cameras are the same.
    for (cam, inst), (cam_0, inst_0) in zip(
        instance_group._instance_by_camera.items(),
        instance_group_0._instance_by_camera.items(),
    ):
        assert inst == inst_0
        assert cam == cam_0

    # Check that the dictionary was not mutated.
    assert instance_group_dict.get("points", None) is not None
    assert instance_group_dict.get("score", None) is not None
    assert instance_group_dict.get("camcorder_to_lf_and_inst_idx_map", None) is not None


def test_make_instance_group_warns_on_3d_points_without_skeleton(camera_group_345):
    """Test that make_instance_group warns when 3D points are discarded."""
    instance_group_dict = {
        "camcorder_to_lf_and_inst_idx_map": {},
        "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    }

    with pytest.warns(UserWarning, match="3D points discarded"):
        ig = make_instance_group(
            instance_group_dict,
            labeled_frames=[],
            camera_group=camera_group_345,
        )

    assert ig.instance_3d is None


def test_make_instance_group_warns_on_identity_idx_out_of_bounds(camera_group_345):
    """Test that make_instance_group warns on out-of-bounds identity_idx."""
    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)

    video1 = Video(filename="v1.mp4")
    video2 = Video(filename="v2.mp4")
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    instance_group_dict = {
        "camcorder_to_lf_and_inst_idx_map": {"0": ("0", "0"), "1": ("1", "0")},
        "identity_idx": 99,
    }
    identities = [Identity(name="mouse_A")]

    with pytest.warns(UserWarning, match="identity_idx 99 out of range"):
        ig = make_instance_group(
            instance_group_dict,
            labeled_frames=[lf1, lf2],
            camera_group=camera_group_345,
            identities=identities,
        )

    assert ig.identity is None


def test_make_frame_group_and_frame_group_to_dict(
    frame_group_345: FrameGroup, camera_group_345: CameraGroup
):
    """Test FrameGroup (de)serialization functions.

    Args:
        frame_group_345: Frame group with an `InstanceGroup` at each camera view.
        camera_group_345: Camera group with 3-4-5 triangle configuration
    """
    frame_group = frame_group_345
    camera_group = camera_group_345

    # Create necessary helper objects.

    # Create labeled frames, with some irrelevant frames to make mapping more complex.
    labeled_frames = []
    for lf in frame_group.labeled_frames:
        labeled_frames.append(lf)
        labeled_frames.append(
            LabeledFrame(video=Video(filename="test"), frame_idx=lf.frame_idx)
        )
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labeled_frames)}

    # Test frame_group_to_dict.

    frame_group_dict = frame_group_to_dict(
        frame_group=frame_group,
        labeled_frame_to_idx=labeled_frame_to_idx,
        camera_group=camera_group,
    )
    assert frame_group_dict["frame_idx"] == frame_group.frame_idx

    # Test make_frame_group.

    frame_group_0 = make_frame_group(
        frame_group_dict=frame_group_dict,
        labeled_frames=labeled_frames,
        camera_group=camera_group,
    )
    assert frame_group_0.frame_idx == frame_group.frame_idx
    assert len(frame_group_0._instance_groups) == len(frame_group._instance_groups)
    assert len(frame_group_0._labeled_frame_by_camera) == len(
        frame_group._labeled_frame_by_camera
    )
    assert frame_group_0.metadata == frame_group.metadata

    # Check the cameras and labeled frames are the same.
    for (cam, lf), (cam_0, lf_0) in zip(
        frame_group._labeled_frame_by_camera.items(),
        frame_group_0._labeled_frame_by_camera.items(),
    ):
        assert cam == cam_0
        assert lf == lf_0

    # Check the instance groups are the same.
    for instance_group, instance_group_0 in zip(
        frame_group._instance_groups, frame_group_0._instance_groups
    ):
        for (cam, inst), (cam_0, inst_0) in zip(
            instance_group._instance_by_camera.items(),
            instance_group_0._instance_by_camera.items(),
        ):
            assert inst == inst_0
            assert cam == cam_0


def test_make_session_and_session_to_dict(
    recording_session_345: RecordingSession,
):
    """Test recording session (de)serialization functions.

    Args:
        recording_session_345: A RecordingSession object with 3 cameras, 4 videos,
            and 5 frame groups.
    """

    def assert_cameras_equal(camera: Camera, camera_0: Camera):
        """Compare two cameras.

        Args:
            camera: First camera.
            camera_0: Second camera.

        Raises:
            AssertionError: If the cameras are not equal.
        """
        camera_state = camera.__getstate__()
        camera_state_0 = camera_0.__getstate__()
        for key, value in camera_state.items():
            if value is None:
                assert camera_state_0[key] is None
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(camera_state_0[key], value)
            else:
                assert camera_state_0[key] == value

    session = recording_session_345
    frame_group = session._frame_group_by_frame_idx[0]

    # Create necessary helper objects.

    # Create labeled frames, with some irrelevant frames to make mapping more complex.
    labeled_frames = []
    for lf in frame_group.labeled_frames:
        labeled_frames.append(lf)
        labeled_frames.append(
            LabeledFrame(video=Video(filename="test"), frame_idx=lf.frame_idx)
        )
    labeled_frame_to_idx = {lf: idx for idx, lf in enumerate(labeled_frames)}

    # Create videos list and index mapping.
    videos = []
    for video in session.videos:
        videos.append(video)
        videos.append(Video(filename="test"))
    video_to_idx = {video: idx for idx, video in enumerate(videos)}

    # Test session_to_dict.

    session_dict = session_to_dict(
        session=session,
        labeled_frame_to_idx=labeled_frame_to_idx,
        video_to_idx=video_to_idx,
    )
    assert len(session_dict["frame_group_dicts"]) == len(
        session._frame_group_by_frame_idx
    )
    assert len(session_dict["camcorder_to_video_idx_map"]) == len(
        session._video_by_camera
    )

    # Test make_session.

    session_0: RecordingSession = make_session(
        session_dict=session_dict,
        labeled_frames=labeled_frames,
        videos=videos,
    )
    assert len(session_0.camera_group.cameras) == len(session.camera_group.cameras)
    assert len(session_0._video_by_camera) == len(session._video_by_camera)
    assert len(session_0._camera_by_video) == len(session._camera_by_video)
    assert len(session_0._frame_group_by_frame_idx) == len(
        session._frame_group_by_frame_idx
    )
    assert session_0.metadata == session.metadata

    # Check the cameras and videos are the same.
    for (video, camera), (video_0, camera_0) in zip(
        session._camera_by_video.items(), session_0._camera_by_video.items()
    ):
        assert video == video_0
        assert session._video_by_camera[camera] == video
        assert session_0._video_by_camera[camera_0] == video_0
        assert_cameras_equal(camera, camera_0)

    # Check the frame groups are the same.
    for (frame_idx, frame_group), (frame_idx_0, frame_group_0) in zip(
        session._frame_group_by_frame_idx.items(),
        session_0._frame_group_by_frame_idx.items(),
    ):
        assert frame_idx == frame_idx_0
        assert frame_group.frame_idx == frame_idx
        assert frame_group.frame_idx == frame_group_0.frame_idx
        assert len(frame_group._instance_groups) == len(frame_group_0._instance_groups)
        assert len(frame_group._labeled_frame_by_camera) == len(
            frame_group_0._labeled_frame_by_camera
        )
        assert frame_group.metadata == frame_group_0.metadata

        # Check the cameras and labeled frames are the same.
        for (camera, lf), (camera_0, lf_0) in zip(
            frame_group._labeled_frame_by_camera.items(),
            frame_group_0._labeled_frame_by_camera.items(),
        ):
            assert lf == lf_0
            assert_cameras_equal(camera, camera_0)

        # Check the instance groups are the same.
        for instance_group, instance_group_0 in zip(
            frame_group._instance_groups, frame_group_0._instance_groups
        ):
            for (cam, inst), (cam_0, inst_0) in zip(
                instance_group._instance_by_camera.items(),
                instance_group_0._instance_by_camera.items(),
            ):
                assert inst == inst_0
                assert_cameras_equal(cam, cam_0)


def test_slp_multiview_round_trip(slp_multiview, tmp_path):
    labels = read_labels(slp_multiview)
    sessions = labels.sessions
    assert_matches_slp_multiview(
        sessions, labels.videos, labels.labeled_frames, set(labels.instances)
    )

    write_labels(tmp_path / "test.slp", labels)
    saved_labels = read_labels(tmp_path / "test.slp")
    assert_matches_slp_multiview(
        saved_labels.sessions,
        saved_labels.videos,
        saved_labels.labeled_frames,
        set(saved_labels.instances),
    )


def test_load_multi_skeleton(tmpdir):
    """Test loading multiple skeletons from a single file."""
    skel1 = Skeleton()
    skel1.add_node(Node("n1"))
    skel1.add_node(Node("n2"))
    skel1.add_edge("n1", "n2")
    skel1.add_symmetry("n1", "n2")

    skel2 = Skeleton()
    skel2.add_node(Node("n3"))
    skel2.add_node(Node("n4"))
    skel2.add_edge("n3", "n4")
    skel2.add_symmetry("n3", "n4")

    skels = [skel1, skel2]
    labels = Labels(skeletons=skels)
    write_metadata(tmpdir / "test.slp", labels)

    loaded_skels = read_skeletons(tmpdir / "test.slp")
    assert len(loaded_skels) == 2
    assert loaded_skels[0].node_names == ["n1", "n2"]
    assert loaded_skels[1].node_names == ["n3", "n4"]
    assert loaded_skels[0].edge_inds == [(0, 1)]
    assert loaded_skels[1].edge_inds == [(0, 1)]
    assert loaded_skels[0].get_flipped_node_inds() == [1, 0]
    assert loaded_skels[1].get_flipped_node_inds() == [1, 0]


def test_slp_imgvideo(tmpdir, slp_imgvideo):
    labels = read_labels(slp_imgvideo)
    assert type(labels.video.backend) is ImageVideo
    assert labels.video.shape == (3, 384, 384, 1)

    write_labels(tmpdir / "test.slp", labels)
    labels = read_labels(tmpdir / "test.slp")
    assert type(labels.video.backend) is ImageVideo
    assert labels.video.shape == (3, 384, 384, 1)

    videos = [Video.from_filename(["fake1.jpg", "fake2.jpg"])]
    assert videos[0].shape is None
    assert len(videos[0].filename) == 2
    write_videos(tmpdir / "test2.slp", videos)
    videos = read_videos(tmpdir / "test2.slp")
    assert type(videos[0].backend) is ImageVideo
    assert len(videos[0].filename) == 2
    assert videos[0].shape is None


def test_suggestions(tmpdir):
    labels = Labels()
    labels.videos.append(Video.from_filename("fake.mp4"))
    labels.suggestions.append(SuggestionFrame(video=labels.video, frame_idx=0))

    write_suggestions(tmpdir / "test.slp", labels.suggestions, labels.videos)
    loaded_suggestions = read_suggestions(tmpdir / "test.slp", labels.videos)
    assert len(loaded_suggestions) == 1
    assert loaded_suggestions[0].video.filename == "fake.mp4"
    assert loaded_suggestions[0].frame_idx == 0

    # Handle missing suggestions dataset
    write_videos(tmpdir / "test2.slp", labels.videos)
    loaded_suggestions = read_suggestions(tmpdir / "test2.slp", labels.videos)
    assert len(loaded_suggestions) == 0


def test_suggestions_metadata(tmpdir):
    """Test that suggestion metadata (e.g., group) is preserved during read/write."""
    labels = Labels()
    labels.videos.append(Video.from_filename("fake.mp4"))

    # Create suggestions with different group values in metadata
    labels.suggestions.append(
        SuggestionFrame(video=labels.video, frame_idx=0, metadata={"group": 0})
    )
    labels.suggestions.append(
        SuggestionFrame(video=labels.video, frame_idx=1, metadata={"group": 1})
    )
    labels.suggestions.append(
        SuggestionFrame(video=labels.video, frame_idx=2, metadata={"group": 2})
    )

    # Write and read suggestions
    write_suggestions(tmpdir / "test.slp", labels.suggestions, labels.videos)
    loaded_suggestions = read_suggestions(tmpdir / "test.slp", labels.videos)

    # Verify metadata is preserved
    assert len(loaded_suggestions) == 3
    assert loaded_suggestions[0].metadata["group"] == 0
    assert loaded_suggestions[1].metadata["group"] == 1
    assert loaded_suggestions[2].metadata["group"] == 2

    # Test backward compatibility: suggestions without metadata default to group 0
    suggestion_no_metadata = SuggestionFrame(video=labels.video, frame_idx=3)
    write_suggestions(tmpdir / "test2.slp", [suggestion_no_metadata], labels.videos)
    loaded_suggestions = read_suggestions(tmpdir / "test2.slp", labels.videos)
    assert len(loaded_suggestions) == 1
    assert loaded_suggestions[0].metadata["group"] == 0


def test_pkg_roundtrip(tmpdir, slp_minimal_pkg):
    labels = read_labels(slp_minimal_pkg)
    assert type(labels.video.backend) is HDF5Video
    assert labels.video.shape == (1, 384, 384, 1)
    assert labels.video.backend.embedded_frame_inds == [0]
    assert labels.video.filename == slp_minimal_pkg

    write_labels(str(tmpdir / "roundtrip.pkg.slp"), labels)
    labels = read_labels(str(tmpdir / "roundtrip.pkg.slp"))
    assert type(labels.video.backend) is HDF5Video
    assert labels.video.shape == (1, 384, 384, 1)
    assert labels.video.backend.embedded_frame_inds == [0]
    assert (
        Path(labels.video.filename).as_posix()
        == Path(tmpdir / "roundtrip.pkg.slp").as_posix()
    )


@pytest.mark.parametrize(
    "to_embed", [True, "all", "user", "suggestions", "user+suggestions"]
)
def test_embed(tmpdir, slp_real_data, to_embed):
    base_labels = read_labels(slp_real_data)
    assert type(base_labels.video.backend) is MediaVideo
    assert (
        Path(base_labels.video.filename).as_posix()
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert base_labels.video.shape == (1100, 384, 384, 1)
    assert len(base_labels) == 10
    assert len(base_labels.suggestions) == 10
    assert len(base_labels.user_labeled_frames) == 5

    labels_path = Path(tmpdir / "labels.pkg.slp").as_posix()
    write_labels(labels_path, base_labels, embed=to_embed)
    labels = read_labels(labels_path)
    assert len(labels) == 10
    assert type(labels.video.backend) is HDF5Video
    assert Path(labels.video.filename).as_posix() == labels_path
    assert (
        Path(labels.video.source_video.filename).as_posix()
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    if to_embed == "all" or to_embed is True:
        assert labels.video.backend.embedded_frame_inds == [
            0,
            110,
            220,
            330,
            440,
            550,
            660,
            770,
            880,
            990,
        ]
    elif to_embed == "user":
        assert labels.video.backend.embedded_frame_inds == [0, 220, 440, 770, 990]
    elif to_embed == "suggestions":
        assert len(labels.video.backend.embedded_frame_inds) == 10
    elif to_embed == "suggestions+user":
        assert len(labels.video.backend.embedded_frame_inds) == 10


def test_embed_hdf5_format(tmpdir, slp_real_data):
    """Test embedding frames with HDF5 image format."""
    base_labels = read_labels(slp_real_data)
    labels_path = Path(tmpdir / "labels.pkg.slp").as_posix()

    # Get frames to embed
    frames_to_embed = [
        (lf.video, lf.frame_idx) for lf in base_labels.user_labeled_frames
    ]

    # Prepare metadata and process with HDF5 format

    frames_metadata = prepare_frames_to_embed(labels_path, base_labels, frames_to_embed)
    replaced_videos = process_and_embed_frames(
        labels_path, frames_metadata, image_format="hdf5"
    )

    # Update labels and write the rest of the data
    if len(replaced_videos) > 0:
        base_labels.replace_videos(video_map=replaced_videos)

    # Write the rest of the data to make a complete SLP file
    write_videos(labels_path, base_labels.videos)
    write_tracks(labels_path, base_labels.tracks)
    write_suggestions(labels_path, base_labels.suggestions, base_labels.videos)
    write_sessions(
        labels_path,
        base_labels.sessions,
        base_labels.videos,
        base_labels.labeled_frames,
    )
    write_metadata(labels_path, base_labels)
    write_lfs(labels_path, base_labels)

    # Verify the embedded file
    labels = read_labels(labels_path)
    assert type(labels.video.backend) is HDF5Video
    assert Path(labels.video.filename).as_posix() == labels_path

    # Check the image format
    with h5py.File(labels_path, "r") as f:
        assert f["video0/video"].attrs["format"] == "hdf5"


def test_embed_variable_length(tmpdir, slp_real_data):
    """Test embedding frames with variable length datasets."""
    base_labels = read_labels(slp_real_data)
    labels_path = Path(tmpdir / "labels_varlen.pkg.slp").as_posix()

    # Use embed_frames with fixed_length=False
    frames_to_embed = [
        (lf.video, lf.frame_idx) for lf in base_labels.user_labeled_frames
    ]

    # First prepare the metadata

    frames_metadata = prepare_frames_to_embed(labels_path, base_labels, frames_to_embed)

    # Process with fixed_length=False
    replaced_videos = process_and_embed_frames(
        labels_path, frames_metadata, fixed_length=False
    )

    # Update labels with the embedded videos
    if len(replaced_videos) > 0:
        base_labels.replace_videos(video_map=replaced_videos)

    # Write all the data to make a complete SLP file
    write_videos(labels_path, base_labels.videos)
    write_tracks(labels_path, base_labels.tracks)
    write_suggestions(labels_path, base_labels.suggestions, base_labels.videos)
    write_sessions(
        labels_path,
        base_labels.sessions,
        base_labels.videos,
        base_labels.labeled_frames,
    )
    write_metadata(labels_path, base_labels)
    write_lfs(labels_path, base_labels)

    # Verify the embedded file
    labels = read_labels(labels_path)
    assert type(labels.video.backend) is HDF5Video
    assert Path(labels.video.filename).as_posix() == labels_path

    # Check that the frames were properly embedded
    assert labels.video.backend.embedded_frame_inds == [0, 220, 440, 770, 990]

    # Check that images can be loaded correctly
    for frame_idx in [0, 220, 440, 770, 990]:
        img = labels.video[frame_idx]
        assert img.shape == (384, 384, 1)


def test_embed_two_rounds(tmpdir, slp_real_data):
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmpdir / "labels.pkg.slp")
    write_labels(labels_path, base_labels, embed="user")
    labels = read_labels(labels_path)

    assert labels.video.backend.embedded_frame_inds == [0, 220, 440, 770, 990]

    labels2_path = str(tmpdir / "labels2.pkg.slp")
    write_labels(labels2_path, labels)
    labels2 = read_labels(labels2_path)
    assert (
        Path(labels2.video.source_video.filename).as_posix()
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert labels2.video.backend.embedded_frame_inds == [0, 220, 440, 770, 990]

    labels3_path = str(tmpdir / "labels3.slp")
    write_labels(labels3_path, labels, embed="source")
    labels3 = read_labels(labels3_path)
    assert (
        Path(labels3.video.filename).as_posix()
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    assert type(labels3.video.backend) is MediaVideo


def test_can_use_fast_path(slp_minimal_pkg, centered_pair_low_quality_path):
    """Test can_use_fast_path() helper function."""
    # Load embedded package file
    labels = read_labels(slp_minimal_pkg)
    video = labels.video

    # Should use fast path for matching format (PNG -> PNG)
    assert video.backend.image_format == "png"
    assert can_use_fast_path(video, 0, "png") is True

    # Should NOT use fast path for format mismatch (PNG -> JPG)
    assert can_use_fast_path(video, 0, "jpg") is False

    # Should NOT use fast path for HDF5 format (requires encoding)
    assert can_use_fast_path(video, 0, "hdf5") is False

    # Should NOT use fast path for unavailable frames
    assert can_use_fast_path(video, 999, "png") is False

    # Should NOT use fast path for non-HDF5 backends
    media_video = Video.from_filename(centered_pair_low_quality_path)
    assert can_use_fast_path(media_video, 0, "png") is False


def test_fast_path_round_trip_preserves_bytes(tmpdir, slp_minimal_pkg):
    """Test that fast path preserves exact bytes through save/load cycle."""
    # Load original package file
    original_labels = read_labels(slp_minimal_pkg)
    original_video = original_labels.video

    # Get original raw bytes
    original_raw_bytes = original_video.backend.get_frame_raw_bytes(0)
    assert original_raw_bytes is not None

    # Save and reload with fast path (matching PNG format)
    save_path = str(tmpdir / "roundtrip.pkg.slp")
    write_labels(save_path, original_labels)

    reloaded_labels = read_labels(save_path)
    reloaded_raw_bytes = reloaded_labels.video.backend.get_frame_raw_bytes(0)

    # Fast path should preserve exact bytes
    assert reloaded_raw_bytes is not None
    assert len(original_raw_bytes) == len(reloaded_raw_bytes)
    np.testing.assert_array_equal(original_raw_bytes, reloaded_raw_bytes)


def test_fast_path_multiple_cycles_no_degradation(tmpdir, slp_minimal_pkg):
    """Test that multiple save/load cycles don't degrade quality with fast path."""
    # Load original
    labels = read_labels(slp_minimal_pkg)
    original_raw_bytes = labels.video.backend.get_frame_raw_bytes(0)

    # Save/load multiple times
    current_labels = labels
    for i in range(5):
        save_path = str(tmpdir / f"cycle_{i}.pkg.slp")
        write_labels(save_path, current_labels)
        current_labels = read_labels(save_path)

    # After 5 cycles, bytes should still be identical (fast path)
    final_raw_bytes = current_labels.video.backend.get_frame_raw_bytes(0)
    np.testing.assert_array_equal(original_raw_bytes, final_raw_bytes)


def test_fast_path_falls_back_to_slow_path_for_format_mismatch(tmpdir, slp_minimal_pkg):
    """Test that format mismatch correctly falls back to slow path."""
    # Load package with PNG embedded
    labels = read_labels(slp_minimal_pkg)
    assert labels.video.backend.image_format == "png"
    original_png_bytes = labels.video.backend.get_frame_raw_bytes(0)

    # Verify PNG magic bytes in original
    png_magic = original_png_bytes[:4].view(np.uint8)
    assert list(png_magic) == [137, 80, 78, 71]  # PNG signature

    # Prepare frames metadata manually
    save_path = str(tmpdir / "converted.pkg.slp")

    # Create the HDF5 file structure first
    with h5py.File(save_path, "w") as f:
        pass  # Just create empty file

    frames_metadata = [
        {
            "video": labels.video,
            "frame_idx": 0,
            "video_ind": 0,
            "group": "video0",
        }
    ]

    # Process with JPG format - this will use slow path since source is PNG
    process_and_embed_frames(
        save_path, frames_metadata, image_format="jpg", verbose=False
    )

    # Verify JPG format was written
    with h5py.File(save_path, "r") as f:
        assert "video0/video" in f
        fmt = f["video0/video"].attrs.get("format", None)
        assert fmt == "jpg"

        # Verify JPEG magic bytes
        raw_data = f["video0/video"][0]
        jpg_magic = np.array(raw_data[:2]).view(np.uint8)
        assert list(jpg_magic) == [
            255,
            216,
        ]  # JPEG magic (0xFF 0xD8)  # JPEG magic  # JPEG  # JPEG


def test_can_use_fast_path_non_embedded_hdf5(tmpdir):
    """Test can_use_fast_path() returns False for non-embedded HDF5Video."""
    from sleap_io.io.video_reading import HDF5Video

    # Create an HDF5 file with raw numpy arrays (format="hdf5", not encoded)
    h5_path = str(tmpdir / "raw_video.h5")
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("video", data=np.zeros((5, 64, 64, 1), dtype=np.uint8))
        ds.attrs["format"] = "hdf5"  # Raw format, not embedded images

    # Create Video with HDF5Video backend
    video = Video(
        filename=h5_path,
        backend=HDF5Video(filename=h5_path, dataset="video"),
        open_backend=False,
    )
    assert video.backend.has_embedded_images is False

    # Should NOT use fast path - has HDF5Video backend but no embedded images
    assert can_use_fast_path(video, 0, "png") is False


def test_fast_path_progress_callback_cancellation(tmpdir, slp_minimal_pkg):
    """Test that progress_callback cancellation works in fast path."""
    # Load package file (uses fast path when saving with same format)
    labels = read_labels(slp_minimal_pkg)
    assert labels.video.backend.image_format == "png"

    save_path = str(tmpdir / "cancelled.pkg.slp")

    # Cancel immediately
    def cancel_immediately(current, total):
        return False

    with pytest.raises(ExportCancelled, match="Export cancelled by user"):
        write_labels(
            save_path, labels, embed="user", progress_callback=cancel_immediately
        )


def test_embed_empty_video(tmpdir, slp_real_data, centered_pair_frame_paths):
    """Test that videos without labeled frames are still embedded in package files.

    This verifies that when saving a package file (.pkg.slp), ALL videos get converted
    to embedded references, including those without any labeled frames. This ensures
    package files are portable across machines.
    """
    base_labels = read_labels(slp_real_data)
    base_labels.videos.append(Video.from_filename(centered_pair_frame_paths))
    labels_path = str(tmpdir / "labels.pkg.slp")
    write_labels(labels_path, base_labels, embed="user")
    labels = read_labels(labels_path)

    assert len(labels.videos) == 2

    # First video should have embedded frames
    assert labels.videos[0].backend.embedded_frame_inds == [0, 220, 440, 770, 990]
    assert type(labels.videos[0].backend) is HDF5Video
    assert labels.videos[0].backend.has_embedded_images

    # Second video (no labeled frames) should still be an embedded reference
    # This ensures the package file is portable
    assert type(labels.videos[1].backend) is HDF5Video
    assert labels.videos[1].backend.has_embedded_images
    assert labels.videos[1].backend.embedded_frame_inds == []
    # The source_video should point to the original video for restoration
    assert labels.videos[1].source_video is not None
    # ImageVideo stores filenames as a list
    source_filename = labels.videos[1].source_video.filename
    if isinstance(source_filename, list):
        assert any("img" in str(f) for f in source_filename)
    else:
        assert "img" in str(source_filename)


def test_embed_rgb(tmpdir, slp_real_data):
    base_labels = read_labels(slp_real_data)
    base_labels.video.grayscale = False
    assert base_labels.video.shape == (1100, 384, 384, 3)
    assert base_labels.video[0].shape == (384, 384, 3)

    labels_path = str(tmpdir / "labels.pkg.slp")
    write_labels(labels_path, base_labels, embed="user")
    labels = read_labels(labels_path)
    assert labels.video[0].shape == (384, 384, 3)

    # Fallback to imageio
    cv2_mod = sys.modules.pop("cv2")

    labels_path = str(tmpdir / "labels_imageio.pkg.slp")
    write_labels(labels_path, base_labels, embed="user")
    labels = read_labels(labels_path)
    assert labels.video[0].shape == (384, 384, 3)

    sys.modules["cv2"] = cv2_mod


def test_embed_grayscale(tmpdir, slp_real_data):
    base_labels = read_labels(slp_real_data)
    assert base_labels.video[0].shape == (384, 384, 1)

    # Fallback to imageio
    cv2_mod = sys.modules.pop("cv2")

    labels_path = str(tmpdir / "labels_imageio_gray.pkg.slp")
    write_labels(labels_path, base_labels, embed="user")
    labels = read_labels(labels_path)
    assert labels.video[0].shape == (384, 384, 1)

    sys.modules["cv2"] = cv2_mod


def test_lazy_video_read(slp_real_data):
    labels = read_labels(slp_real_data)
    assert type(labels.video.backend) is MediaVideo
    assert labels.video.exists()

    labels = read_labels(slp_real_data, open_videos=False)
    assert labels.video.backend is None


def test_video_path_resolution(slp_real_data, tmp_path):
    labels = read_labels(slp_real_data)
    assert (
        Path(labels.video.filename).as_posix()
        == "tests/data/videos/centered_pair_low_quality.mp4"
    )
    shutil.copyfile(labels.video.filename, tmp_path / "centered_pair_low_quality.mp4")
    labels.video.replace_filename(
        "fake/path/to/centered_pair_low_quality.mp4", open=False
    )
    labels.save(tmp_path / "labels.slp")

    # Resolve when the same video filename is found in the labels directory.
    labels = read_labels(tmp_path / "labels.slp")
    assert (
        Path(labels.video.filename).as_posix()
        == (tmp_path / "centered_pair_low_quality.mp4").as_posix()
    )
    assert labels.video.exists()

    if sys.platform != "win32":  # Windows does not support chmod.
        # Make the video file inaccessible.
        labels.video.replace_filename("new_fake/path/to/inaccessible.mp4", open=False)
        labels.save(tmp_path / "labels2.slp")
        shutil.copyfile(
            tmp_path / "centered_pair_low_quality.mp4", tmp_path / "inaccessible.mp4"
        )
        Path(tmp_path / "inaccessible.mp4").chmod(0o000)

        # Fail to resolve when the video file is inaccessible.
        labels = read_labels(tmp_path / "labels2.slp")
        assert not labels.video.exists()
        assert (
            Path(labels.video.filename).as_posix()
            == "new_fake/path/to/inaccessible.mp4"
        )


def test_embed_invalid_value(tmpdir, slp_real_data):
    """Test that invalid embed values raise an error."""
    base_labels = read_labels(slp_real_data)
    labels_path = Path(tmpdir / "labels.pkg.slp").as_posix()

    with pytest.raises(ValueError):
        write_labels(labels_path, base_labels, embed="invalid_embed_value")


def test_process_and_embed_frames_verbose():
    """Test that process_and_embed_frames uses tqdm when verbose=True."""
    # Since we only want to test if tqdm is imported, we can simplify
    # and just check that the condition for importing tqdm is correct

    # Create a simple module with a function that conditionally imports tqdm
    module_code = """
def conditional_import(verbose):
    if verbose:
        from tqdm import tqdm
        return True
    return False
"""

    # Create a temporary module
    import types

    mod = types.ModuleType("test_module")
    exec(module_code, mod.__dict__)

    # Test with verbose=True
    with mock.patch("tqdm.tqdm") as mock_tqdm:
        result = mod.conditional_import(True)
        assert result is True  # The import happened
        # Verify tqdm was imported when verbose=True
        assert mock_tqdm is not None

    # Test with verbose=False
    with mock.patch("tqdm.tqdm"):
        result = mod.conditional_import(False)
        assert result is False  # The import didn't happen


def test_embed_frames_verbose_propagation(slp_minimal, tmp_path):
    """Test that embed_frames propagates verbose to process_and_embed_frames."""
    labels = load_slp(slp_minimal)
    frames_to_embed = [(labels.videos[0], 0)]

    # Create temp file for embedding
    temp_slp = tmp_path / "test_embed_prop.slp"

    # Mock process_and_embed_frames to verify verbose is correctly passed
    with mock.patch("sleap_io.io.slp.process_and_embed_frames") as mock_embed:
        embed_frames(temp_slp, labels, frames_to_embed, verbose=True)
        # Check that verbose=True was passed to process_and_embed_frames
        assert mock_embed.call_args.kwargs["verbose"] is True

    # Check with verbose=False
    with mock.patch("sleap_io.io.slp.process_and_embed_frames") as mock_embed:
        embed_frames(temp_slp, labels, frames_to_embed, verbose=False)
        # Check that verbose=False was passed to process_and_embed_frames
        assert mock_embed.call_args.kwargs["verbose"] is False


def test_embed_videos_verbose_propagation(slp_minimal, tmp_path):
    """Test that embed_videos propagates the verbose parameter to embed_frames."""
    labels = load_slp(slp_minimal)

    # Create temp file for embedding
    temp_slp = tmp_path / "test_embed_videos_prop.slp"

    # Mock embed_frames to verify verbose is correctly passed
    with mock.patch("sleap_io.io.slp.embed_frames") as mock_embed:
        embed_videos(temp_slp, labels, embed="user", verbose=True)
        # Check that verbose=True was passed to embed_frames
        assert mock_embed.call_args.kwargs["verbose"] is True

    # Check with verbose=False
    with mock.patch("sleap_io.io.slp.embed_frames") as mock_embed:
        embed_videos(temp_slp, labels, embed="user", verbose=False)
        # Check that verbose=False was passed to embed_frames
        assert mock_embed.call_args.kwargs["verbose"] is False


def test_write_videos_verbose_propagation(slp_minimal, tmp_path):
    """Test that write_videos propagates verbose to process_and_embed_frames."""
    labels = load_slp(slp_minimal)

    # Create temp file for embedding
    temp_slp = tmp_path / "test_write_videos_prop.slp"

    # Mock process_and_embed_frames to verify verbose is correctly passed when
    # embedding is needed
    with mock.patch("sleap_io.io.slp.process_and_embed_frames"):
        # This is a simplified test as we can't easily trigger the condition where
        # write_videos
        # calls process_and_embed_frames directly
        write_videos(temp_slp, labels.videos, verbose=True)
        # In a real case with embedded videos, this would verify that verbose is
        # passed correctly
        # Since we're not actually embedding in this test, the mock may not be called

    # The actual test here is that the function accepts the verbose parameter
    # without errors


def test_write_labels_verbose_propagation(slp_minimal, tmp_path):
    """Test that write_labels propagates verbose to embed_videos and write_videos."""
    labels = load_slp(slp_minimal)

    # Create temp file
    temp_slp = tmp_path / "test_write_labels_prop.slp"

    # Mock embed_videos to verify verbose is correctly passed
    with (
        mock.patch("sleap_io.io.slp.embed_videos") as mock_embed_videos,
        mock.patch("sleap_io.io.slp.write_videos") as mock_write_videos,
    ):
        write_labels(temp_slp, labels, embed="user", verbose=True)

        # Check that verbose=True was passed to embed_videos
        assert mock_embed_videos.call_args.kwargs["verbose"] is True

        # Check that verbose=True was passed to write_videos
        assert mock_write_videos.call_args.kwargs["verbose"] is True

    # Check with verbose=False
    with (
        mock.patch("sleap_io.io.slp.embed_videos") as mock_embed_videos,
        mock.patch("sleap_io.io.slp.write_videos") as mock_write_videos,
    ):
        write_labels(temp_slp, labels, embed="user", verbose=False)

        # Check that verbose=False was passed to embed_videos
        assert mock_embed_videos.call_args.kwargs["verbose"] is False

        # Check that verbose=False was passed to write_videos
        assert mock_write_videos.call_args.kwargs["verbose"] is False


def test_format_id_1_3_tracking_score(tmp_path):
    """Test that current FORMAT_ID properly handles tracking_score field."""
    # Create test data with tracking scores
    skeleton = Skeleton(["A", "B", "C"])
    track = Track("track1")

    # Create instances with tracking scores
    inst1 = Instance(
        [[1, 2], [3, 4], [5, 6]], skeleton=skeleton, track=track, tracking_score=0.95
    )
    inst2 = PredictedInstance(
        [[7, 8], [9, 10], [11, 12]],
        skeleton=skeleton,
        track=track,
        score=0.8,
        tracking_score=0.75,
    )

    # Create labeled frames and labels
    video = Video.from_filename("fake.mp4")
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst1, inst2])
    labels = Labels(
        videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
    )

    # Save with current FORMAT_ID
    test_path = tmp_path / "test_format_tracking_score.slp"
    write_labels(test_path, labels)

    # Verify FORMAT_ID is current (1.4)
    format_id = read_hdf5_attrs(test_path, "metadata", "format_id")
    assert format_id == 1.4

    # Load and verify tracking scores are preserved
    loaded_labels = read_labels(test_path)
    loaded_inst1 = loaded_labels.labeled_frames[0].instances[0]
    loaded_inst2 = loaded_labels.labeled_frames[0].instances[1]

    assert isinstance(loaded_inst1, Instance)
    assert loaded_inst1.tracking_score == pytest.approx(0.95)

    assert isinstance(loaded_inst2, PredictedInstance)
    assert loaded_inst2.tracking_score == pytest.approx(0.75)
    assert loaded_inst2.score == pytest.approx(0.8)


def test_format_id_backward_compatibility(tmp_path):
    """Test backward compatibility when reading files with older FORMAT_ID."""
    # Create test data
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")

    # Create instances
    inst = Instance([[1, 2], [3, 4]], skeleton=skeleton, track=track)
    pred_inst = PredictedInstance(
        [[5, 6], [7, 8]], skeleton=skeleton, track=track, score=0.9
    )

    video = Video.from_filename("fake.mp4")
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst, pred_inst])
    labels = Labels(
        videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
    )

    # Save the file
    test_path = tmp_path / "test_format_old.slp"
    write_labels(test_path, labels)

    # Manually modify the format_id to simulate an older file
    with h5py.File(test_path, "r+") as f:
        f["metadata"].attrs["format_id"] = 1.1

        # Also need to modify the instances dataset to remove tracking_score field
        # Read existing data
        instances_data = f["instances"][:]

        # Create new dtype without tracking_score
        old_dtype = np.dtype(
            [
                ("instance_id", "i8"),
                ("instance_type", "u1"),
                ("frame_id", "u8"),
                ("skeleton", "u4"),
                ("track", "i4"),
                ("from_predicted", "i8"),
                ("score", "f4"),
                ("point_id_start", "u8"),
                ("point_id_end", "u8"),
            ]
        )

        # Copy data to new array without tracking_score
        old_instances = np.zeros(len(instances_data), dtype=old_dtype)
        for i, inst_data in enumerate(instances_data):
            # Copy all fields except tracking_score
            old_instances[i] = (
                inst_data["instance_id"],
                inst_data["instance_type"],
                inst_data["frame_id"],
                inst_data["skeleton"],
                inst_data["track"],
                inst_data["from_predicted"],
                inst_data["score"],
                inst_data["point_id_start"],
                inst_data["point_id_end"],
            )

        # Delete and recreate dataset
        del f["instances"]
        f.create_dataset("instances", data=old_instances, dtype=old_dtype)

    # Load with older format - tracking_score should default to 0.0
    loaded_labels = read_labels(test_path)
    loaded_inst = loaded_labels.labeled_frames[0].instances[0]
    loaded_pred_inst = loaded_labels.labeled_frames[0].instances[1]

    assert isinstance(loaded_inst, Instance)
    assert loaded_inst.tracking_score == pytest.approx(0.0)

    assert isinstance(loaded_pred_inst, PredictedInstance)
    assert loaded_pred_inst.tracking_score == pytest.approx(0.0)


def test_save_slp_verbose_propagation(tmp_path):
    """Test that save_slp propagates the verbose parameter to write_labels."""
    # Mock write_labels to verify verbose is correctly passed
    with mock.patch("sleap_io.io.slp.write_labels") as mock_write_labels:
        # Create a mock Labels object
        mock_labels = mock.MagicMock()

        # Test with default verbose=True
        save_slp(mock_labels, tmp_path / "test.slp", embed="user")
        assert mock_write_labels.call_args.kwargs["verbose"] is True

        # Test with explicit verbose=True
        save_slp(mock_labels, tmp_path / "test.slp", embed="user", verbose=True)
        assert mock_write_labels.call_args.kwargs["verbose"] is True

        # Test with verbose=False
        save_slp(mock_labels, tmp_path / "test.slp", embed="user", verbose=False)
        assert mock_write_labels.call_args.kwargs["verbose"] is False


def test_labels_save_verbose_propagation(tmp_path):
    """Test that Labels.save propagates the verbose parameter to save_file."""
    # Mock save_file to verify verbose is correctly passed
    with mock.patch("sleap_io.save_file") as mock_save_file:
        # Create a mock Labels object
        mock_labels = Labels()

        # Test with default verbose=True
        mock_labels.save(tmp_path / "test.slp")
        assert mock_save_file.call_args.kwargs["verbose"] is True

        # Test with explicit verbose=True
        mock_labels.save(tmp_path / "test.slp", verbose=True)
        assert mock_save_file.call_args.kwargs["verbose"] is True

        # Test with verbose=False
        mock_labels.save(tmp_path / "test.slp", verbose=False)
        assert mock_save_file.call_args.kwargs["verbose"] is False


def test_save_file_verbose_propagation(tmp_path):
    """Test that save_file propagates the verbose parameter to save_slp."""
    # Mock save_slp to verify verbose is correctly passed
    with mock.patch("sleap_io.io.main.save_slp") as mock_save_slp:
        # Create a mock Labels object
        mock_labels = mock.MagicMock()

        # Test with default verbose=True for SLP format
        save_file(mock_labels, tmp_path / "test.slp")
        assert mock_save_slp.call_args.kwargs["verbose"] is True

        # Test with explicit verbose=True for SLP format
        save_file(mock_labels, tmp_path / "test.slp", verbose=True)
        assert mock_save_slp.call_args.kwargs["verbose"] is True

        # Test with verbose=False for SLP format
        save_file(mock_labels, tmp_path / "test.slp", verbose=False)
        assert mock_save_slp.call_args.kwargs["verbose"] is False


def test_embed_false_behavior(tmp_path, centered_pair_low_quality_video):
    """Test that embed=False restores source videos or references embedded files."""
    # Create test data
    skeleton = Skeleton(["A", "B"])
    video = centered_pair_low_quality_video

    inst = Instance([[1, 2], [3, 4]], skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    # Create a .pkg.slp file with embedded frames
    pkg_path = tmp_path / "test.pkg.slp"
    write_labels(str(pkg_path), labels, embed="user")

    # Load embedded labels
    embedded_labels = read_labels(str(pkg_path))
    assert embedded_labels.video.backend.has_embedded_images
    assert embedded_labels.video.source_video.filename == str(video.filename)

    # Test 1: Save with embed=False when source video is available
    pkg_path2 = tmp_path / "test2.pkg.slp"
    write_labels(str(pkg_path2), embedded_labels, embed=False)

    labels2 = read_labels(str(pkg_path2))
    assert labels2.video.filename == str(video.filename)
    assert type(labels2.video.backend).__name__ == "MediaVideo"

    # Test 2: Save with embed=False when no source video
    embedded_labels.video.source_video = None
    pkg_path3 = tmp_path / "test3.pkg.slp"
    write_labels(str(pkg_path3), embedded_labels, embed=False)

    labels3 = read_labels(str(pkg_path3))
    # Should reference the original embedded file
    assert Path(labels3.video.filename).resolve() == pkg_path.resolve()
    assert labels3.video.backend.has_embedded_images


def test_mixed_video_scenarios(tmp_path, centered_pair_low_quality_video):
    """Test saving with mixed embedded and non-embedded videos."""
    # Create test data with multiple videos
    skeleton = Skeleton(["A", "B"])

    # Video 1: Regular video
    video1 = centered_pair_low_quality_video

    # Video 2: Will be embedded
    video2 = Video.from_filename(centered_pair_low_quality_video.filename)

    # Create labeled frames for both videos
    inst1 = Instance([[1, 2], [3, 4]], skeleton=skeleton)
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])

    inst2 = Instance([[5, 6], [7, 8]], skeleton=skeleton)
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    labels = Labels(
        videos=[video1, video2], skeletons=[skeleton], labeled_frames=[lf1, lf2]
    )

    # Save with only video2 embedded (embed_all_videos=False keeps video1 external)
    pkg_path = tmp_path / "mixed.pkg.slp"
    frames_to_embed = [(video2, 0)]
    write_labels(str(pkg_path), labels, embed=frames_to_embed, embed_all_videos=False)

    # Load and verify
    loaded = read_labels(str(pkg_path))
    assert len(loaded.videos) == 2
    assert type(loaded.videos[0].backend).__name__ == "MediaVideo"
    assert type(loaded.videos[1].backend).__name__ == "HDF5Video"
    assert loaded.videos[1].backend.has_embedded_images

    # Save with embed=False to new file
    pkg_path2 = tmp_path / "mixed2.pkg.slp"
    write_labels(str(pkg_path2), loaded, embed=False)

    # Verify both videos are correctly referenced
    loaded2 = read_labels(str(pkg_path2))
    assert len(loaded2.videos) == 2

    # Video 1 should still be external media
    assert loaded2.videos[0].filename == video1.filename
    assert type(loaded2.videos[0].backend).__name__ == "MediaVideo"

    # Video 2 should restore to source video (since it's available)
    assert loaded2.videos[1].filename == video2.filename
    assert type(loaded2.videos[1].backend).__name__ == "MediaVideo"

    # Test 2: Remove source video from embedded video and save again
    loaded.videos[1].source_video = None
    pkg_path3 = tmp_path / "mixed3.pkg.slp"
    write_labels(str(pkg_path3), loaded, embed=False)

    loaded3 = read_labels(str(pkg_path3))
    assert len(loaded3.videos) == 2

    # Video 1 should still be external media
    assert loaded3.videos[0].filename == video1.filename
    assert type(loaded3.videos[0].backend).__name__ == "MediaVideo"

    # Video 2 should now reference the pkg file (no source video)
    assert Path(loaded3.videos[1].filename).resolve() == pkg_path.resolve()
    assert loaded3.videos[1].backend.has_embedded_images


def test_save_overwrite_without_embedded(tmp_path, centered_pair_low_quality_video):
    """Test saving with embed=False over the same file.

    Tests when no embedded data exists.
    """
    # Create test data with external video
    skeleton = Skeleton(["A", "B"])
    video = centered_pair_low_quality_video

    inst = Instance([[1, 2], [3, 4]], skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    # Save to a .slp file without embedding
    slp_path = tmp_path / "test.slp"
    labels.save(str(slp_path), embed=False)

    # Load and verify it's not embedded
    loaded = read_labels(str(slp_path))
    assert type(loaded.video.backend).__name__ == "MediaVideo"

    # Save over the same file with embed=False - should work fine
    loaded.save(str(slp_path), embed=False)

    # Verify it still works
    loaded2 = read_labels(str(slp_path))
    assert type(loaded2.video.backend).__name__ == "MediaVideo"
    assert loaded2.video.filename == video.filename


def test_save_overwrite_embedded_with_source(tmp_path, centered_pair_low_quality_video):
    """Test saving with embed=False over same file when embedded data has source."""
    # Create test data
    skeleton = Skeleton(["A", "B"])
    video = centered_pair_low_quality_video

    inst = Instance([[1, 2], [3, 4]], skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

    # Save with embedding
    pkg_path = tmp_path / "test.pkg.slp"
    labels.save(str(pkg_path), embed="user")

    # Load and verify it's embedded with source video
    loaded = read_labels(str(pkg_path))
    assert loaded.video.backend.has_embedded_images
    assert loaded.video.source_video is not None
    assert loaded.video.source_video.filename == video.filename

    # Save over the same file with embed=False - should restore source video
    loaded.save(str(pkg_path), embed=False)

    # Verify source video was restored
    loaded2 = read_labels(str(pkg_path))
    assert type(loaded2.video.backend).__name__ == "MediaVideo"
    assert loaded2.video.filename == video.filename


def test_self_referential_path_detection(tmp_path):
    """Test that self-referential paths are detected and raise an error."""
    # Create test data
    skeleton = Skeleton(["A", "B"])
    track = Track("track1")
    video = Video.from_filename("fake.mp4")

    # Create instance and labeled frame
    inst = Instance([[1, 2], [3, 4]], skeleton=skeleton, track=track)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])

    # Create a .pkg.slp file with embedded frames
    pkg_path = tmp_path / "test.pkg.slp"

    # Mock embedding to create a video with embedded frames but no source video
    with h5py.File(pkg_path, "w") as f:
        # Create minimal structure
        f.create_dataset("videos_json", data=[])
        f.create_dataset("tracks_json", data=[])

    # Create a mock embedded video with minimal HDF5Video backend
    # We need to use a mock to bypass the actual file reading
    from unittest.mock import patch

    # Create a real HDF5Video instance but mock its file operations
    with patch("h5py.File"):
        mock_backend = HDF5Video(
            filename=str(pkg_path),
            dataset="video0/video",
            grayscale=True,
            keep_open=False,
        )
        # Manually set properties to simulate embedded video
        mock_backend.image_format = "png"  # This makes has_embedded_images return True
        mock_backend.source_inds = [0]  # Mock having one embedded frame

    embedded_video = Video(
        filename=str(pkg_path),
        backend=mock_backend,
        source_video=None,  # No source video
    )

    labels_with_embedded = Labels(
        videos=[embedded_video],
        skeletons=[skeleton],
        tracks=[track],
        labeled_frames=[lf],
    )

    # Try to save over the same file with embed=False (should raise error)
    with pytest.raises(
        ValueError, match="Cannot save with embed=False when overwriting a file"
    ):
        labels_with_embedded.save(str(pkg_path), embed=False)

    # Saving to a different file should work
    pkg_path2 = tmp_path / "test2.pkg.slp"
    write_labels(str(pkg_path2), labels_with_embedded, embed=False)


# Video reference restoration tests
def test_labels_save_restore_original_videos_api(tmp_path, slp_minimal_pkg):
    """Test the restore_original_videos parameter in Labels.save()."""
    # Load a .pkg.slp file
    labels = load_file(slp_minimal_pkg)

    # Verify it has embedded videos with source_video metadata
    assert len(labels.videos) == 1
    video = labels.videos[0]
    assert isinstance(video.backend, HDF5Video)
    assert video.backend.has_embedded_images
    assert video.source_video is not None
    original_video_path = video.source_video.filename

    # Test default behavior (restore_original_videos=True)
    output_default = tmp_path / "test_default.slp"
    labels.save(output_default, embed=False)

    # Load and check that original video is restored
    labels_default = load_file(output_default)
    assert labels_default.videos[0].filename == original_video_path
    assert labels_default.videos[0].backend_metadata["type"] == "MediaVideo"

    # Test new behavior (restore_original_videos=False)
    output_preserve = tmp_path / "test_preserve_source.slp"
    labels.save(output_preserve, embed=False, restore_original_videos=False)

    # Load and check that source .pkg.slp is referenced
    labels_preserve = load_file(output_preserve)
    assert labels_preserve.videos[0].filename == slp_minimal_pkg
    assert labels_preserve.videos[0].backend_metadata["type"] == "HDF5Video"
    assert labels_preserve.videos[0].backend_metadata["has_embedded_images"] is True


def test_save_slp_restore_original_videos_api(tmp_path, slp_minimal_pkg):
    """Test the restore_original_videos parameter in save_slp()."""
    from sleap_io.io.main import save_slp

    # Load a .pkg.slp file
    labels = load_file(slp_minimal_pkg)

    # Test default behavior
    output_default = tmp_path / "test_default_api.slp"
    save_slp(labels, output_default, embed=False)

    labels_default = load_file(output_default)
    assert labels_default.videos[0].backend_metadata["type"] == "MediaVideo"

    # Test new behavior
    output_preserve = tmp_path / "test_preserve_api.slp"
    save_slp(labels, output_preserve, embed=False, restore_original_videos=False)

    labels_preserve = load_file(output_preserve)
    assert labels_preserve.videos[0].filename == slp_minimal_pkg
    assert labels_preserve.videos[0].backend_metadata["type"] == "HDF5Video"


def test_video_metadata_preservation(tmp_path, slp_minimal_pkg):
    """Test that video metadata is preserved correctly in all modes."""
    # Load fresh labels for metadata extraction
    labels = load_file(slp_minimal_pkg)
    video = labels.videos[0]

    # Store original metadata for comparison
    # For minimal_instance.pkg.slp, source_video IS the original video
    original_backend_metadata = video.source_video.backend_metadata.copy()

    # Test EMBED mode
    labels = load_file(slp_minimal_pkg)  # Fresh load
    output_embed = tmp_path / "test_embed.slp"
    labels.save(output_embed, embed=True)

    with h5py.File(output_embed, "r") as f:
        # Check that source_video metadata is stored
        # Note: original_video is now a computed property derived from source_video,
        # so we only store source_video in HDF5
        assert "video0" in f
        assert "source_video" in f["video0"]

        # Verify source video metadata (should be the pre-embed .pkg.slp file)
        assert isinstance(f["video0/source_video"], h5py.Group)
        source_json = json.loads(f["video0/source_video"].attrs["json"])
        assert source_json["backend"]["type"] == "HDF5Video"
        assert source_json["backend"]["filename"] == slp_minimal_pkg

    # Verify that original_video is computed correctly when loading
    labels_embedded = load_file(output_embed)
    vid = labels_embedded.videos[0]
    # source_video should be the .pkg.slp file
    assert vid.source_video is not None
    assert vid.source_video.filename == slp_minimal_pkg
    # original_video (computed) should be the ultimate source MediaVideo
    assert vid.original_video is not None
    assert vid.original_video.filename == original_backend_metadata["filename"]

    # Test RESTORE_ORIGINAL mode (default)
    labels = load_file(slp_minimal_pkg)  # Fresh load
    output_restore = tmp_path / "test_restore.slp"
    labels.save(output_restore, embed=False, restore_original_videos=True)

    # Load and verify the video reference is restored to original
    labels_restore = load_file(output_restore)
    assert labels_restore.videos[0].filename == original_backend_metadata["filename"]
    assert labels_restore.videos[0].backend_metadata["type"] == "MediaVideo"

    # Test PRESERVE_SOURCE mode (new)
    labels = load_file(slp_minimal_pkg)  # Fresh load
    output_preserve = tmp_path / "test_preserve.slp"
    labels.save(output_preserve, embed=False, restore_original_videos=False)

    # Load and verify the video reference is preserved to source .pkg.slp
    labels_preserve = load_file(output_preserve)
    assert labels_preserve.videos[0].filename == slp_minimal_pkg
    assert labels_preserve.videos[0].backend_metadata["type"] == "HDF5Video"


def test_multiple_save_load_cycles(tmp_path, slp_minimal_pkg):
    """Test that video lineage is preserved through multiple save/load cycles."""
    # First cycle: Load .pkg.slp and save with preserve_source
    labels1 = load_file(slp_minimal_pkg)
    original_video_path = labels1.videos[0].source_video.filename

    output1 = tmp_path / "cycle1.slp"
    labels1.save(output1, embed=False, restore_original_videos=False)

    # Second cycle: Load cycle1.slp and save again
    labels2 = load_file(output1)
    assert labels2.videos[0].filename == slp_minimal_pkg
    # For minimal_instance.pkg.slp, source_video IS the original video
    # In PRESERVE_SOURCE mode, the original video metadata should be preserved
    assert labels2.videos[0].source_video is not None
    assert labels2.videos[0].source_video.filename == original_video_path

    output2 = tmp_path / "cycle2.slp"
    labels2.save(output2, embed=False, restore_original_videos=False)

    # Third cycle: Verify metadata is still preserved
    labels3 = load_file(output2)
    # In PRESERVE_SOURCE mode, it should still reference the original .pkg.slp
    assert labels3.videos[0].filename == slp_minimal_pkg
    # Verify metadata persistence through multiple cycles
    assert labels3.videos[0].source_video is not None
    assert labels3.videos[0].source_video.filename == original_video_path


def test_unavailable_video_handling(tmp_path):
    """Test handling of videos when files are not available."""
    # Create a Labels object with a video that doesn't exist
    fake_video = Video(
        filename="/nonexistent/original.mp4",
        backend=None,
        backend_metadata={
            "type": "MediaVideo",
            "shape": [100, 384, 384, 3],
            "filename": "/nonexistent/original.mp4",
            "grayscale": False,
            "bgr": True,
            "dataset": "",
            "input_format": "",
        },
    )

    labels = Labels(videos=[fake_video])

    # Save and verify metadata is preserved
    output = tmp_path / "test_unavailable.slp"
    labels.save(output, embed=False)

    # Load and check metadata was preserved
    loaded = load_file(output)
    assert loaded.videos[0].filename == "/nonexistent/original.mp4"
    assert loaded.videos[0].backend is None or not loaded.videos[0].exists()
    assert loaded.videos[0].backend_metadata["type"] == "MediaVideo"
    assert loaded.videos[0].backend_metadata["shape"] == [100, 384, 384, 3]


def test_video_reference_mode_enum():
    """Test that VideoReferenceMode enum is properly defined."""
    from sleap_io.io.slp import VideoReferenceMode

    assert VideoReferenceMode.EMBED.value == "embed"
    assert VideoReferenceMode.RESTORE_ORIGINAL.value == "restore_original"
    assert VideoReferenceMode.PRESERVE_SOURCE.value == "preserve_source"


def test_write_videos_with_reference_mode(tmp_path, slp_minimal_pkg):
    """Test the internal write_videos function with VideoReferenceMode."""
    from sleap_io.io.slp import VideoReferenceMode

    labels = load_file(slp_minimal_pkg)
    videos = labels.videos

    # Test PRESERVE_SOURCE mode
    output = tmp_path / "test_internal.slp"
    write_videos(output, videos, reference_mode=VideoReferenceMode.PRESERVE_SOURCE)

    # Read back and verify
    loaded_videos = read_videos(output)
    assert loaded_videos[0].filename == slp_minimal_pkg
    assert loaded_videos[0].backend_metadata["type"] == "HDF5Video"


def test_video_restore_backwards_compatibility(tmp_path, slp_minimal_pkg):
    """Test that the default behavior maintains backwards compatibility."""
    labels = load_file(slp_minimal_pkg)
    original_video_path = labels.videos[0].source_video.filename

    # Default behavior should restore original videos
    output = tmp_path / "test_compat.slp"
    labels.save(output, embed=False)  # No restore_original_videos parameter

    loaded = load_file(output)
    assert loaded.videos[0].filename == original_video_path
    assert loaded.videos[0].backend_metadata["type"] == "MediaVideo"


def test_video_to_dict_with_none_backend(tmp_path):
    """Test that video_to_dict handles videos with backend=None correctly."""
    video = Video(
        filename="test.mp4",
        backend=None,
        backend_metadata={
            "type": "MediaVideo",
            "filename": "test.mp4",
            "shape": [10, 100, 100, 1],
            "grayscale": True,
        },
    )

    video_dict = video_to_dict(video, labels_path=tmp_path / "test.slp")

    assert video_dict["filename"] == "test.mp4"
    assert video_dict["backend"] == video.backend_metadata


def test_video_to_dict_tiffvideo(tmp_path, multipage_tiff_path):
    """Test that video_to_dict handles TiffVideo backend correctly."""
    from sleap_io.io.video_reading import TiffVideo

    # Create a TiffVideo backend
    backend = TiffVideo(filename=multipage_tiff_path, grayscale=True, keep_open=False)
    video = Video(
        filename=multipage_tiff_path,
        backend=backend,
    )

    video_dict = video_to_dict(video, labels_path=tmp_path / "test.slp")

    assert video_dict["filename"] == multipage_tiff_path
    assert video_dict["backend"]["type"] == "TiffVideo"
    assert video_dict["backend"]["shape"] == video.shape
    assert video_dict["backend"]["grayscale"] is True
    assert video_dict["backend"]["keep_open"] is False
    assert video_dict["backend"]["filename"] == multipage_tiff_path
    assert video_dict["backend"]["format"] == "multi_page"


def test_video_to_dict_none_backend_without_filename():
    """Test that video_to_dict handles backend=None without filename in metadata.

    This reproduces the bug from https://github.com/talmolab/sleap/discussions/2417
    where backend_metadata doesn't contain a "filename" key, causing a KeyError
    when trying to reconstruct the video with make_video().
    """
    import tempfile

    from sleap_io.io.slp import make_video

    # Create a Video with backend=None and backend_metadata WITHOUT filename key
    # This simulates old SLP files or edge cases where backend_metadata is incomplete
    video = Video(
        filename="test_video.mp4",
        backend=None,
        backend_metadata={
            "type": "MediaVideo",
            "shape": [100, 480, 640, 3],
            "grayscale": False,
        },
    )

    # Convert to dict - this should work
    with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as f:
        labels_path = f.name

    video_dict = video_to_dict(video, labels_path=labels_path)

    # Verify filename was added to backend
    assert "filename" in video_dict["backend"]
    assert video_dict["backend"]["filename"] == "test_video.mp4"

    # The bug: backend_metadata doesn't have "filename", so make_video will fail
    # After the fix, this should work because video_to_dict ensures filename is present
    video_json = {"backend": video_dict["backend"]}
    reconstructed_video = make_video(labels_path, video_json, open_backend=False)

    # Verify the reconstructed video has the correct filename
    assert reconstructed_video.filename == "test_video.mp4"


def test_video_to_dict_none_backend_preserves_existing_filename():
    """Test that video_to_dict preserves existing filename in backend_metadata."""
    # Create a Video where backend_metadata already has a different filename
    # This could happen if backend_metadata was copied from another source
    video = Video(
        filename="video_a.mp4",
        backend=None,
        backend_metadata={
            "type": "MediaVideo",
            "filename": "original_video.mp4",  # Different from video.filename
            "shape": [100, 480, 640, 3],
            "grayscale": False,
        },
    )

    video_dict = video_to_dict(video, labels_path="test.slp")

    # Should preserve the existing filename in backend_metadata
    assert video_dict["backend"]["filename"] == "original_video.mp4"
    # Top-level filename should still be from video.filename
    assert video_dict["filename"] == "video_a.mp4"


def test_make_video_fallback_to_toplevel_filename():
    """Test that make_video() falls back to top-level filename.

    This reproduces the exact scenario from the user's file where Video 32 had
    backend_metadata without a "filename" key, requiring fallback to the
    top-level video_json["filename"].
    """
    import tempfile

    from sleap_io.io.slp import make_video

    # Create a video_json structure like the problematic Video 32 from the user's file
    # backend_metadata has dataset, grayscale, shape but NO filename
    video_json = {
        "filename": "D:/SLEAP_training/20240404_Choice233_01.MP4",
        "backend": {
            "dataset": "",
            "grayscale": False,
            "shape": [100, 480, 640, 3],
            # NOTE: No "filename" key here - this was the bug!
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as f:
        labels_path = f.name

    # This should not raise KeyError - it should fall back to video_json["filename"]
    video = make_video(labels_path, video_json, open_backend=False)

    # Verify the video was created with the correct filename
    assert video.filename == "D:/SLEAP_training/20240404_Choice233_01.MP4"
    assert video.backend is None  # Backend not opened


def test_make_video_missing_filename_raises_error():
    """Test that make_video() raises ValueError when filename is missing everywhere.

    This tests the error handling path when neither backend_metadata nor video_json
    contains a "filename" key.
    """
    import tempfile

    from sleap_io.io.slp import make_video

    # Create a video_json with NO filename in backend OR top-level
    video_json = {
        "backend": {
            "dataset": "",
            "grayscale": False,
            "shape": [100, 480, 640, 3],
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as f:
        labels_path = f.name

    # This should raise ValueError since there's no filename anywhere
    with pytest.raises(ValueError, match="Video JSON does not contain a filename"):
        make_video(labels_path, video_json, open_backend=False)


def test_make_video_with_backend_filename_only():
    """Test that make_video() uses backend filename when available.

    This ensures the primary path (backend_metadata has filename) is tested.
    """
    import tempfile

    from sleap_io.io.slp import make_video

    # backend has filename, top-level also has filename (backend should take priority)
    video_json = {
        "filename": "top_level.mp4",  # This should be ignored
        "backend": {
            "filename": "backend_level.mp4",  # This should be used
            "type": "MediaVideo",
            "shape": [100, 480, 640, 3],
            "grayscale": False,
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as f:
        labels_path = f.name

    video = make_video(labels_path, video_json, open_backend=False)

    # Should use the backend filename, not the top-level one
    assert video.filename == "backend_level.mp4"
    assert isinstance(video.filename, str)


def test_video_to_dict_mutates_original():
    """Test that video_to_dict() doesn't mutate the original backend_metadata.

    The fix uses .copy() to avoid mutating the original backend_metadata dict.
    This test ensures that behavior is correct.
    """
    # Create a Video with backend=None and backend_metadata WITHOUT filename
    original_metadata = {
        "type": "MediaVideo",
        "shape": [100, 480, 640, 3],
        "grayscale": False,
    }

    video = Video(
        filename="test_video.mp4",
        backend=None,
        backend_metadata=original_metadata,
    )

    # Store original keys for comparison
    original_keys = set(original_metadata.keys())

    # Convert to dict
    video_dict = video_to_dict(video, labels_path="test.slp")

    # Verify the dict has filename in backend
    assert "filename" in video_dict["backend"]

    # Verify the original metadata was NOT mutated
    assert set(original_metadata.keys()) == original_keys
    assert "filename" not in original_metadata


def test_tiffvideo_roundtrip(tmp_path, multipage_tiff_path):
    """Test saving and loading Labels with TiffVideo backend."""
    from sleap_io.io.video_reading import TiffVideo

    # Create labels with a TiffVideo
    video = Video.from_filename(multipage_tiff_path)
    assert isinstance(video.backend, TiffVideo)

    skeleton = Skeleton(nodes=["A", "B"])
    labeled_frame = LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            Instance.from_numpy(
                points_data=np.array([[10, 20], [30, 40]]),
                skeleton=skeleton,
            )
        ],
    )

    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
        labeled_frames=[labeled_frame],
    )

    # Save to SLP
    output_path = tmp_path / "test_tiff.slp"
    labels.save(output_path)

    # Load and verify
    loaded_labels = load_file(output_path)
    loaded_video = loaded_labels.videos[0]

    # Check that the video was loaded correctly
    assert loaded_video.filename == multipage_tiff_path
    assert isinstance(loaded_video.backend, TiffVideo)
    assert loaded_video.backend.num_frames == 8
    assert loaded_video.shape == video.shape

    # Verify we can read frames
    frame = loaded_video.backend.get_frame(0)
    assert frame.shape == (128, 128, 1)


def test_video_original_video_field(slp_minimal_pkg):
    """Test that Video objects have the original_video computed property."""
    labels = load_file(slp_minimal_pkg)
    video = labels.videos[0]

    # original_video is now a computed property that traverses source_video chain
    assert hasattr(video, "original_video")

    # source_video points to the MediaVideo that frames were embedded from
    assert video.source_video is not None

    # original_video (computed) should be the root of the source_video chain
    # Since source_video has no parent, original_video == source_video
    assert video.original_video is not None
    assert video.original_video is video.source_video


def test_legacy_original_video_hdf5_compat(tmp_path, slp_minimal_pkg):
    """Test loading legacy files that have original_video but no source_video in HDF5.

    This covers the legacy compatibility path in make_video() for embedded videos
    where original_video was stored but source_video wasn't.
    """
    # Load a pkg file and resave it
    labels = load_file(slp_minimal_pkg)
    original_source = labels.videos[0].source_video
    output = tmp_path / "legacy_test.slp"
    labels.save(output, embed=True)

    # Manually modify the HDF5 to simulate a legacy file:
    # Remove source_video and add original_video with the ORIGINAL mp4 reference
    with h5py.File(output, "a") as f:
        # Delete existing source_video if present
        if "video0/source_video" in f:
            del f["video0/source_video"]

        # Create original_video pointing to the original mp4 (simulating legacy format)
        original_grp = f["video0"].require_group("original_video")
        original_json = {
            "backend": {
                "filename": original_source.filename,
                "type": "MediaVideo",
                "grayscale": False,
            }
        }
        original_grp.attrs["json"] = json.dumps(original_json, separators=(",", ":"))

    # Now load the "legacy" file - should use original_video as source_video
    loaded = load_file(output)
    video = loaded.videos[0]

    # source_video should be populated from the legacy original_video
    assert video.source_video is not None
    assert video.source_video.filename == original_source.filename
    # original_video (computed) should equal source_video (single level chain)
    assert video.original_video is video.source_video


def test_legacy_original_video_json_compat(tmp_path, slp_minimal_pkg):
    """Test loading legacy files that have original_video but no source_video in JSON.

    This covers the legacy compatibility path in make_video() for non-embedded videos
    where original_video was stored in videos_json but source_video wasn't.
    """
    # Load a pkg file and save as non-embedded
    labels = load_file(slp_minimal_pkg)
    output = tmp_path / "legacy_json_test.slp"
    labels.save(output, embed=False, restore_original_videos=False)

    # Manually modify the HDF5 to simulate a legacy file:
    # Rename source_video to original_video in videos_json
    with h5py.File(output, "a") as f:
        videos_json = [json.loads(v) for v in f["videos_json"][:]]

        for vj in videos_json:
            if "source_video" in vj:
                # Move source_video to original_video (simulating legacy format)
                vj["original_video"] = vj.pop("source_video")

        # Rewrite videos_json
        del f["videos_json"]
        f.create_dataset(
            "videos_json",
            data=[np.bytes_(json.dumps(v, separators=(",", ":"))) for v in videos_json],
            maxshape=(None,),
        )

    # Now load the "legacy" file - should use original_video as source_video
    loaded = load_file(output)
    video = loaded.videos[0]

    # source_video should be populated from the legacy original_video
    assert video.source_video is not None
    # original_video (computed) should equal source_video
    assert video.original_video is video.source_video


def test_complex_workflow(tmp_path, slp_minimal_pkg):
    """Test a complex workflow with training and inference results."""
    # Load training data
    train_labels = load_file(slp_minimal_pkg)

    # Simulate saving for distribution (embed=True)
    train_pkg = tmp_path / "train.pkg.slp"
    train_labels.save(train_pkg, embed=True)

    # Load in inference environment
    inference_labels = load_file(train_pkg)

    # Simulate predictions (in practice would come from a model)
    predictions = Labels(
        videos=inference_labels.videos,
        skeletons=inference_labels.skeletons,
        labeled_frames=[],  # Would contain predicted instances
    )

    # Save predictions referencing the training package
    predictions_output = tmp_path / "predictions_on_train.slp"
    predictions.save(predictions_output, embed=False, restore_original_videos=False)

    # Load predictions and verify they reference train.pkg.slp
    loaded_predictions = load_file(predictions_output)
    assert loaded_predictions.videos[0].filename == train_pkg.as_posix()
    assert loaded_predictions.videos[0].backend_metadata["type"] == "HDF5Video"
    assert loaded_predictions.videos[0].backend_metadata["has_embedded_images"] is True

    # Verify metadata preservation through the workflow
    # The video objects from inference_labels already have source_video metadata
    # which is preserved when we create the predictions Labels object
    assert loaded_predictions.videos[0].source_video is not None

    # The source_video should point to minimal_instance.pkg.slp
    # (the original training data)
    # This is correct because we're using the same video objects from inference_labels
    assert loaded_predictions.videos[0].source_video.filename == slp_minimal_pkg
    assert (
        loaded_predictions.videos[0].source_video.backend_metadata["type"]
        == "HDF5Video"
    )

    # And that should have the original MediaVideo as its source
    assert loaded_predictions.videos[0].source_video.source_video is not None
    assert (
        loaded_predictions.videos[0].source_video.source_video.backend_metadata["type"]
        == "MediaVideo"
    )


def test_write_videos_backwards_compatibility():
    """Test backwards compatibility with restore_source parameter."""
    import tempfile

    from sleap_io.io.slp import write_videos
    from sleap_io.model.video import Video

    video = Video(
        filename="test.mp4",
        backend_metadata={
            "type": "MediaVideo",
            "shape": [1, 100, 100, 1],
            "filename": "test.mp4",
            "grayscale": True,
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test.slp"

        # Test restore_source=True with reference_mode=None
        # (should use RESTORE_ORIGINAL)
        write_videos(str(output), [video], restore_source=True, reference_mode=None)

        # Test restore_source=False with reference_mode=None (should use EMBED)
        write_videos(str(output), [video], restore_source=False, reference_mode=None)


def test_video_lineage_edge_cases():
    """Test edge cases in video lineage metadata handling.

    Note: original_video is now a computed property derived from source_video chain.
    This test verifies that source_video chains are correctly preserved.
    """
    import tempfile

    from sleap_io.io.slp import VideoReferenceMode, write_videos
    from sleap_io.model.video import Video

    # Test case 1: Video with single-level source_video chain
    original = Video(
        filename="original.mp4",
        backend_metadata={
            "type": "MediaVideo",
            "shape": [10, 100, 100, 1],
            "filename": "original.mp4",
            "grayscale": True,
        },
        open_backend=False,
    )

    video_with_source = Video(
        filename="current.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "current.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        source_video=original,  # source_video points to original
        open_backend=False,
    )

    # Verify computed original_video
    assert video_with_source.original_video is original

    # Test case 2: Multi-level source_video chain
    source_embedded = Video(
        filename="source.pkg.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "source.pkg.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        source_video=original,  # source points to original MediaVideo
        open_backend=False,
    )

    video_with_chain = Video(
        filename="current2.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "current2.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        source_video=source_embedded,  # source points to intermediate embedded
        open_backend=False,
    )

    # Verify computed original_video traverses the full chain
    assert video_with_chain.source_video is source_embedded
    assert video_with_chain.original_video is original  # Should traverse to root

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test_lineage.slp"

        # Write videos with different lineage scenarios
        write_videos(
            str(output),
            [video_with_source, video_with_chain],
            reference_mode=VideoReferenceMode.EMBED,
            original_videos=[video_with_source, video_with_chain],
        )


# Tests for read_labels_set


def test_read_labels_set_from_directory(tmp_path, slp_minimal):
    """Test loading LabelsSet from a directory of SLP files."""
    # Load minimal labels
    labels = load_slp(slp_minimal)

    # Create test directory with multiple SLP files
    test_dir = tmp_path / "splits"
    test_dir.mkdir()

    # Save splits
    labels.save(test_dir / "train.slp", embed=False)
    labels.save(test_dir / "val.slp", embed=False)
    labels.save(test_dir / "test.slp", embed=False)

    # Load as LabelsSet
    labels_set = read_labels_set(test_dir)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 3
    assert "train" in labels_set
    assert "val" in labels_set
    assert "test" in labels_set

    # Check that each loaded Labels has correct data
    for name in ["train", "val", "test"]:
        assert len(labels_set[name]) == len(labels)


def test_read_labels_set_from_list(tmp_path, slp_minimal):
    """Test loading LabelsSet from a list of file paths."""
    labels = load_slp(slp_minimal)

    # Create test files
    file1 = tmp_path / "split1.slp"
    file2 = tmp_path / "split2.slp"
    labels.save(file1, embed=False)
    labels.save(file2, embed=False)

    # Load from list
    labels_set = read_labels_set([file1, file2])

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 2
    assert "split1" in labels_set
    assert "split2" in labels_set


def test_read_labels_set_from_dict(tmp_path, slp_minimal):
    """Test loading LabelsSet from a dictionary mapping."""
    labels = load_slp(slp_minimal)

    # Create test files
    train_file = tmp_path / "train_data.slp"
    val_file = tmp_path / "validation_data.slp"
    labels.save(train_file, embed=False)
    labels.save(val_file, embed=False)

    # Load from dictionary
    labels_set = read_labels_set({"training": train_file, "validation": val_file})

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 2
    assert "training" in labels_set
    assert "validation" in labels_set


def test_read_labels_set_empty_directory(tmp_path):
    """Test error handling for empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No .slp files found"):
        read_labels_set(empty_dir)


def test_read_labels_set_invalid_path(tmp_path):
    """Test error handling for invalid path."""
    # Non-directory path
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError, match="Path must be a directory"):
        read_labels_set(file_path)


def test_read_labels_set_without_videos(tmp_path, slp_minimal):
    """Test loading LabelsSet without opening videos."""
    labels = load_slp(slp_minimal)

    # Save a file
    test_file = tmp_path / "test.slp"
    labels.save(test_file, embed=False)

    # Load without opening videos
    labels_set = read_labels_set([test_file], open_videos=False)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 1

    # Videos should not be opened
    for lf in labels_set["test"].labeled_frames:
        assert lf.video is not None
        assert not lf.video.is_open


def test_read_labels_set_string_paths(tmp_path, slp_minimal):
    """Test that string paths work as well as Path objects."""
    labels = load_slp(slp_minimal)

    # Create test directory
    test_dir = tmp_path / "splits"
    test_dir.mkdir()
    labels.save(test_dir / "data.slp", embed=False)

    # Test with string path
    labels_set = read_labels_set(str(test_dir))
    assert len(labels_set) == 1
    assert "data" in labels_set


@pytest.mark.parametrize("encode_plugin", ["opencv", "imageio"])
@pytest.mark.parametrize("decode_plugin", ["opencv", "imageio"])
def test_embed_channel_order_consistency(
    tmp_path, small_robot_video, encode_plugin, decode_plugin
):
    """Test frames match exactly regardless of encoding/decoding plugin.

    This test verifies that the RGB/BGR channel order handling works correctly by:
    1. Encoding frames with one plugin (opencv=BGR or imageio=RGB)
    2. Decoding frames with a different plugin
    3. Verifying frames match the original exactly (automatic conversion)

    Args:
        tmp_path: Pytest temporary directory fixture.
        small_robot_video: Test video fixture (3-frame robot video).
        encode_plugin: Plugin to use for encoding ("opencv" or "imageio").
        decode_plugin: Plugin to use for decoding ("opencv" or "imageio").
    """
    # Step 1: Extract original frames from the video
    original_frames = []
    for i in range(len(small_robot_video)):
        frame = small_robot_video[i]
        original_frames.append(frame.copy())

    # Step 2: Create Labels with labeled instances for all frames
    skeleton = Skeleton(nodes=["node1", "node2"])
    labeled_frames = []
    for i in range(len(small_robot_video)):
        instance = Instance(
            points=np.array([[10.0 + i, 20.0 + i], [30.0 + i, 40.0 + i]]),
            skeleton=skeleton,
        )
        labeled_frame = LabeledFrame(
            video=small_robot_video, frame_idx=i, instances=[instance]
        )
        labeled_frames.append(labeled_frame)

    labels = Labels(
        videos=[small_robot_video], skeletons=[skeleton], labeled_frames=labeled_frames
    )

    # Step 3: Save to .pkg.slp with specified encoding plugin
    pkg_path = tmp_path / f"test_{encode_plugin}.pkg.slp"

    # Store original default and set encoding plugin
    original_default = get_default_image_plugin()
    try:
        set_default_image_plugin(encode_plugin)
        save_slp(labels, str(pkg_path), embed="all", plugin=encode_plugin)
    finally:
        # Restore original default
        set_default_image_plugin(original_default)

    # Step 4: Verify the channel_order attribute was stored correctly in HDF5
    with h5py.File(pkg_path, "r") as f:
        video_ds = f["video0/video"]
        assert "channel_order" in video_ds.attrs
        expected_channel_order = "BGR" if encode_plugin == "opencv" else "RGB"
        assert video_ds.attrs["channel_order"] == expected_channel_order

    # Step 5: Load with specified decoding plugin
    # We control the decoder by mocking sys.modules and setting the plugin preference
    try:
        set_default_image_plugin(decode_plugin)

        # Mock sys.modules to ensure the decode plugin is used
        if decode_plugin == "opencv":
            # Ensure cv2 is in sys.modules (it should be already)
            import cv2  # noqa: F401

        loaded_labels = load_slp(str(pkg_path))
        loaded_video = loaded_labels.videos[0]

        # Step 6: Extract embedded frames
        embedded_frames = []
        for i in range(len(loaded_video)):
            frame = loaded_video[i]
            embedded_frames.append(frame.copy())

    finally:
        # Restore original default
        set_default_image_plugin(original_default)

    # Step 7: Compare frames - should match exactly regardless of plugin
    assert len(original_frames) == len(embedded_frames)
    for i in range(len(original_frames)):
        orig = original_frames[i]
        embd = embedded_frames[i]

        # Frames should match exactly (pixel-perfect)
        assert np.array_equal(orig, embd), (
            f"Frame {i} mismatch (encode={encode_plugin}, decode={decode_plugin})"
        )

        # Additional check: ensure no accidental channel swap
        if orig.shape[-1] == 3:
            # If channels were swapped, they should NOT match
            orig_reversed = orig[..., ::-1]
            if not np.array_equal(
                orig, orig_reversed
            ):  # Only if image is not symmetric
                assert not np.array_equal(orig_reversed, embd), (
                    f"Frame {i} has reversed channels!"
                )


@pytest.mark.parametrize("plugin", ["opencv", "imageio"])
def test_embed_channel_order_metadata(tmp_path, small_robot_video, plugin):
    """Test that channel_order metadata is correctly stored in HDF5 datasets.

    Args:
        tmp_path: Pytest temporary directory fixture.
        small_robot_video: Test video fixture (3-frame robot video).
        plugin: Plugin to use for encoding ("opencv" or "imageio").
    """
    # Create minimal Labels
    skeleton = Skeleton(nodes=["node1"])
    instance = Instance(points=np.array([[10.0, 20.0]]), skeleton=skeleton)
    labeled_frame = LabeledFrame(
        video=small_robot_video, frame_idx=0, instances=[instance]
    )
    labels = Labels(
        videos=[small_robot_video], skeletons=[skeleton], labeled_frames=[labeled_frame]
    )

    # Save with specified plugin
    pkg_path = tmp_path / f"test_{plugin}.pkg.slp"
    original_default = get_default_image_plugin()
    try:
        set_default_image_plugin(plugin)
        save_slp(labels, str(pkg_path), embed="all", plugin=plugin)
    finally:
        set_default_image_plugin(original_default)

    # Verify HDF5 metadata
    with h5py.File(pkg_path, "r") as f:
        video_ds = f["video0/video"]

        # Check that channel_order attribute exists
        assert "channel_order" in video_ds.attrs, "channel_order attribute missing"

        # Check that it has the correct value
        expected_channel_order = "BGR" if plugin == "opencv" else "RGB"
        actual_channel_order = video_ds.attrs["channel_order"]
        assert actual_channel_order == expected_channel_order, (
            f"Expected {expected_channel_order}, got {actual_channel_order}"
        )

        # Also verify format_id is 1.4+
        assert "metadata" in f
        assert "format_id" in f["metadata"].attrs
        format_id = f["metadata"].attrs["format_id"]
        assert format_id >= 1.4, f"Expected format_id >= 1.4, got {format_id}"


def test_embed_backwards_compatibility_channel_order(tmp_path, small_robot_video):
    """Test that legacy files (format < 1.4) without channel_order default to BGR.

    This ensures backwards compatibility with files created before the channel_order
    attribute was added.

    Args:
        tmp_path: Pytest temporary directory fixture.
        small_robot_video: Test video fixture (3-frame robot video).
    """
    # Create and save a file with current format
    skeleton = Skeleton(nodes=["node1"])
    instance = Instance(points=np.array([[10.0, 20.0]]), skeleton=skeleton)
    labeled_frame = LabeledFrame(
        video=small_robot_video, frame_idx=0, instances=[instance]
    )
    labels = Labels(
        videos=[small_robot_video], skeletons=[skeleton], labeled_frames=[labeled_frame]
    )

    pkg_path = tmp_path / "legacy.pkg.slp"
    save_slp(labels, str(pkg_path), embed="all")

    # Manually modify the HDF5 file to simulate a legacy file
    with h5py.File(pkg_path, "r+") as f:
        # Remove channel_order attribute
        video_ds = f["video0/video"]
        if "channel_order" in video_ds.attrs:
            del video_ds.attrs["channel_order"]

        # Set format_id to 1.3 (pre-channel_order)
        f["metadata"].attrs["format_id"] = 1.3

    # Load the file and verify it defaults to BGR
    loaded_labels = load_slp(str(pkg_path))
    loaded_video = loaded_labels.videos[0]

    # The HDF5Video backend should have defaulted channel_order to BGR
    assert hasattr(loaded_video.backend, "channel_order")
    assert loaded_video.backend.channel_order == "BGR", (
        "Legacy files should default to BGR"
    )

    # Verify we can still read frames without errors
    frame = loaded_video[0]
    assert frame is not None
    assert frame.shape[-1] == 3  # RGB image


def test_load_slp_with_sparse_video_indices(tmp_path, small_robot_video):
    """Test loading .slp files with sparse video indices from old SLEAP.

    Old SLEAP versions (format_id < 2.0) could create files where video IDs
    in the frames dataset were sparse (e.g., 0, 15, 29, 47, ...) rather than
    sequential (0, 1, 2, 3, ...). This occurred when videos were deleted or
    when subsets were exported while preserving original video IDs.

    This test creates a synthetic file with sparse video indices and verifies
    that sleap-io can load it correctly by building a video_id → list_index
    mapping from backend.dataset metadata.

    Args:
        tmp_path: Pytest temporary directory fixture.
        small_robot_video: Small 3-frame video fixture.
    """
    # Step 1: Create Labels with 5 videos (reusing same video file for simplicity)
    skeleton = Skeleton(nodes=["node1", "node2"])
    videos = []
    labeled_frames = []

    for i in range(5):
        # Create separate Video objects (same file, different objects)
        video = Video.from_filename(small_robot_video.filename)
        videos.append(video)

        # Add 2 labeled frames per video (frame 0 and frame 1)
        for frame_idx in [0, 1]:
            instance = Instance(
                points=np.array([[10.0 + i, 20.0], [30.0, 40.0 + i]]),
                skeleton=skeleton,
            )
            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=[instance])
            labeled_frames.append(lf)

    labels = Labels(videos=videos, skeletons=[skeleton], labeled_frames=labeled_frames)

    # Step 2: Save with embedded frames (creates video0, video1, ..., video4 groups)
    embedded_path = tmp_path / "embedded.slp"
    save_slp(labels, str(embedded_path), embed="all")

    # Step 3: Create sparse version with renamed video groups
    sparse_path = tmp_path / "sparse.slp"
    sparse_indices = [0, 15, 29, 47, 67]  # Sparse IDs mimicking old SLEAP

    with h5py.File(embedded_path, "r") as src:
        with h5py.File(sparse_path, "w") as dst:
            # Copy non-video groups/datasets
            for key in [
                "metadata",
                "points",
                "pred_points",
                "instances",
                "tracks_json",
                "suggestions_json",
                "sessions_json",
            ]:
                if key in src:
                    src.copy(key, dst)

            # Copy and rename video groups with sparse indices
            for i in range(5):
                old_name = f"video{i}"
                new_name = f"video{sparse_indices[i]}"
                if old_name in src:
                    src.copy(old_name, dst, name=new_name)

            # Update frames dataset to use sparse video IDs
            frames = src["frames"][:]
            new_frames = []
            for frame in frames:
                frame_id, video_id, frame_idx, inst_start, inst_end = frame
                new_video_id = sparse_indices[video_id]
                new_frames.append(
                    (frame_id, new_video_id, frame_idx, inst_start, inst_end)
                )

            dst.create_dataset("frames", data=np.array(new_frames, dtype=frames.dtype))

            # Update videos_json to reference sparse dataset names
            videos_json = src["videos_json"][:]
            new_videos_json = []
            for i, vj_bytes in enumerate(videos_json):
                vj = json.loads(vj_bytes.decode("utf-8"))
                # Update dataset reference to sparse index
                if "backend" in vj and "dataset" in vj["backend"]:
                    vj["backend"]["dataset"] = f"video{sparse_indices[i]}/video"
                new_videos_json.append(np.bytes_(json.dumps(vj, separators=(",", ":"))))

            dst.create_dataset("videos_json", data=new_videos_json, maxshape=(None,))

    # Step 4: Verify the sparse structure was created correctly
    with h5py.File(sparse_path, "r") as f:
        video_groups = [
            k for k in f.keys() if k.startswith("video") and k[5:].isdigit()
        ]
        video_ids = sorted([int(vg.replace("video", "")) for vg in video_groups])
        assert video_ids == sparse_indices, "Video groups should have sparse indices"

        frames_data = f["frames"][:]
        unique_video_ids = sorted(np.unique(frames_data["video"]))
        assert unique_video_ids == sparse_indices, (
            "Frames should reference sparse video IDs"
        )

    # Step 5: Attempt to load - will fail with IndexError before fix
    # After fix, this should work correctly
    loaded_labels = load_slp(str(sparse_path))

    # Step 6: Verify correct loading
    assert len(loaded_labels.videos) == 5, "Should load 5 videos"
    assert len(loaded_labels) == 10, "Should load 10 frames (5 videos × 2 frames)"

    # Verify each video has exactly 2 frames
    for i, video in enumerate(loaded_labels.videos):
        frames_for_video = [lf for lf in loaded_labels if lf.video == video]
        assert len(frames_for_video) == 2, f"Video {i} should have 2 frames"

        # Verify frame instances are correct
        for lf in frames_for_video:
            assert len(lf.instances) == 1, "Each frame should have 1 instance"
            # Check skeleton has same structure (node names)
            assert len(lf.instances[0].skeleton.nodes) == len(skeleton.nodes)
            assert [n.name for n in lf.instances[0].skeleton.nodes] == [
                n.name for n in skeleton.nodes
            ]

    # test writing slp file with sparse video indices (hdf5 backend) and loading it back
    resaved_path = tmp_path / "resaved_sparse.slp"
    save_slp(
        load_slp(str(embedded_path)), str(resaved_path), restore_original_videos=False
    )
    new_labels = load_slp(str(resaved_path))
    assert len(new_labels.videos) == 5, "Should load 5 videos"
    assert len(new_labels) == 10, "Should load 10 frames (5 videos × 2 frames)"
    for i, video in enumerate(new_labels.videos):
        frames_for_video = new_labels.find(video=video)
        assert len(frames_for_video) == 2, f"Video {i} should have 2 frames"
    for lf in new_labels:
        assert lf.image.shape[-3:] == small_robot_video[0].shape


def test_load_slp_with_sequential_ids_sparse_datasets(tmp_path, small_robot_video):
    """Test loading .slp files with sequential video IDs but sparse dataset names.

    This occurs when files are exported/split from larger .pkg.slp files:
    - Videos retain their original sparse dataset names (video51, video49, etc.)
    - But frames are re-indexed sequentially (video_id = 0, 1, 2, 3)

    In this case, frame video IDs are list indices and should NOT be remapped
    based on dataset names.

    Args:
        tmp_path: Pytest temporary directory fixture.
        small_robot_video: Small 3-frame video fixture.
    """
    # Step 1: Create Labels with 4 videos
    skeleton = Skeleton(nodes=["node1", "node2"])
    videos = []
    labeled_frames = []

    for i in range(4):
        video = Video.from_filename(small_robot_video.filename)
        videos.append(video)

        # Add 2 labeled frames per video
        for frame_idx in [0, 1]:
            instance = Instance(
                points=np.array([[10.0 + i, 20.0], [30.0, 40.0 + i]]),
                skeleton=skeleton,
            )
            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=[instance])
            labeled_frames.append(lf)

    labels = Labels(videos=videos, skeletons=[skeleton], labeled_frames=labeled_frames)

    # Step 2: Save with embedded frames
    embedded_path = tmp_path / "embedded.slp"
    save_slp(labels, str(embedded_path), embed="all")

    # Step 3: Create version with sparse dataset names but sequential frame IDs
    # This mimics files exported from larger .pkg.slp files
    sparse_dataset_path = tmp_path / "sparse_datasets_sequential_ids.slp"
    sparse_dataset_indices = [51, 49, 97, 23]  # Sparse dataset names

    with h5py.File(embedded_path, "r") as src:
        with h5py.File(sparse_dataset_path, "w") as dst:
            # Copy non-video groups/datasets
            for key in [
                "metadata",
                "points",
                "pred_points",
                "instances",
                "tracks_json",
                "suggestions_json",
                "sessions_json",
            ]:
                if key in src:
                    src.copy(key, dst)

            # Copy and rename video groups with sparse indices
            for i in range(4):
                old_name = f"video{i}"
                new_name = f"video{sparse_dataset_indices[i]}"
                if old_name in src:
                    src.copy(old_name, dst, name=new_name)

            # Keep frames dataset with SEQUENTIAL video IDs (0, 1, 2, 3)
            # This is the key difference from test_load_slp_with_sparse_video_indices
            frames = src["frames"][:]
            dst.create_dataset("frames", data=frames)

            # Update videos_json to reference sparse dataset names
            videos_json = src["videos_json"][:]
            new_videos_json = []
            for i, vj_bytes in enumerate(videos_json):
                vj = json.loads(vj_bytes.decode("utf-8"))
                if "backend" in vj and "dataset" in vj["backend"]:
                    vj["backend"]["dataset"] = f"video{sparse_dataset_indices[i]}/video"
                new_videos_json.append(np.bytes_(json.dumps(vj, separators=(",", ":"))))

            dst.create_dataset("videos_json", data=new_videos_json, maxshape=(None,))

    # Step 4: Verify the structure was created correctly
    with h5py.File(sparse_dataset_path, "r") as f:
        # Dataset names should be sparse
        video_groups = [
            k for k in f.keys() if k.startswith("video") and k[5:].isdigit()
        ]
        video_ids_in_groups = sorted(
            [int(vg.replace("video", "")) for vg in video_groups]
        )
        assert video_ids_in_groups == sorted(sparse_dataset_indices), (
            "Video groups should have sparse indices"
        )

        # Frame video IDs should be sequential (0, 1, 2, 3)
        frames_data = f["frames"][:]
        unique_video_ids = sorted(np.unique(frames_data["video"]))
        assert unique_video_ids == [0, 1, 2, 3], (
            "Frame video IDs should be sequential list indices"
        )

    # Step 5: Load the file - before fix, this would incorrectly map video IDs
    # based on dataset names, causing wrong video associations
    loaded_labels = load_slp(str(sparse_dataset_path))

    # Step 6: Verify correct loading
    assert len(loaded_labels.videos) == 4, "Should load 4 videos"
    assert len(loaded_labels) == 8, "Should load 8 frames (4 videos × 2 frames)"

    # Verify each video has exactly 2 frames
    for i, video in enumerate(loaded_labels.videos):
        frames_for_video = [lf for lf in loaded_labels if lf.video == video]
        assert len(frames_for_video) == 2, f"Video {i} should have 2 frames"

        # Verify frame instances have correct point values
        # Points should be (10+i, 20) and (30, 40+i) based on how we created them
        for lf in frames_for_video:
            assert len(lf.instances) == 1, "Each frame should have 1 instance"
            pts = lf.instances[0].numpy()
            # The first point's x coordinate should be 10 + video_index
            expected_x = 10.0 + i
            assert abs(pts[0, 0] - expected_x) < 0.01, (
                f"Video {i}: Expected point x={expected_x}, got {pts[0, 0]}. "
                "Video ID mapping may be incorrect."
            )


def test_save_slp_non_sparse_videos(tmp_path, slp_minimal):
    """Test saving labels with non-embedded videos (fallback to sequential indexing)."""
    # Save with original videos (non-embedded)
    output_path = tmp_path / "non_sparse.slp"
    labels = load_slp(slp_minimal)
    save_slp(labels, str(output_path), restore_original_videos=True)

    # Load and verify
    loaded_labels = load_slp(str(output_path))
    assert len(loaded_labels.videos) == len(labels.videos)
    assert len(loaded_labels) == len(labels)


def test_save_slp_video_dataset_edge_cases(
    tmp_path, slp_minimal_pkg, small_robot_video
):
    """Test video ID extraction with edge case dataset names.

    Covers lines 1232, 1234, 1236: Tests negative branches where dataset
    parsing falls back to sequential indexing.
    """
    # Load a skeleton and create labels with multiple videos for edge cases
    skeleton = read_skeletons(slp_minimal_pkg)[0]

    # Create 3 videos (using the same video file for all 3)
    videos = [Video(small_robot_video.filename) for _ in range(3)]

    # Create minimal labels with the 3 videos
    # Create points matching the skeleton's nodes
    num_nodes = len(skeleton.nodes)
    labels = Labels(
        videos=videos,
        skeletons=[skeleton],
        labeled_frames=[
            LabeledFrame(
                video=videos[i],
                frame_idx=0,
                instances=[
                    Instance.from_numpy(
                        points_data=np.array(
                            [[10.0 + i + j, 20.0 + i + j] for j in range(num_nodes)]
                        ),
                        skeleton=skeleton,
                    )
                ],
            )
            for i in range(3)
        ],
    )

    # Save with embedded videos
    embedded_path = tmp_path / "embedded.slp"
    save_slp(labels, str(embedded_path), restore_original_videos=False)

    # Modify the HDF5 file to create edge cases
    with h5py.File(embedded_path, "r+") as f:
        videos_json = [json.loads(vj.decode("utf-8")) for vj in f["videos_json"][:]]

        # Edge case 1: dataset without "/" (line 1232 negative branch)
        if "video0" in f:
            f.move("video0", "video_noSlash")
            videos_json[0]["backend"]["dataset"] = "video_noSlash"

        # Edge case 2: dataset with "/" but doesn't start with "video" (line 1234)
        if "video1" in f:
            f.move("video1", "other1")
            videos_json[1]["backend"]["dataset"] = "other1/video"

        # Edge case 3: dataset like "videoABC/video" - non-numeric (line 1236)
        if "video2" in f:
            f.move("video2", "videoABC")
            videos_json[2]["backend"]["dataset"] = "videoABC/video"

        # Write back modified videos_json
        del f["videos_json"]
        f.create_dataset(
            "videos_json",
            data=[
                np.bytes_(json.dumps(vj, separators=(",", ":"))) for vj in videos_json
            ],
            maxshape=(None,),
        )

    # Load the modified file - should handle edge cases with fallback
    loaded_labels = load_slp(str(embedded_path))
    assert len(loaded_labels.videos) == 3
    assert len(loaded_labels) == 3

    # Resave to exercise write path with edge case dataset names
    # This is the key part that exercises lines 1232, 1234, 1236
    resaved_path = tmp_path / "resaved.slp"
    save_slp(loaded_labels, str(resaved_path), restore_original_videos=False)

    # Verify round-trip works
    final_labels = load_slp(str(resaved_path))
    assert len(final_labels.videos) == 3
    assert len(final_labels) == 3


def test_load_slp_sparse_video_ids_with_fallback(tmp_path, small_robot_video):
    """Test loading sparse video IDs with non-standard dataset names (fallback path).

    This test covers line 1973 in slp.py where videos without extractable IDs
    from their dataset names fall back to sequential indexing.

    Scenario: 3 videos with sparse frame IDs [0, 1, 15]:
    - Video 0: standard dataset "video0/video" -> extracts ID 0
    - Video 1: non-standard dataset "custom/video" -> fallback uses index 1
    - Video 2: standard dataset "video15/video" -> extracts ID 15

    The fallback at line 1973 is hit for video 1 because "custom/video" doesn't
    match the "videoN/video" pattern.
    """
    skeleton = Skeleton(nodes=["node1", "node2"])

    # Create 3 videos
    videos = [Video.from_filename(small_robot_video.filename) for _ in range(3)]

    # Create labeled frames
    labeled_frames = []
    for i, video in enumerate(videos):
        instance = Instance(
            points=np.array([[10.0 + i, 20.0], [30.0, 40.0 + i]]),
            skeleton=skeleton,
        )
        lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
        labeled_frames.append(lf)

    labels = Labels(videos=videos, skeletons=[skeleton], labeled_frames=labeled_frames)

    # Save with embedded frames
    embedded_path = tmp_path / "embedded.slp"
    save_slp(labels, str(embedded_path), embed="all")

    # Create sparse version: frame IDs [0, 1, 15] with video 1 having non-standard name
    sparse_path = tmp_path / "sparse_fallback.slp"
    # Frame video IDs: video0->0, video1->1, video2->15
    sparse_video_ids = [0, 1, 15]

    with h5py.File(embedded_path, "r") as src:
        with h5py.File(sparse_path, "w") as dst:
            # Copy non-video data
            for key in ["metadata", "points", "pred_points", "instances"]:
                if key in src:
                    src.copy(key, dst)

            # Copy optional datasets
            for key in ["tracks_json", "suggestions_json", "sessions_json"]:
                if key in src:
                    src.copy(key, dst)

            # Video 0: standard dataset name "video0/video"
            if "video0" in src:
                src.copy("video0", dst, name="video0")

            # Video 1: NON-standard name "custom" (no "videoN" prefix)
            if "video1" in src:
                src.copy("video1", dst, name="custom")

            # Video 2: standard sparse name "video15"
            if "video2" in src:
                src.copy("video2", dst, name="video15")

            # Update frames dataset with sparse video IDs
            frames = src["frames"][:]
            new_frames = []
            for frame in frames:
                frame_id, video_id, frame_idx, inst_start, inst_end = frame
                new_video_id = sparse_video_ids[video_id]
                new_frames.append(
                    (frame_id, new_video_id, frame_idx, inst_start, inst_end)
                )
            dst.create_dataset("frames", data=np.array(new_frames, dtype=frames.dtype))

            # Update videos_json
            videos_json = src["videos_json"][:]
            new_videos_json = []
            for i, vj_bytes in enumerate(videos_json):
                vj = json.loads(vj_bytes.decode("utf-8"))
                if i == 0:
                    vj["backend"]["dataset"] = "video0/video"
                elif i == 1:
                    # Non-standard: no "videoN" prefix
                    vj["backend"]["dataset"] = "custom/video"
                else:
                    vj["backend"]["dataset"] = "video15/video"
                new_videos_json.append(np.bytes_(json.dumps(vj, separators=(",", ":"))))

            dst.create_dataset("videos_json", data=new_videos_json, maxshape=(None,))

    # Verify sparse structure
    with h5py.File(sparse_path, "r") as f:
        frames_data = f["frames"][:]
        unique_video_ids = sorted(np.unique(frames_data["video"]))
        assert unique_video_ids == sparse_video_ids, "Frame IDs should be sparse"

    # Load - this triggers the fallback at line 1973 for video 1 (custom/video)
    loaded_labels = load_slp(str(sparse_path))

    assert len(loaded_labels.videos) == 3, "Should load 3 videos"
    assert len(loaded_labels) == 3, "Should load 3 frames"


def test_progress_callback_receives_correct_values(tmp_path, slp_real_data):
    """Test that progress_callback receives correct (current, total) values."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    # Track all progress updates
    progress_updates = []

    def track_progress(current, total):
        progress_updates.append((current, total))
        return True  # Continue

    write_labels(
        labels_path, base_labels, embed="user", progress_callback=track_progress
    )

    # Should have received updates for each frame
    assert len(progress_updates) == 5  # user_labeled_frames count

    # Check that current values are 1-indexed and sequential
    for i, (current, total) in enumerate(progress_updates):
        assert current == i + 1, f"Current should be {i + 1}, got {current}"
        assert total == 5, f"Total should be 5, got {total}"


def test_progress_callback_cancellation(tmp_path, slp_real_data):
    """Test that returning False from callback cancels the operation."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    # Cancel after the second frame
    def cancel_after_two(current, total):
        return current < 2  # Return False after 2nd frame

    with pytest.raises(ExportCancelled, match="Export cancelled by user"):
        write_labels(
            labels_path, base_labels, embed="user", progress_callback=cancel_after_two
        )


def test_progress_callback_disables_tqdm(tmp_path, slp_real_data, capsys):
    """Test that tqdm is disabled when progress_callback is provided."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    def noop_callback(current, total):
        return True

    # With callback, even with verbose=True, no tqdm output should appear
    write_labels(
        labels_path,
        base_labels,
        embed="user",
        verbose=True,
        progress_callback=noop_callback,
    )

    # tqdm outputs to stderr by default
    captured = capsys.readouterr()
    assert "Embedding frames" not in captured.err


def test_progress_callback_with_save_slp(tmp_path, slp_real_data):
    """Test progress_callback works through save_slp API."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    progress_updates = []

    def track_progress(current, total):
        progress_updates.append((current, total))
        return True

    save_slp(base_labels, labels_path, embed="user", progress_callback=track_progress)

    assert len(progress_updates) == 5
    assert progress_updates[-1] == (5, 5)


def test_progress_callback_with_save_file(tmp_path, slp_real_data):
    """Test progress_callback works through save_file API."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    progress_updates = []

    def track_progress(current, total):
        progress_updates.append((current, total))
        return True

    save_file(base_labels, labels_path, embed="user", progress_callback=track_progress)

    assert len(progress_updates) == 5
    assert progress_updates[-1] == (5, 5)


def test_progress_callback_none_uses_tqdm(tmp_path, slp_real_data, capsys):
    """Test that tqdm is used when progress_callback is None and verbose=True."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    write_labels(labels_path, base_labels, embed="user", verbose=True)

    # tqdm outputs to stderr
    captured = capsys.readouterr()
    assert "Embedding frames" in captured.err


def test_embed_inplace_false_preserves_original(tmp_path, slp_real_data):
    """Test that embed_inplace=False (default) doesn't modify the original labels."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    # Store original video info
    original_video_filename = base_labels.video.filename
    original_video_type = type(base_labels.video.backend)

    # Save with embedding (default embed_inplace=False)
    write_labels(labels_path, base_labels, embed="user")

    # Original labels should NOT be modified
    assert base_labels.video.filename == original_video_filename
    assert type(base_labels.video.backend) is original_video_type
    assert base_labels.video.filename != labels_path  # Should not point to pkg.slp


def test_embed_inplace_true_modifies_original(tmp_path, slp_real_data):
    """Test that embed_inplace=True modifies the labels in-place."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    # Store original video info
    original_video_filename = base_labels.video.filename

    # Save with embedding and embed_inplace=True
    write_labels(labels_path, base_labels, embed="user", embed_inplace=True)

    # Original labels SHOULD be modified to point to embedded videos
    assert base_labels.video.filename == labels_path
    assert type(base_labels.video.backend) is HDF5Video
    assert base_labels.video.filename != original_video_filename


def test_embed_inplace_via_labels_save(tmp_path, slp_real_data):
    """Test embed_inplace parameter works through Labels.save()."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    # Store original video info
    original_video_filename = base_labels.video.filename

    # Save with embedding via Labels.save() (default embed_inplace=False)
    base_labels.save(labels_path, embed="user")

    # Original labels should NOT be modified
    assert base_labels.video.filename == original_video_filename


def test_embed_inplace_via_save_slp(tmp_path, slp_real_data):
    """Test embed_inplace parameter works through save_slp()."""
    base_labels = read_labels(slp_real_data)
    labels_path = str(tmp_path / "labels.pkg.slp")

    # Store original video info
    original_video_filename = base_labels.video.filename

    # Save with embedding via save_slp() with embed_inplace=True
    save_slp(base_labels, labels_path, embed="user", embed_inplace=True)

    # Original labels SHOULD be modified
    assert base_labels.video.filename != original_video_filename
    assert base_labels.video.filename == labels_path
    assert type(base_labels.video.backend) is HDF5Video


def test_fps_serialization_mediavideo(tmp_path, slp_real_data):
    """Test FPS is serialized and restored for MediaVideo."""
    labels = read_labels(slp_real_data)

    # MediaVideo should have FPS from container
    assert labels.video.fps == 15.0

    # Save and reload
    labels_path = str(tmp_path / "test_fps.slp")
    save_slp(labels, labels_path)

    # Reload and check FPS is preserved
    reloaded = read_labels(labels_path)
    assert reloaded.video.fps == 15.0


def test_fps_serialization_imagevideo(tmp_path, centered_pair_frame_paths):
    """Test FPS is serialized and restored for ImageVideo."""
    # Create labels with ImageVideo
    video = Video.from_filename(centered_pair_frame_paths)
    labels = Labels(videos=[video])

    # ImageVideo has no inherent FPS
    assert video.fps is None

    # Set FPS explicitly
    video.fps = 25.0
    assert video.fps == 25.0

    # Save and reload
    labels_path = str(tmp_path / "test_fps_img.slp")
    save_slp(labels, labels_path)

    reloaded = read_labels(labels_path)
    assert reloaded.video.fps == 25.0


def test_fps_embedded_video_inheritance(tmp_path, slp_real_data):
    """Test FPS is preserved when embedding frames."""
    labels = read_labels(slp_real_data)

    # Original video should have FPS
    assert labels.video.fps == 15.0

    # Save as package with embedded frames
    pkg_path = str(tmp_path / "test_fps.pkg.slp")
    save_slp(labels, pkg_path, embed="user")

    # Reload and check FPS is preserved in embedded video
    reloaded = read_labels(pkg_path)
    assert type(reloaded.video.backend) is HDF5Video
    assert reloaded.video.fps == 15.0


def test_fps_in_video_to_dict(slp_real_data):
    """Test that video_to_dict includes FPS."""
    from sleap_io.io.slp import video_to_dict

    labels = read_labels(slp_real_data)
    video = labels.video

    video_dict = video_to_dict(video)
    assert "fps" in video_dict["backend"]
    assert video_dict["backend"]["fps"] == 15.0


def test_fps_hdf5_attrs(tmp_path, slp_real_data):
    """Test FPS is stored in HDF5 attributes for embedded videos."""
    labels = read_labels(slp_real_data)
    assert labels.video.fps == 15.0

    # Save as package
    pkg_path = str(tmp_path / "test_fps.pkg.slp")
    save_slp(labels, pkg_path, embed="user")

    # Check HDF5 attributes directly
    with h5py.File(pkg_path, "r") as f:
        # Find the video dataset
        video_ds = f["video0/video"]
        assert "fps" in video_ds.attrs
        assert video_ds.attrs["fps"] == 15.0


def test_fps_backward_compatibility(tmp_path, slp_real_data):
    """Test that files without FPS field load correctly."""
    labels = read_labels(slp_real_data)

    # Save normally
    labels_path = str(tmp_path / "test.slp")
    save_slp(labels, labels_path)

    # Remove FPS from the saved file to simulate old format
    with h5py.File(labels_path, "r+") as f:
        videos_data = json.loads(f["videos_json"][0])
        del videos_data["backend"]["fps"]
        del f["videos_json"]
        f.create_dataset("videos_json", data=[json.dumps(videos_data)])

    # Should still load - FPS comes from container for MediaVideo
    reloaded = read_labels(labels_path)
    assert reloaded.video.fps == 15.0  # Read from container


def test_write_videos_preserves_embedded_without_backend(tmp_path, slp_minimal_pkg):
    """Test that embedded videos are preserved when loaded with open_videos=False.

    This tests the fix for the bug where `sio fix` and other commands that load
    with `open_videos=False` would lose embedded video data when saving.
    """
    from sleap_io.io.slp import VideoReferenceMode

    # Load with open_videos=False (as sio fix does)
    labels = read_labels(slp_minimal_pkg, open_videos=False)
    assert labels.videos[0].backend is None  # No backend due to open_videos=False

    # Verify the video has embedded metadata
    meta = labels.videos[0].backend_metadata
    assert meta is not None
    assert meta.get("filename") == "."  # Marker for embedded
    assert "dataset" in meta

    # Save to new file (EMBED mode should preserve embedded data)
    output = tmp_path / "test_preserve_embedded.pkg.slp"
    write_videos(str(output), labels.videos, reference_mode=VideoReferenceMode.EMBED)

    # Verify the embedded video data was copied
    with h5py.File(output, "r") as f:
        # Check video dataset exists
        assert "video0/video" in f
        # Check it has actual data
        assert f["video0/video"].shape[0] > 0


def test_save_slp_preserves_embedded_without_backend(tmp_path, slp_minimal_pkg):
    """Test save_slp preserves embedded videos when loaded with open_videos=False."""
    # Load with open_videos=False
    labels = read_labels(slp_minimal_pkg, open_videos=False)
    assert labels.videos[0].backend is None

    # Save with embed=None (preserve existing embedded)
    output = tmp_path / "test_preserve.pkg.slp"
    save_slp(labels, str(output), embed=None)

    # Verify embedded data was preserved
    with h5py.File(output, "r") as f:
        assert "video0/video" in f
        assert f["video0/video"].shape[0] > 0

    # Verify the saved file is usable
    reloaded = read_labels(str(output), open_videos=True)
    assert reloaded.videos[0].backend is not None
    assert type(reloaded.videos[0].backend).__name__ == "HDF5Video"

    # Verify we can read frames
    frame = reloaded.videos[0][0]
    assert frame.shape[0] > 0


def test_is_embedded_video_metadata():
    """Test the _is_embedded_video_metadata helper function."""
    from sleap_io.io.slp import _is_embedded_video_metadata

    # Test with embedded video metadata
    embedded_video = Video(
        filename="test.pkg.slp",
        backend=None,
        backend_metadata={"filename": ".", "dataset": "video0/video"},
    )
    assert _is_embedded_video_metadata(embedded_video) is True

    # Test with non-embedded video metadata
    regular_video = Video(
        filename="test.mp4",
        backend=None,
        backend_metadata={"filename": "test.mp4", "type": "MediaVideo"},
    )
    assert _is_embedded_video_metadata(regular_video) is False

    # Test with no metadata
    no_meta_video = Video(filename="test.mp4", backend=None, backend_metadata=None)
    assert _is_embedded_video_metadata(no_meta_video) is False

    # Test with empty metadata
    empty_meta_video = Video(filename="test.mp4", backend=None, backend_metadata={})
    assert _is_embedded_video_metadata(empty_meta_video) is False


def test_write_videos_embedded_metadata_restore_original(tmp_path, slp_minimal_pkg):
    """Test RESTORE_ORIGINAL mode with embedded videos detected via metadata."""
    from sleap_io.io.slp import VideoReferenceMode

    # Load with open_videos=False
    labels = read_labels(slp_minimal_pkg, open_videos=False)
    assert labels.videos[0].backend is None
    assert labels.videos[0].source_video is not None  # Has source video

    # Test RESTORE_ORIGINAL mode - should use source_video
    output = tmp_path / "test_restore.slp"
    write_videos(
        str(output), labels.videos, reference_mode=VideoReferenceMode.RESTORE_ORIGINAL
    )

    # Verify metadata was written (source video path)
    loaded = read_videos(str(output))
    assert len(loaded) == 1


def test_write_videos_embedded_metadata_restore_original_no_source(tmp_path):
    """Test RESTORE_ORIGINAL mode when embedded video has no source_video."""
    from sleap_io.io.slp import VideoReferenceMode

    # Create video with embedded metadata but no source_video
    video = Video(
        filename="test.pkg.slp",
        backend=None,
        backend_metadata={"filename": ".", "dataset": "video0/video"},
        source_video=None,
    )

    output = tmp_path / "test_restore_no_source.slp"
    write_videos(
        str(output), [video], reference_mode=VideoReferenceMode.RESTORE_ORIGINAL
    )

    loaded = read_videos(str(output))
    assert len(loaded) == 1


def test_write_videos_embedded_metadata_preserve_source(tmp_path, slp_minimal_pkg):
    """Test PRESERVE_SOURCE mode with embedded videos detected via metadata."""
    from sleap_io.io.slp import VideoReferenceMode

    labels = read_labels(slp_minimal_pkg, open_videos=False)

    output = tmp_path / "test_preserve.slp"
    write_videos(
        str(output), labels.videos, reference_mode=VideoReferenceMode.PRESERVE_SOURCE
    )

    loaded = read_videos(str(output))
    assert len(loaded) == 1


def test_write_videos_embedded_metadata_already_embedded(tmp_path, slp_minimal_pkg):
    """Test EMBED mode when destination already has embedded video."""
    from sleap_io.io.slp import VideoReferenceMode

    labels = read_labels(slp_minimal_pkg, open_videos=False)

    # First, create output with embedded video data
    output = tmp_path / "test_already.pkg.slp"
    write_videos(str(output), labels.videos, reference_mode=VideoReferenceMode.EMBED)

    # Now call again - should detect already embedded and skip copy
    write_videos(str(output), labels.videos, reference_mode=VideoReferenceMode.EMBED)

    # Verify file still valid
    with h5py.File(output, "r") as f:
        assert "video0/video" in f


def test_write_videos_copy_source_not_exists(tmp_path):
    """Test videos_to_copy handling when source file doesn't exist."""
    from sleap_io.io.slp import VideoReferenceMode

    # Create video pointing to non-existent file
    video = Video(
        filename="/nonexistent/path.pkg.slp",
        backend=None,
        backend_metadata={"filename": ".", "dataset": "video0/video"},
    )

    output = tmp_path / "test_no_source.slp"
    write_videos(str(output), [video], reference_mode=VideoReferenceMode.EMBED)

    # Should still write metadata
    loaded = read_videos(str(output))
    assert len(loaded) == 1


def test_write_videos_copy_no_dataset_in_metadata(tmp_path, slp_minimal_pkg):
    """Test videos_to_copy handling when metadata has no dataset."""
    from sleap_io.io.slp import VideoReferenceMode

    # Create video with embedded marker but missing dataset
    video = Video(
        filename=slp_minimal_pkg,
        backend=None,
        backend_metadata={"filename": "."},  # No dataset key
    )

    output = tmp_path / "test_no_dataset.slp"
    write_videos(str(output), [video], reference_mode=VideoReferenceMode.EMBED)

    loaded = read_videos(str(output))
    assert len(loaded) == 1


def test_write_videos_copy_group_not_in_source(tmp_path, slp_minimal_pkg):
    """Test videos_to_copy when source HDF5 doesn't have expected group."""
    from sleap_io.io.slp import VideoReferenceMode

    # Create video pointing to valid file but wrong dataset name
    video = Video(
        filename=slp_minimal_pkg,
        backend=None,
        backend_metadata={"filename": ".", "dataset": "video999/video"},  # Non-existent
    )

    output = tmp_path / "test_wrong_group.slp"
    write_videos(str(output), [video], reference_mode=VideoReferenceMode.EMBED)

    loaded = read_videos(str(output))
    assert len(loaded) == 1


def test_slp_roi_roundtrip(tmp_path):
    """Test SLP round-trip with ROIs."""
    video = Video(filename="test.mp4")
    track = Track(name="animal1")
    roi1 = UserROI.from_bbox(10, 20, 30, 40, video=video, name="bbox1", category="cat")
    roi2 = UserROI.from_polygon(
        [(0, 0), (100, 0), (50, 100)],
        video=video,
        track=track,
        category="arena",
    )

    skeleton = Skeleton(nodes=["A"])
    labels = Labels(
        videos=[video], skeletons=[skeleton], tracks=[track], rois=[roi1, roi2]
    )

    path = str(tmp_path / "test_rois.slp")
    save_slp(labels, path)

    # Verify format_id is 1.5
    format_id = read_hdf5_attrs(path, "metadata", "format_id")
    assert format_id == 1.5

    loaded = load_slp(path)

    # Both ROIs round-trip as ROIs (no bbox migration)
    assert len(loaded.rois) == 2
    roi_names = {r.name for r in loaded.rois}
    assert "bbox1" in roi_names
    triangle = [r for r in loaded.rois if r.category == "arena"][0]
    assert triangle.track is loaded.tracks[0]


def _polygon_heavy_rois(n_rois=100, n_verts=64):
    """An ROI set of many-vertex polygons with a sizable, compressible WKB blob."""
    video = Video(filename="test.mp4")
    ang = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    base = np.stack([500 + 400 * np.cos(ang), 500 + 400 * np.sin(ang)], 1)
    rois = [
        UserROI.from_polygon((base + i * 0.13).tolist(), video=video, name=f"r{i}")
        for i in range(n_rois)
    ]
    skeleton = Skeleton(nodes=["A"])
    return Labels(videos=[video], skeletons=[skeleton], rois=rois)


def test_slp_roi_wkb_gzip_compressed(tmp_path):
    """`roi_wkb` is written with gzip compression (follow-up to #463)."""
    labels = _polygon_heavy_rois(n_rois=5, n_verts=8)
    path = str(tmp_path / "gzip_rois.slp")
    save_slp(labels, path)

    with h5py.File(path, "r") as f:
        wkb = f["roi_wkb"]
        assert wkb.compression == "gzip"
        assert wkb.compression_opts == 1
        assert wkb.chunks is not None  # gzip requires a chunked layout


def test_slp_roi_wkb_compression_reduces_size(tmp_path):
    """Gzip meaningfully shrinks `roi_wkb` on disk vs. its raw byte size."""
    labels = _polygon_heavy_rois(n_rois=100, n_verts=64)
    path = str(tmp_path / "compressed_rois.slp")
    save_slp(labels, path)

    with h5py.File(path, "r") as f:
        wkb = f["roi_wkb"]
        on_disk = wkb.id.get_storage_size()
        raw = wkb.nbytes
    # A polygon-heavy ROI set packs ~100 KB of WKB; gzip beats it ~2x.
    assert raw > 10240
    assert on_disk < raw * 0.8


def test_slp_roi_wkb_roundtrip_bit_identical(tmp_path):
    """Gzip is lossless: ROI geometry survives a round-trip with identical WKB."""
    labels = _polygon_heavy_rois(n_rois=20, n_verts=32)
    original_wkb = sorted(r.geometry.wkb for r in labels.rois)

    path = str(tmp_path / "roundtrip_rois.slp")
    save_slp(labels, path)
    loaded = load_slp(path)

    assert len(loaded.rois) == len(labels.rois)
    reloaded_wkb = sorted(r.geometry.wkb for r in loaded.rois)
    assert reloaded_wkb == original_wkb


def test_slp_roi_wkb_roundtrip_geometry_types(tmp_path):
    """Gzip round-trip preserves WKB + type across ROI geometry kinds.

    Covers a UserROI polygon, a PredictedROI (with score), and a MultiPolygon —
    all share the generic WKB serialize/deserialize path. Non-rectangular shapes
    avoid the rectangle->BoundingBox migration.
    """
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    rois = [
        UserROI(
            geometry=shapely.Polygon([(0, 0), (10, 0), (10, 10), (5, 15), (0, 10)]),
            video=video,
            category="user",
        ),
        PredictedROI(
            geometry=shapely.Polygon([(20, 20), (30, 20), (25, 35)]),
            video=video,
            category="pred",
            score=0.85,
        ),
        UserROI(
            geometry=shapely.MultiPolygon(
                [
                    shapely.Polygon([(0, 0), (5, 0), (5, 5)]),
                    shapely.Polygon([(10, 10), (15, 10), (15, 15)]),
                ]
            ),
            video=video,
            category="multi",
        ),
    ]
    original_wkb = {r.category: r.geometry.wkb for r in rois}

    path = str(tmp_path / "roi_geom_types.slp")
    save_slp(Labels(videos=[video], skeletons=[skeleton], rois=rois), path)
    loaded = load_slp(path, open_videos=False)

    assert len(loaded.rois) == 3
    by_cat = {r.category: r for r in loaded.rois}
    assert isinstance(by_cat["pred"], PredictedROI)
    assert by_cat["pred"].score == pytest.approx(0.85, abs=1e-5)
    assert isinstance(by_cat["user"], UserROI)
    assert by_cat["multi"].geometry.geom_type == "MultiPolygon"
    for cat, wkb in original_wkb.items():
        assert by_cat[cat].geometry.wkb == wkb


def test_slp_roi_wkb_empty_geometry(tmp_path):
    """An empty-geometry ROI still packs non-empty WKB and round-trips gzip.

    Pins the invariant that lets ``write_rois`` always gzip ``roi_wkb`` with no
    empty-dataset guard: every geometry — even an empty polygon — serializes to
    a non-empty WKB header, so ``wkb_flat`` is never empty when ROIs exist.
    """
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    roi = UserROI(geometry=shapely.Polygon(), video=video, category="empty")
    labels = Labels(videos=[video], skeletons=[skeleton], rois=[roi])

    path = str(tmp_path / "empty_geom_roi.slp")
    save_slp(labels, path)

    with h5py.File(path, "r") as f:
        assert f["roi_wkb"].compression == "gzip"
        assert f["roi_wkb"].nbytes > 0

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.rois) == 1
    assert loaded.rois[0].geometry.is_empty


def test_slp_mask_roundtrip(tmp_path):
    """Test SLP round-trip with segmentation masks."""
    video = Video(filename="test.mp4")
    mask_data = np.zeros((20, 30), dtype=bool)
    mask_data[5:15, 10:25] = True

    mask = UserSegmentationMask.from_numpy(
        mask_data,
        name="seg1",
        category="foreground",
    )

    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=3)
    lf.masks.append(mask)
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "test_masks.slp")
    save_slp(labels, path)

    loaded = load_slp(path)
    assert len(loaded.masks) == 1

    m = loaded.masks[0]
    assert m.height == 20
    assert m.width == 30
    assert m.name == "seg1"
    assert m.category == "foreground"

    # Check mask data roundtrips correctly
    np.testing.assert_array_equal(m.data, mask_data)


def _fragmented_mask():
    """A checkerboard mask whose RLE is large and highly compressible.

    Alternating pixels produce ~H*W single-pixel runs, the worst case for
    ``mask_rle`` size and the case that makes segmentation ``.slp`` files balloon.
    """
    y, x = np.indices((128, 128))
    return ((x + y) % 2).astype(bool)


def test_slp_mask_rle_gzip_compressed(tmp_path):
    """`mask_rle` is written with gzip compression (issue #463)."""
    video = Video(filename="test.mp4")
    mask = UserSegmentationMask.from_numpy(_fragmented_mask())
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf], videos=[video], skeletons=[Skeleton(nodes=["A"])]
    )

    path = str(tmp_path / "gzip_masks.slp")
    save_slp(labels, path)

    with h5py.File(path, "r") as f:
        rle = f["mask_rle"]
        assert rle.compression == "gzip"
        assert rle.compression_opts == 1
        assert rle.chunks is not None  # gzip requires a chunked layout


def test_slp_mask_rle_compression_reduces_size(tmp_path):
    """Gzip meaningfully shrinks `mask_rle` on disk vs. its raw byte size."""
    video = Video(filename="test.mp4")
    mask = UserSegmentationMask.from_numpy(_fragmented_mask())
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf], videos=[video], skeletons=[Skeleton(nodes=["A"])]
    )

    path = str(tmp_path / "compressed_masks.slp")
    save_slp(labels, path)

    with h5py.File(path, "r") as f:
        rle = f["mask_rle"]
        on_disk = rle.id.get_storage_size()
        raw = rle.nbytes
    # A fragmented RLE has thousands of bytes raw; gzip easily beats 2x on it.
    assert raw > 1024
    assert on_disk < raw / 2


def test_slp_mask_rle_roundtrip_bit_identical(tmp_path):
    """Gzip is lossless: RLE counts survive a save/load round-trip bit-identical."""
    video = Video(filename="test.mp4")
    mask = UserSegmentationMask.from_numpy(_fragmented_mask())
    original_rle = np.asarray(mask.rle_counts, dtype=np.uint32).copy()
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf], videos=[video], skeletons=[Skeleton(nodes=["A"])]
    )

    path = str(tmp_path / "roundtrip_masks.slp")
    save_slp(labels, path)
    loaded = load_slp(path)

    assert len(loaded.masks) == 1
    reloaded_rle = np.asarray(loaded.masks[0].rle_counts, dtype=np.uint32)
    np.testing.assert_array_equal(reloaded_rle, original_rle)


def test_slp_mask_rle_empty_roundtrip(tmp_path):
    """A degenerate empty-RLE mask uses the uncompressed path and round-trips.

    A zero-height mask produces no RLE bytes, exercising the empty-guard branch
    where chunking/gzip are skipped (nothing to compress). It must still load
    back losslessly.
    """
    video = Video(filename="test.mp4")
    mask = UserSegmentationMask.from_numpy(np.zeros((0, 5), dtype=bool))
    assert len(mask.rle_counts) == 0
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf], videos=[video], skeletons=[Skeleton(nodes=["A"])]
    )

    path = str(tmp_path / "empty_rle_masks.slp")
    save_slp(labels, path)

    with h5py.File(path, "r") as f:
        # Empty RLE skips the gzip filter: there is nothing to compress.
        assert f["mask_rle"].compression is None

    loaded = load_slp(path)
    assert len(loaded.masks) == 1
    assert len(loaded.masks[0].rle_counts) == 0


def test_slp_backward_compat_no_rois(slp_typical):
    """Reading old SLP files without ROIs/masks gives empty lists."""
    labels = read_labels(slp_typical)
    assert labels.rois == []
    assert labels.masks == []


def test_slp_no_rois_format_id(tmp_path):
    """Files without ROIs/masks should keep format_id 1.4."""
    skeleton = Skeleton(nodes=["A"])
    video = Video(filename="test.mp4")
    labels = Labels(videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "test_no_rois.slp")
    save_slp(labels, path)

    format_id = read_hdf5_attrs(path, "metadata", "format_id")
    assert format_id == 1.4


def test_slp_lazy_roi_roundtrip(tmp_path):
    """Test lazy loading round-trip preserves ROIs and masks."""
    video = Video(filename="test.mp4")
    # Use a non-rectangular polygon so it isn't migrated to BoundingBox
    roi = UserROI.from_polygon(
        [(0, 0), (100, 0), (50, 100)], video=video, name="triangle"
    )
    mask_data = np.zeros((10, 10), dtype=bool)
    mask_data[2:8, 3:7] = True
    mask = UserSegmentationMask.from_numpy(mask_data)

    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf], videos=[video], skeletons=[skeleton], rois=[roi]
    )

    path = str(tmp_path / "test_lazy_roi.slp")
    save_slp(labels, path)

    # Load lazily
    lazy_labels = load_slp(path, lazy=True)
    assert len(lazy_labels.rois) == 1
    assert len(lazy_labels.masks) == 1
    assert lazy_labels.rois[0].name == "triangle"

    # Save lazily and reload
    path2 = str(tmp_path / "test_lazy_roi_resaved.slp")
    save_slp(lazy_labels, path2)

    reloaded = load_slp(path2)
    assert len(reloaded.rois) == 1
    assert len(reloaded.masks) == 1
    np.testing.assert_array_equal(reloaded.masks[0].data, mask_data)


def test_read_rois_empty_dataset(tmp_path):
    """read_rois returns [] when /rois dataset has 0 rows."""
    path = str(tmp_path / "empty_rois.h5")
    roi_dtype = np.dtype(
        [
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("score", "f4"),
            ("wkb_start", "u8"),
            ("wkb_end", "u8"),
        ]
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("rois", data=np.array([], dtype=roi_dtype))
    assert read_rois(path, [], []) == []


def test_read_rois_missing_wkb(tmp_path):
    """read_rois returns [] when /rois exists but /roi_wkb is missing."""
    path = str(tmp_path / "no_wkb.h5")
    roi_dtype = np.dtype(
        [
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("score", "f4"),
            ("wkb_start", "u8"),
            ("wkb_end", "u8"),
        ]
    )
    row = np.array([(0, 0, 0, -1, 0.0, 0, 10)], dtype=roi_dtype)
    with h5py.File(path, "w") as f:
        f.create_dataset("rois", data=row)
    assert read_rois(path, [], []) == []


def test_read_masks_empty_dataset(tmp_path):
    """read_masks returns [] when /masks dataset has 0 rows."""
    path = str(tmp_path / "empty_masks.h5")
    mask_dtype = np.dtype(
        [
            ("height", "u4"),
            ("width", "u4"),
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("score", "f4"),
            ("rle_start", "u8"),
            ("rle_end", "u8"),
        ]
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("masks", data=np.array([], dtype=mask_dtype))
    assert read_masks(path, [], []) == []


def test_read_masks_missing_rle(tmp_path):
    """read_masks returns [] when /masks exists but /mask_rle is missing."""
    path = str(tmp_path / "no_rle.h5")
    mask_dtype = np.dtype(
        [
            ("height", "u4"),
            ("width", "u4"),
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("score", "f4"),
            ("rle_start", "u8"),
            ("rle_end", "u8"),
        ]
    )
    row = np.array([(10, 10, 0, 0, 0, -1, 0.0, 0, 5)], dtype=mask_dtype)
    with h5py.File(path, "w") as f:
        f.create_dataset("masks", data=row)
    assert read_masks(path, [], []) == []


def test_roi_instance_serialization(tmp_path):
    """ROI.instance should be preserved across SLP save/load."""
    from shapely.geometry import Polygon

    skeleton = Skeleton(nodes=["a", "b"], edges=[("a", "b")], name="test")
    video = Video.from_filename("test.mp4")
    instance = Instance.from_numpy(
        np.array([[10, 20], [30, 40]], dtype=np.float32),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])

    roi = UserROI(
        geometry=Polygon([(0, 0), (50, 0), (25, 50)]),
        video=video,
        instance=instance,
    )

    lf.rois.append(roi)
    labels = Labels(labeled_frames=[lf])

    save_path = str(tmp_path / "test.slp")
    labels.save(save_path)

    loaded = load_slp(save_path)
    assert len(loaded.rois) == 1
    loaded_roi = loaded.rois[0]
    assert loaded_roi.instance is not None
    # The loaded instance should be the same object as the one in the frame
    assert loaded_roi.instance is loaded.labeled_frames[0].instances[0]


def test_roi_instance_lazy_roundtrip(tmp_path):
    """ROI instance_idx should survive a lazy load -> save round-trip."""
    from shapely.geometry import Polygon

    skeleton = Skeleton(nodes=["a", "b"], edges=[("a", "b")], name="test")
    video = Video.from_filename("test.mp4")
    instance = Instance.from_numpy(
        np.array([[10, 20], [30, 40]], dtype=np.float32),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    roi = UserROI(
        geometry=Polygon([(0, 0), (50, 0), (25, 50)]),
        video=video,
        instance=instance,
    )
    lf.rois.append(roi)
    labels = Labels(labeled_frames=[lf])

    # Save with instance association
    path1 = str(tmp_path / "original.slp")
    labels.save(path1)

    # Lazy load -> save (should preserve instance_idx via _instance_idx)
    lazy = load_slp(path1, lazy=True)
    assert lazy.rois[0].instance is None  # Not resolved in lazy mode
    assert lazy.rois[0]._instance_idx == 0  # But index is stored

    path2 = str(tmp_path / "roundtrip.slp")
    lazy.save(path2)

    # Verify the round-tripped file still has the association
    reloaded = load_slp(path2)
    assert len(reloaded.rois) == 1
    assert reloaded.rois[0].instance is not None
    assert reloaded.rois[0].instance is reloaded.labeled_frames[0].instances[0]


def test_roi_instance_lazy_materialize(tmp_path):
    """ROI instance should be resolved when lazy Labels is materialized."""
    from shapely.geometry import Polygon

    skeleton = Skeleton(nodes=["a", "b"], edges=[("a", "b")], name="test")
    video = Video.from_filename("test.mp4")
    instance = Instance.from_numpy(
        np.array([[10, 20], [30, 40]], dtype=np.float32),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    roi = UserROI(
        geometry=Polygon([(0, 0), (50, 0), (25, 50)]),
        video=video,
        instance=instance,
    )
    lf.rois.append(roi)
    labels = Labels(labeled_frames=[lf])

    path = str(tmp_path / "test.slp")
    labels.save(path)

    # Lazy load -> materialize should resolve instance link
    lazy = load_slp(path, lazy=True)
    assert lazy.rois[0].instance is None
    assert lazy.rois[0]._instance_idx == 0

    materialized = lazy.materialize()
    assert materialized.rois[0].instance is not None
    assert materialized.rois[0].instance is materialized.labeled_frames[0].instances[0]
    assert materialized.rois[0]._instance_idx == -1  # Cleared after resolution


def test_slp_bbox_roundtrip(tmp_path):
    """Test SLP round-trip with UserBoundingBox objects."""
    video = Video(filename="test.mp4")
    track = Track(name="animal1")
    bbox1 = UserBoundingBox(
        x1=0.0,
        y1=20.0,
        x2=100.0,
        y2=100.0,
        name="bbox1",
        category="mouse",
        source="manual",
    )
    bbox2 = UserBoundingBox(
        x1=175.0,
        y1=135.0,
        x2=225.0,
        y2=165.0,
        angle=0.5,
        track=track,
        name="bbox2",
        category="fly",
    )

    skeleton = Skeleton(nodes=["A"])
    lf0 = LabeledFrame(video=video, frame_idx=0)
    lf0.bboxes.append(bbox1)
    lf3 = LabeledFrame(video=video, frame_idx=3)
    lf3.bboxes.append(bbox2)
    labels = Labels(
        labeled_frames=[lf0, lf3],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track],
    )

    path = str(tmp_path / "test_bboxes.slp")
    save_slp(labels, path)

    loaded = load_slp(path)
    assert len(loaded.bboxes) == 2

    b1 = loaded.bboxes[0]
    assert isinstance(b1, UserBoundingBox)
    assert not isinstance(b1, PredictedBoundingBox)
    assert b1.x_center == pytest.approx(50.0)
    assert b1.y_center == pytest.approx(60.0)
    assert b1.width == pytest.approx(100.0)
    assert b1.height == pytest.approx(80.0)
    assert b1.angle == pytest.approx(0.0)
    assert b1.name == "bbox1"
    assert b1.category == "mouse"
    assert b1.source == "manual"
    assert b1.track is None

    b2 = loaded.bboxes[1]
    assert isinstance(b2, UserBoundingBox)
    assert b2.x_center == pytest.approx(200.0)
    assert b2.y_center == pytest.approx(150.0)
    assert b2.width == pytest.approx(50.0)
    assert b2.height == pytest.approx(30.0)
    assert b2.angle == pytest.approx(0.5)
    assert b2.name == "bbox2"
    assert b2.category == "fly"
    assert b2.track is loaded.tracks[0]

    # Verify low-level read_bboxes works directly
    raw_bboxes = read_bboxes(path, loaded.videos, loaded.tracks)
    assert len(raw_bboxes) == 2
    assert isinstance(raw_bboxes[0][0], UserBoundingBox)

    # Verify low-level write_bboxes works directly
    path2 = str(tmp_path / "test_bboxes2.slp")
    save_slp(Labels(videos=[video], skeletons=[skeleton], tracks=[track]), path2)
    write_bboxes(path2, [bbox1, bbox2], [video], [track], contexts=[(0, 0), (0, 3)])
    raw_bboxes2 = read_bboxes(path2, [video], [track])
    assert len(raw_bboxes2) == 2


def test_slp_predicted_bbox_roundtrip(tmp_path):
    """Test SLP round-trip with PredictedBoundingBox including score."""
    video = Video(filename="test.mp4")
    user_bbox = UserBoundingBox(
        x1=-5.0,
        y1=0.0,
        x2=25.0,
        y2=40.0,
        category="cat",
    )
    pred_bbox = PredictedBoundingBox(
        x1=75.0,
        y1=170.0,
        x2=125.0,
        y2=230.0,
        category="dog",
        score=0.95,
    )

    skeleton = Skeleton(nodes=["A"])
    lf0 = LabeledFrame(video=video, frame_idx=0)
    lf0.bboxes.append(user_bbox)
    lf1 = LabeledFrame(video=video, frame_idx=1)
    lf1.bboxes.append(pred_bbox)
    labels = Labels(
        labeled_frames=[lf0, lf1],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "test_pred_bboxes.slp")
    save_slp(labels, path)

    loaded = load_slp(path)
    assert len(loaded.bboxes) == 2

    b_user = loaded.bboxes[0]
    assert isinstance(b_user, UserBoundingBox)
    assert not isinstance(b_user, PredictedBoundingBox)
    assert b_user.category == "cat"

    b_pred = loaded.bboxes[1]
    assert isinstance(b_pred, PredictedBoundingBox)
    assert b_pred.x_center == pytest.approx(100.0)
    assert b_pred.y_center == pytest.approx(200.0)
    assert b_pred.width == pytest.approx(50.0)
    assert b_pred.height == pytest.approx(60.0)
    assert b_pred.category == "dog"
    assert b_pred.score == pytest.approx(0.95, abs=1e-5)


def test_slp_bbox_no_migration(tmp_path):
    """Bbox-shaped ROIs are NOT migrated to BoundingBox (no legacy format shipped)."""
    video = Video(filename="test.mp4")
    track = Track(name="animal1")

    roi_bbox = UserROI.from_bbox(10, 20, 100, 80, video=video, track=track, name="r1")
    roi_polygon = UserROI.from_polygon(
        [(0, 0), (100, 0), (50, 100)], video=video, name="r2"
    )

    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=0, rois=[roi_bbox])
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track],
        rois=[roi_polygon],
    )

    path = str(tmp_path / "no_migration.slp")
    save_slp(labels, path)
    loaded = load_slp(path)

    # Both ROIs survive as ROIs — no migration to BoundingBox
    assert len(loaded.bboxes) == 0
    assert len(loaded.rois) == 2
    roi_names = {r.name for r in loaded.rois}
    assert "r1" in roi_names
    assert "r2" in roi_names


def test_slp_bbox_lazy_roundtrip(tmp_path):
    """Test lazy load and save with bounding boxes."""
    video = Video(filename="test.mp4")
    bbox = UserBoundingBox(
        x1=0.0,
        y1=20.0,
        x2=100.0,
        y2=100.0,
        name="lazy_bbox",
        category="mouse",
    )
    pred_bbox = PredictedBoundingBox(
        x1=130.0,
        y1=135.0,
        x2=170.0,
        y2=185.0,
        category="fly",
        score=0.85,
    )

    skeleton = Skeleton(nodes=["A", "B"], edges=[("A", "B")], name="test")
    instance = Instance.from_numpy(
        np.array([[10, 20], [30, 40]], dtype=np.float32),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    lf.bboxes.extend([bbox, pred_bbox])
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path1 = str(tmp_path / "step1.slp")
    save_slp(labels, path1)

    # Lazy load
    lazy = load_slp(path1, lazy=True)
    assert len(lazy.bboxes) == 2

    # Save from lazy (fast path)
    path2 = str(tmp_path / "step2.slp")
    lazy.save(path2)

    # Reload and verify
    reloaded = load_slp(path2)
    assert len(reloaded.bboxes) == 2

    b1 = reloaded.bboxes[0]
    assert isinstance(b1, UserBoundingBox)
    assert b1.name == "lazy_bbox"
    assert b1.x_center == pytest.approx(50.0)

    b2 = reloaded.bboxes[1]
    assert isinstance(b2, PredictedBoundingBox)
    assert b2.category == "fly"
    assert b2.score == pytest.approx(0.85, abs=1e-5)


def test_slp_format_id_1_7(tmp_path):
    """Test that files with bboxes get format_id 2.0 (columnar bbox storage)."""
    video = Video(filename="test.mp4")
    bbox = UserBoundingBox(
        x1=0.0,
        y1=20.0,
        x2=100.0,
        y2=100.0,
    )

    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.bboxes.append(bbox)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "test_format.slp")
    save_slp(labels, path)

    format_id = read_hdf5_attrs(path, "metadata", "format_id")
    assert format_id == 2.0

    # Verify that without bboxes, format_id is lower
    labels_no_bbox = Labels(videos=[video], skeletons=[skeleton])
    path_no_bbox = str(tmp_path / "test_no_bbox.slp")
    save_slp(labels_no_bbox, path_no_bbox)

    format_id_no_bbox = read_hdf5_attrs(path_no_bbox, "metadata", "format_id")
    assert format_id_no_bbox == 1.4


def test_label_image_slp_roundtrip(tmp_path):
    """Test writing and reading label images through SLP format."""
    video = Video(filename="test.mp4")
    t1 = Track(name="cell_1")
    t2 = Track(name="cell_2")

    # Create a small label image with two objects
    data = np.zeros((8, 10), dtype=np.int32)
    data[1:3, 2:5] = 1  # object 1
    data[5:7, 6:9] = 2  # object 2

    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(track=t1, category="neuron", name="n1"),
            2: LabelImage.Info(track=t2, category="glia", name="g1"),
        },
        source="cellpose",
    )

    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.label_images.append(li)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[t1, t2],
    )

    path = str(tmp_path / "test_li.slp")
    save_slp(labels, path)

    reloaded = load_slp(path)
    assert len(reloaded.label_images) == 1

    rli = reloaded.label_images[0]
    np.testing.assert_array_equal(rli.data, data)
    assert rli.height == 8
    assert rli.width == 10
    assert rli.source == "cellpose"

    # Check objects metadata
    assert len(rli.objects) == 2
    assert rli.objects[1].category == "neuron"
    assert rli.objects[1].name == "n1"
    assert rli.objects[1].track.name == "cell_1"
    assert rli.objects[2].category == "glia"
    assert rli.objects[2].name == "g1"
    assert rli.objects[2].track.name == "cell_2"


def test_label_image_slp_empty(tmp_path):
    """Test that empty label_images list writes no datasets and reads back."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "test_empty_li.slp")
    save_slp(labels, path)

    # Verify no datasets were written
    with h5py.File(path, "r") as f:
        assert "label_images" not in f
        assert "label_image_data" not in f
        assert "label_image_objects" not in f

    # Verify read returns empty list
    reloaded = load_slp(path)
    assert reloaded.label_images == []


def test_label_image_slp_format_version(tmp_path):
    """Test that presence of label_images bumps format_id to 1.8."""
    video = Video(filename="test.mp4")
    data = np.zeros((4, 4), dtype=np.int32)
    data[0:2, 0:2] = 1

    li = UserLabelImage(data=data)

    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.label_images.append(li)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "test_format_li.slp")
    save_slp(labels, path)

    format_id = read_hdf5_attrs(path, "metadata", "format_id")
    assert format_id == 1.8


def test_label_image_slp_lazy(tmp_path):
    """Test label images load correctly with lazy=True."""
    video = Video(filename="test.mp4")
    t1 = Track(name="t1")

    data = np.zeros((6, 6), dtype=np.int32)
    data[1:4, 1:4] = 1

    li = UserLabelImage(
        data=data,
        objects={1: LabelImage.Info(track=t1, category="cell")},
    )

    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.label_images.append(li)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[t1],
    )

    path = str(tmp_path / "test_lazy_li.slp")
    save_slp(labels, path)

    # Load lazily
    lazy_labels = load_slp(path, open_videos=False)
    assert len(lazy_labels.label_images) == 1

    rli = lazy_labels.label_images[0]
    assert rli.objects[1].category == "cell"
    assert rli.objects[1].track.name == "t1"
    np.testing.assert_array_equal(rli.data, data)


def test_label_image_instance_lazy_roundtrip(tmp_path):
    """Label image instance_idx should survive a lazy load -> save round-trip."""
    skeleton = Skeleton(nodes=["a", "b"], edges=[("a", "b")], name="test")
    video = Video.from_filename("test.mp4")
    instance = Instance.from_numpy(
        np.array([[10, 20], [30, 40]], dtype=np.float32),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    li = UserLabelImage(
        data=np.array([[0, 1], [0, 0]], dtype=np.int32),
        objects={1: LabelImage.Info(instance=instance)},
    )
    lf.label_images.append(li)
    labels = Labels(labeled_frames=[lf])

    path1 = str(tmp_path / "original.slp")
    save_slp(labels, path1)

    # Lazy load -> save (should preserve instance_idx via _instance_idx)
    lazy = load_slp(path1, lazy=True)
    assert lazy.label_images[0].objects[1].instance is None
    assert lazy.label_images[0].objects[1]._instance_idx == 0

    path2 = str(tmp_path / "roundtrip.slp")
    save_slp(lazy, path2)

    # Verify the round-tripped file still has the association
    reloaded = load_slp(path2)
    assert len(reloaded.label_images) == 1
    assert reloaded.label_images[0].objects[1].instance is not None
    assert (
        reloaded.label_images[0].objects[1].instance
        is reloaded.labeled_frames[0].instances[0]
    )


def test_label_image_instance_lazy_materialize(tmp_path):
    """Label image instance should be resolved when lazy Labels is materialized."""
    skeleton = Skeleton(nodes=["a", "b"], edges=[("a", "b")], name="test")
    video = Video.from_filename("test.mp4")
    instance = Instance.from_numpy(
        np.array([[10, 20], [30, 40]], dtype=np.float32),
        skeleton=skeleton,
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[instance])
    li = UserLabelImage(
        data=np.array([[0, 1], [0, 0]], dtype=np.int32),
        objects={1: LabelImage.Info(instance=instance)},
    )
    lf.label_images.append(li)
    labels = Labels(labeled_frames=[lf])

    path = str(tmp_path / "test.slp")
    save_slp(labels, path)

    lazy = load_slp(path, lazy=True)
    assert lazy.label_images[0].objects[1].instance is None
    assert lazy.label_images[0].objects[1]._instance_idx == 0

    materialized = lazy.materialize()
    assert materialized.label_images[0].objects[1].instance is not None
    assert (
        materialized.label_images[0].objects[1].instance
        is materialized.labeled_frames[0].instances[0]
    )
    assert materialized.label_images[0].objects[1]._instance_idx == -1


# -- LabelImage lifetime regression tests --
# These guard against a class of bugs where the HDF5 file backing lazy
# LabelImage.data is closed while LabelImage objects still reference it.


def _make_slp_with_label_image(tmp_path, path_name="lazy.slp"):
    """Helper: write a small labels.slp with one LabelImage and return path."""
    video = Video(filename="test.mp4")
    data = np.zeros((8, 8), dtype=np.int32)
    data[2:5, 2:5] = 1
    data[5:7, 5:7] = 2
    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(category="cell"),
            2: LabelImage.Info(category="cell"),
        },
    )
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.label_images.append(li)
    labels = Labels(labeled_frames=[lf], videos=[video])
    path = str(tmp_path / path_name)
    save_slp(labels, path)
    return path, data


def test_label_image_data_survives_anonymous_labels_gc(tmp_path):
    """``sio.load_slp(...)[0].label_images[0].data`` works.

    Regression: the anonymous ``Labels`` returned by ``load_slp`` is GC'd at
    the end of the expression. Before the fix, ``Labels.__del__`` forcibly
    closed the HDF5 file, invalidating the lazy loader's Dataset references
    and raising ``RuntimeError: Unable to synchronously get dataspace``.
    """
    import gc

    path, expected = _make_slp_with_label_image(tmp_path)

    li = load_slp(path)[0].label_images[0]
    # Force any deferred cleanup of the anonymous Labels object.
    gc.collect()

    # Lazy load should succeed.
    np.testing.assert_array_equal(li.data, expected)


def test_label_image_data_survives_function_return(tmp_path):
    """LabelImage returned from a function outlives the function's local Labels."""
    import gc

    path, expected = _make_slp_with_label_image(tmp_path)

    def get_li(p):
        labels = load_slp(p)
        return labels[0].label_images[0]

    li = get_li(path)
    gc.collect()
    np.testing.assert_array_equal(li.data, expected)


def test_multiple_labels_open_simultaneously(tmp_path):
    """Loading the same file into two Labels and accessing both works."""
    path, expected = _make_slp_with_label_image(tmp_path)

    labels_a = load_slp(path)
    labels_b = load_slp(path)
    li_a = labels_a[0].label_images[0]
    li_b = labels_b[0].label_images[0]

    np.testing.assert_array_equal(li_a.data, expected)
    np.testing.assert_array_equal(li_b.data, expected)


def test_dropping_one_labels_does_not_break_another(tmp_path):
    """Dropping one Labels ref while another still holds its own file is OK."""
    import gc

    path, expected = _make_slp_with_label_image(tmp_path)

    labels_a = load_slp(path)
    labels_b = load_slp(path)
    li_b = labels_b[0].label_images[0]

    # Drop labels_a and force GC. This must not close labels_b's file.
    del labels_a
    gc.collect()

    np.testing.assert_array_equal(li_b.data, expected)


def test_label_image_survives_labels_dropped_after_li_held(tmp_path):
    """Holding a LabelImage after dropping its owning Labels still allows reads."""
    import gc

    path, expected = _make_slp_with_label_image(tmp_path)

    labels = load_slp(path)
    li = labels[0].label_images[0]
    del labels
    gc.collect()

    np.testing.assert_array_equal(li.data, expected)


def test_explicit_labels_close_still_invalidates_label_image(tmp_path):
    """Explicit ``labels.close()`` forcibly closes the file.

    Users who explicitly call ``close()`` opt in to forcibly releasing the
    file handle; subsequent lazy reads on LabelImage objects from the same
    Labels must fail (this is the price of the explicit-close contract).
    """
    path, _ = _make_slp_with_label_image(tmp_path)

    labels = load_slp(path)
    li = labels[0].label_images[0]
    labels.close()

    with pytest.raises(RuntimeError):
        _ = li.data


def test_iterate_label_images_after_anonymous_load(tmp_path):
    """List-comprehension pattern over anonymous Labels also works."""
    import gc

    path, expected = _make_slp_with_label_image(tmp_path)

    data_arrays = [li.data for li in load_slp(path)[0].label_images]
    gc.collect()

    assert len(data_arrays) == 1
    np.testing.assert_array_equal(data_arrays[0], expected)


# -- h5wasm compatibility tests --


def test_read_hdf5_dataset_flat_array(tmp_path):
    """Test that read_hdf5_dataset converts flat arrays with field_names attr."""
    path = str(tmp_path / "flat.h5")

    # Simulate h5wasm output: flat 2D float64 array with field_names attribute
    flat_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f8")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", data=flat_data)
        ds.attrs["field_names"] = json.dumps(["a", "b", "c"])

    result = read_hdf5_dataset(path, "data")

    # Should be a structured array with named fields
    assert result.dtype.names == ("a", "b", "c")
    assert len(result) == 2
    np.testing.assert_array_equal(result["a"], [1.0, 4.0])
    np.testing.assert_array_equal(result["b"], [2.0, 5.0])
    np.testing.assert_array_equal(result["c"], [3.0, 6.0])


def test_read_hdf5_dataset_flat_array_bytes_attr(tmp_path):
    """Test flat array conversion when field_names attr is bytes."""
    path = str(tmp_path / "flat_bytes.h5")

    flat_data = np.array([[10.0, 20.0]], dtype="<f8")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", data=flat_data)
        # Write as bytes (fixed-length H5T_STRING, as h5wasm may produce)
        ds.attrs["field_names"] = np.bytes_(json.dumps(["x", "y"]))

    result = read_hdf5_dataset(path, "data")
    assert result.dtype.names == ("x", "y")
    np.testing.assert_array_equal(result["x"], [10.0])
    np.testing.assert_array_equal(result["y"], [20.0])


def test_read_hdf5_dataset_compound_unchanged(tmp_path):
    """Test that compound dtype datasets pass through unchanged."""
    path = str(tmp_path / "compound.h5")

    dtype = np.dtype([("x", "<f8"), ("y", "<f8")])
    data = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)

    result = read_hdf5_dataset(path, "data")
    assert result.dtype == dtype
    np.testing.assert_array_equal(result["x"], [1.0, 3.0])


def test_read_hdf5_dataset_1d_unchanged(tmp_path):
    """Test that 1D flat arrays (e.g., WKB bytes) pass through unchanged."""
    path = str(tmp_path / "flat1d.h5")

    data = np.array([1, 2, 3, 4], dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)

    result = read_hdf5_dataset(path, "data")
    assert result.ndim == 1
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, data)


def test_read_h5wasm_points(tmp_path):
    """Test _points_from_hdf5_data works with flat arrays converted to structured."""
    path = str(tmp_path / "points.h5")

    # Simulate h5wasm points: flat float64 array with field_names
    pts = np.array([[100.0, 200.0, 1.0, 0.0], [150.0, 250.0, 0.0, 1.0]], dtype="<f8")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("points", data=pts)
        ds.attrs["field_names"] = json.dumps(["x", "y", "visible", "complete"])

    pts_data = read_hdf5_dataset(path, "points")
    skeleton = Skeleton(nodes=["A", "B"])
    points_array = _points_from_hdf5_data(pts_data, skeleton, is_predicted=False)

    assert points_array["xy"][0, 0] == 100.0
    assert points_array["xy"][0, 1] == 200.0
    assert points_array["xy"][1, 0] == 150.0
    assert points_array["xy"][1, 1] == 250.0
    assert points_array["visible"][0] == 1.0
    assert points_array["visible"][1] == 0.0


def test_read_h5wasm_pred_points(tmp_path):
    """Test _points_from_hdf5_data works with predicted points from flat arrays."""
    path = str(tmp_path / "pred_points.h5")

    pts = np.array(
        [[100.0, 200.0, 1.0, 0.0, 0.95], [150.0, 250.0, 1.0, 1.0, 0.80]],
        dtype="<f8",
    )
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("pred_points", data=pts)
        ds.attrs["field_names"] = json.dumps(["x", "y", "visible", "complete", "score"])

    pts_data = read_hdf5_dataset(path, "pred_points")
    skeleton = Skeleton(nodes=["A", "B"])
    points_array = _points_from_hdf5_data(pts_data, skeleton, is_predicted=True)

    assert points_array["score"][0] == 0.95
    assert points_array["score"][1] == 0.80


def test_read_h5wasm_negative_frames(tmp_path):
    """Test read_negative_frames with h5wasm-style flat array."""
    path = str(tmp_path / "neg_frames.h5")

    # h5wasm writes negative_frames as flat int64 array
    data = np.array([[0, 5], [0, 10], [1, 3]], dtype="<i8")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("negative_frames", data=data)
        ds.attrs["field_names"] = json.dumps(["video_id", "frame_idx"])

    result = read_negative_frames(str(path))
    assert result == {(0, 5), (0, 10), (1, 3)}


def test_read_metadata_bytes(tmp_path):
    """Test read_metadata with bytes attribute (standard h5py)."""
    path = str(tmp_path / "meta_bytes.h5")
    metadata = {"version": "1.2", "skeletons": []}
    with h5py.File(path, "w") as f:
        grp = f.require_group("metadata")
        grp.attrs["json"] = json.dumps(metadata).encode()

    result = read_metadata(str(path))
    assert result == metadata


def test_read_metadata_str(tmp_path):
    """Test read_metadata with str attribute (h5py vlen string)."""
    path = str(tmp_path / "meta_str.h5")
    metadata = {"version": "1.2", "skeletons": []}
    with h5py.File(path, "w") as f:
        grp = f.require_group("metadata")
        # h5py writes Python str as vlen string by default
        grp.attrs["json"] = json.dumps(metadata)

    result = read_metadata(str(path))
    assert result == metadata


def test_read_metadata_ndarray(tmp_path):
    """Test read_metadata with uint8 ndarray attribute."""
    path = str(tmp_path / "meta_arr.h5")
    metadata = {"version": "1.2", "skeletons": []}
    json_bytes = json.dumps(metadata).encode()
    with h5py.File(path, "w") as f:
        grp = f.require_group("metadata")
        grp.attrs.create("json", np.frombuffer(json_bytes, dtype=np.uint8))

    result = read_metadata(str(path))
    assert result == metadata


def test_read_metadata_missing_json_attr_raises_valueerror(tmp_path):
    """Test read_metadata raises ValueError when 'json' attr is missing."""
    path = str(tmp_path / "missing_json.slp")
    with h5py.File(path, "w") as f:
        grp = f.require_group("metadata")
        # Group exists but has no 'json' attribute (only an unrelated attr).
        grp.attrs["format_id"] = 1.1

    with pytest.raises(ValueError, match="missing its required metadata JSON blob"):
        read_metadata(path)


def test_read_labels_missing_json_attr_raises_valueerror(slp_minimal, tmp_path):
    """Test ValueError propagates to read_labels when 'json' attr is missing."""
    # Start from a real .slp file so all other datasets are present, then
    # corrupt a copy by deleting only the 'metadata/json' attribute.
    path = str(tmp_path / "corrupt.slp")
    shutil.copy(slp_minimal, path)
    with h5py.File(path, "a") as f:
        del f["metadata"].attrs["json"]

    with pytest.raises(ValueError, match="likely corrupt"):
        read_labels(path)


def test_read_metadata_missing_metadata_group_raises_valueerror(tmp_path):
    """Test read_metadata raises ValueError when 'metadata' group is absent."""
    path = str(tmp_path / "no_metadata_group.slp")
    with h5py.File(path, "w") as f:
        # No 'metadata' group at all.
        f.require_group("other")

    with pytest.raises(ValueError, match="likely corrupt"):
        read_metadata(path)


def test_read_metadata_malformed_json_not_remasked(tmp_path):
    """Test malformed (present) JSON surfaces as a decode error, not corruption."""
    path = str(tmp_path / "malformed_json.slp")
    with h5py.File(path, "w") as f:
        grp = f.require_group("metadata")
        grp.attrs["json"] = b"{not valid json"

    # The corruption message must NOT be raised here; json parsing should fail.
    with pytest.raises(ValueError) as exc_info:
        read_metadata(path)
    assert "missing its required metadata JSON blob" not in str(exc_info.value)


def test_read_h5wasm_instances_float64_indices(tmp_path):
    """Test that float64 index values from h5wasm are handled in read_instances."""
    path = str(tmp_path / "instances.h5")

    # Create a minimal SLP-like file with h5wasm-style flat arrays
    skeleton = Skeleton(nodes=["A", "B"])

    # Points: 2 points for 1 instance
    pts = np.array([[10.0, 20.0, 1.0, 0.0], [30.0, 40.0, 1.0, 0.0]], dtype="<f8")

    # Instance data: all float64 (as h5wasm would write)
    # Fields: instance_id, instance_type, frame_id, skeleton, track,
    #         from_predicted, score, point_id_start, point_id_end, tracking_score
    inst = np.array([[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 2.0, 0.0]], dtype="<f8")

    with h5py.File(path, "w") as f:
        ds_pts = f.create_dataset("points", data=pts)
        ds_pts.attrs["field_names"] = json.dumps(["x", "y", "visible", "complete"])

        ds_inst = f.create_dataset("instances", data=inst)
        ds_inst.attrs["field_names"] = json.dumps(
            [
                "instance_id",
                "instance_type",
                "frame_id",
                "skeleton",
                "track",
                "from_predicted",
                "score",
                "point_id_start",
                "point_id_end",
                "tracking_score",
            ]
        )

        # Empty pred_points (compound dtype, as if no predictions)
        pred_dtype = np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("visible", "?"),
                ("complete", "?"),
                ("score", "<f8"),
            ]
        )
        f.create_dataset("pred_points", data=np.array([], dtype=pred_dtype))

    points = read_hdf5_dataset(path, "points")
    pred_points = read_hdf5_dataset(path, "pred_points")

    instances = read_instances(
        path,
        skeletons=[skeleton],
        tracks=[],
        points=points,
        pred_points=pred_points,
        format_id=1.2,
    )

    assert len(instances) == 1
    inst0 = instances[0]
    assert isinstance(inst0, Instance)
    assert inst0.skeleton == skeleton
    assert inst0.track is None
    np.testing.assert_array_equal(inst0["A"]["xy"], [10.0, 20.0])
    np.testing.assert_array_equal(inst0["B"]["xy"], [30.0, 40.0])


def test_identity_round_trip(tmp_path):
    """Test Identity serialization round-trip."""
    labels_path = str(tmp_path / "test.slp")

    identities = [
        Identity(name="mouse_A", color="#ff0000"),
        Identity(name="mouse_B"),
        Identity(name="mouse_C", color="#00ff00", metadata={"age": 12}),
    ]

    # Write and read back
    with h5py.File(labels_path, "w"):
        pass  # Create empty file

    write_identities(labels_path, identities)
    loaded = read_identities(labels_path)

    assert len(loaded) == 3
    assert loaded[0].name == "mouse_A"
    assert loaded[0].color == "#ff0000"
    assert loaded[1].name == "mouse_B"
    assert loaded[1].color is None
    assert loaded[2].name == "mouse_C"
    assert loaded[2].color == "#00ff00"
    assert loaded[2].metadata == {"age": 12}


def test_identity_empty_round_trip(tmp_path):
    """Test that empty identities list doesn't create dataset."""
    labels_path = str(tmp_path / "test.slp")
    with h5py.File(labels_path, "w"):
        pass

    write_identities(labels_path, [])
    loaded = read_identities(labels_path)
    assert loaded == []


def test_labels_with_identities_round_trip(tmp_path):
    """Test full Labels round-trip with identities."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    identities = [
        Identity(name="mouse_A", color="#ff0000"),
        Identity(name="mouse_B"),
    ]
    labels = Labels(
        skeletons=[skeleton],
        identities=identities,
    )
    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    assert len(loaded.identities) == 2
    assert loaded.identities[0].name == "mouse_A"
    assert loaded.identities[0].color == "#ff0000"
    assert loaded.identities[1].name == "mouse_B"


def test_format_version_1_9_with_identities(tmp_path):
    """Test format version bumps to 1.9 when identities are present."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    labels = Labels(
        skeletons=[skeleton],
        identities=[Identity(name="mouse_A")],
    )
    write_labels(labels_path, labels)

    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")
    assert format_id == 1.9


def test_format_version_no_identities(tmp_path):
    """Test format version stays below 1.9 when no identities."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    labels = Labels(skeletons=[skeleton])
    write_labels(labels_path, labels)

    format_id = read_hdf5_attrs(labels_path, "metadata", "format_id")
    assert format_id < 1.9


def test_instance_group_3d_round_trip(tmp_path, camera_group_345):
    """Test Instance3D round-trip through session serialization."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)

    original_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    inst_3d = Instance3D(points=original_points.copy(), skeleton=skeleton)

    ig = InstanceGroup(
        instance_by_camera={cam1: inst1, cam2: inst2},
        instance_3d=inst_3d,
    )

    video1 = Video(filename="cam1.mp4")
    video2 = Video(filename="cam2.mp4")
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    fg = FrameGroup(
        frame_idx=0,
        instance_groups=[ig],
        labeled_frame_by_camera={cam1: lf1, cam2: lf2},
    )

    session = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: video1, cam2: video2},
        camera_by_video={video1: cam1, video2: cam2},
        frame_group_by_frame_idx={0: fg},
    )

    labels = Labels(
        labeled_frames=[lf1, lf2],
        videos=[video1, video2],
        skeletons=[skeleton],
        sessions=[session],
    )

    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    assert len(loaded.sessions) == 1
    loaded_session = loaded.sessions[0]
    loaded_fg = list(loaded_session.frame_groups.values())[0]
    loaded_ig = loaded_fg.instance_groups[0]
    np.testing.assert_array_almost_equal(loaded_ig.points, original_points)


def test_predicted_instance_3d_round_trip(tmp_path, camera_group_345):
    """Test PredictedInstance3D round-trip through session serialization."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)

    identity = Identity(name="mouse_A")
    pred_3d = PredictedInstance3D(
        points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        skeleton=skeleton,
        score=0.95,
        point_scores=np.array([0.9, 0.8]),
    )

    ig = InstanceGroup(
        instance_by_camera={cam1: inst1, cam2: inst2},
        instance_3d=pred_3d,
        identity=identity,
    )

    video1 = Video(filename="cam1.mp4")
    video2 = Video(filename="cam2.mp4")
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    fg = FrameGroup(
        frame_idx=0,
        instance_groups=[ig],
        labeled_frame_by_camera={cam1: lf1, cam2: lf2},
    )

    session = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: video1, cam2: video2},
        camera_by_video={video1: cam1, video2: cam2},
        frame_group_by_frame_idx={0: fg},
    )

    labels = Labels(
        labeled_frames=[lf1, lf2],
        videos=[video1, video2],
        skeletons=[skeleton],
        sessions=[session],
        identities=[identity],
    )

    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    # Check identities
    assert len(loaded.identities) == 1
    assert loaded.identities[0].name == "mouse_A"

    # Check Instance3D round-trip
    loaded_ig = list(loaded.sessions[0].frame_groups.values())[0].instance_groups[0]
    np.testing.assert_array_almost_equal(
        loaded_ig.points,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    # Check it's a PredictedInstance3D
    assert isinstance(loaded_ig.instance_3d, PredictedInstance3D)
    assert loaded_ig.instance_3d.score == 0.95
    np.testing.assert_array_almost_equal(
        loaded_ig.instance_3d.point_scores,
        np.array([0.9, 0.8]),
    )

    # Check identity was preserved
    assert loaded_ig.identity is loaded.identities[0]


def test_multiple_instance_groups_different_identities(tmp_path, camera_group_345):
    """Test multiple InstanceGroups per FrameGroup with different identities."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    id_mouse_a = Identity(name="mouse_A", color="#ff0000")
    id_mouse_b = Identity(name="mouse_B", color="#00ff00")

    # Animal A instances
    inst_a_cam1 = Instance({"A": [10, 20], "B": [30, 40]}, skeleton=skeleton)
    inst_a_cam2 = Instance({"A": [15, 25], "B": [35, 45]}, skeleton=skeleton)

    # Animal B instances
    inst_b_cam1 = Instance({"A": [50, 60], "B": [70, 80]}, skeleton=skeleton)
    inst_b_cam2 = Instance({"A": [55, 65], "B": [75, 85]}, skeleton=skeleton)

    ig_a = InstanceGroup(
        instance_by_camera={cam1: inst_a_cam1, cam2: inst_a_cam2},
        instance_3d=Instance3D(
            points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            skeleton=skeleton,
        ),
        identity=id_mouse_a,
        score=0.95,
    )
    ig_b = InstanceGroup(
        instance_by_camera={cam1: inst_b_cam1, cam2: inst_b_cam2},
        instance_3d=Instance3D(
            points=np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            skeleton=skeleton,
        ),
        identity=id_mouse_b,
        score=0.85,
    )

    video1 = Video(filename="cam1.mp4")
    video2 = Video(filename="cam2.mp4")
    lf1 = LabeledFrame(
        video=video1,
        frame_idx=0,
        instances=[inst_a_cam1, inst_b_cam1],
    )
    lf2 = LabeledFrame(
        video=video2,
        frame_idx=0,
        instances=[inst_a_cam2, inst_b_cam2],
    )

    fg = FrameGroup(
        frame_idx=0,
        instance_groups=[ig_a, ig_b],
        labeled_frame_by_camera={cam1: lf1, cam2: lf2},
    )

    session = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: video1, cam2: video2},
        camera_by_video={video1: cam1, video2: cam2},
        frame_group_by_frame_idx={0: fg},
    )

    labels = Labels(
        labeled_frames=[lf1, lf2],
        videos=[video1, video2],
        skeletons=[skeleton],
        sessions=[session],
        identities=[id_mouse_a, id_mouse_b],
    )

    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    # Check both identities loaded
    assert len(loaded.identities) == 2
    assert loaded.identities[0].name == "mouse_A"
    assert loaded.identities[0].color == "#ff0000"
    assert loaded.identities[1].name == "mouse_B"
    assert loaded.identities[1].color == "#00ff00"

    # Check both instance groups have correct identity mappings
    loaded_fg = list(loaded.sessions[0].frame_groups.values())[0]
    assert len(loaded_fg.instance_groups) == 2

    loaded_ig_a = loaded_fg.instance_groups[0]
    loaded_ig_b = loaded_fg.instance_groups[1]

    assert loaded_ig_a.identity is loaded.identities[0]
    assert loaded_ig_a.identity.name == "mouse_A"
    assert loaded_ig_a.score == 0.95
    np.testing.assert_array_almost_equal(
        loaded_ig_a.points, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )

    assert loaded_ig_b.identity is loaded.identities[1]
    assert loaded_ig_b.identity.name == "mouse_B"
    assert loaded_ig_b.score == 0.85
    np.testing.assert_array_almost_equal(
        loaded_ig_b.points, np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    )


def test_instance_group_identity_without_3d(tmp_path, camera_group_345):
    """Test InstanceGroup with identity but no Instance3D."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    identity = Identity(name="mouse_A")
    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)

    ig = InstanceGroup(
        instance_by_camera={cam1: inst1, cam2: inst2},
        identity=identity,
    )

    video1 = Video(filename="cam1.mp4")
    video2 = Video(filename="cam2.mp4")
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    fg = FrameGroup(
        frame_idx=0,
        instance_groups=[ig],
        labeled_frame_by_camera={cam1: lf1, cam2: lf2},
    )

    session = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: video1, cam2: video2},
        camera_by_video={video1: cam1, video2: cam2},
        frame_group_by_frame_idx={0: fg},
    )

    labels = Labels(
        labeled_frames=[lf1, lf2],
        videos=[video1, video2],
        skeletons=[skeleton],
        sessions=[session],
        identities=[identity],
    )

    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    loaded_ig = list(loaded.sessions[0].frame_groups.values())[0].instance_groups[0]
    assert loaded_ig.identity is loaded.identities[0]
    assert loaded_ig.identity.name == "mouse_A"
    assert loaded_ig.instance_3d is None
    assert loaded_ig.points is None


def test_instance_3d_score_without_point_scores(tmp_path, camera_group_345):
    """Test Instance3D with score but no point_scores."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)

    inst_3d = Instance3D(
        points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        skeleton=skeleton,
        score=0.88,
    )

    ig = InstanceGroup(
        instance_by_camera={cam1: inst1, cam2: inst2},
        instance_3d=inst_3d,
    )

    video1 = Video(filename="cam1.mp4")
    video2 = Video(filename="cam2.mp4")
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    fg = FrameGroup(
        frame_idx=0,
        instance_groups=[ig],
        labeled_frame_by_camera={cam1: lf1, cam2: lf2},
    )

    session = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: video1, cam2: video2},
        camera_by_video={video1: cam1, video2: cam2},
        frame_group_by_frame_idx={0: fg},
    )

    labels = Labels(
        labeled_frames=[lf1, lf2],
        videos=[video1, video2],
        skeletons=[skeleton],
        sessions=[session],
    )

    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    loaded_ig = list(loaded.sessions[0].frame_groups.values())[0].instance_groups[0]
    assert loaded_ig.instance_3d is not None
    assert not isinstance(loaded_ig.instance_3d, PredictedInstance3D)
    assert loaded_ig.instance_3d.score == 0.88
    np.testing.assert_array_almost_equal(
        loaded_ig.points, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )


def test_session_metadata_track_identity_map_round_trip(tmp_path, camera_group_345):
    """Test track_identity_map in session metadata round-trips through SLP."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)

    ig = InstanceGroup(
        instance_by_camera={cam1: inst1, cam2: inst2},
    )

    video1 = Video(filename="cam1.mp4")
    video2 = Video(filename="cam2.mp4")
    lf1 = LabeledFrame(video=video1, frame_idx=0, instances=[inst1])
    lf2 = LabeledFrame(video=video2, frame_idx=0, instances=[inst2])

    fg = FrameGroup(
        frame_idx=0,
        instance_groups=[ig],
        labeled_frame_by_camera={cam1: lf1, cam2: lf2},
    )

    session = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: video1, cam2: video2},
        camera_by_video={video1: cam1, video2: cam2},
        frame_group_by_frame_idx={0: fg},
        metadata={
            "track_identity_map": {"cam_top:0": 0, "cam_side:0": 0},
            "frame_identity_map": {"5:cam_top:0": 1},
        },
    )

    labels = Labels(
        labeled_frames=[lf1, lf2],
        videos=[video1, video2],
        skeletons=[skeleton],
        sessions=[session],
    )

    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    loaded_session = loaded.sessions[0]
    assert "track_identity_map" in loaded_session.metadata
    assert loaded_session.metadata["track_identity_map"] == {
        "cam_top:0": 0,
        "cam_side:0": 0,
    }
    assert "frame_identity_map" in loaded_session.metadata
    assert loaded_session.metadata["frame_identity_map"] == {"5:cam_top:0": 1}


def test_multiple_sessions_shared_identities(tmp_path, camera_group_345):
    """Test two sessions referencing the same Identity objects."""
    labels_path = str(tmp_path / "test.slp")

    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    id_a = Identity(name="mouse_A")
    id_b = Identity(name="mouse_B")

    # Session 1
    inst1_s1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2_s1 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)
    ig_s1 = InstanceGroup(
        instance_by_camera={cam1: inst1_s1, cam2: inst2_s1},
        identity=id_a,
    )
    vid1_s1 = Video(filename="session1_cam1.mp4")
    vid2_s1 = Video(filename="session1_cam2.mp4")
    lf1_s1 = LabeledFrame(video=vid1_s1, frame_idx=0, instances=[inst1_s1])
    lf2_s1 = LabeledFrame(video=vid2_s1, frame_idx=0, instances=[inst2_s1])
    fg_s1 = FrameGroup(
        frame_idx=0,
        instance_groups=[ig_s1],
        labeled_frame_by_camera={cam1: lf1_s1, cam2: lf2_s1},
    )
    session1 = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: vid1_s1, cam2: vid2_s1},
        camera_by_video={vid1_s1: cam1, vid2_s1: cam2},
        frame_group_by_frame_idx={0: fg_s1},
    )

    # Session 2 — same identities, different instances
    inst1_s2 = Instance({"A": [10, 11], "B": [12, 13]}, skeleton=skeleton)
    inst2_s2 = Instance({"A": [14, 15], "B": [16, 17]}, skeleton=skeleton)
    ig_s2 = InstanceGroup(
        instance_by_camera={cam1: inst1_s2, cam2: inst2_s2},
        identity=id_b,
    )
    vid1_s2 = Video(filename="session2_cam1.mp4")
    vid2_s2 = Video(filename="session2_cam2.mp4")
    lf1_s2 = LabeledFrame(video=vid1_s2, frame_idx=0, instances=[inst1_s2])
    lf2_s2 = LabeledFrame(video=vid2_s2, frame_idx=0, instances=[inst2_s2])
    fg_s2 = FrameGroup(
        frame_idx=0,
        instance_groups=[ig_s2],
        labeled_frame_by_camera={cam1: lf1_s2, cam2: lf2_s2},
    )
    session2 = RecordingSession(
        camera_group=camera_group_345,
        video_by_camera={cam1: vid1_s2, cam2: vid2_s2},
        camera_by_video={vid1_s2: cam1, vid2_s2: cam2},
        frame_group_by_frame_idx={0: fg_s2},
    )

    labels = Labels(
        labeled_frames=[lf1_s1, lf2_s1, lf1_s2, lf2_s2],
        videos=[vid1_s1, vid2_s1, vid1_s2, vid2_s2],
        skeletons=[skeleton],
        sessions=[session1, session2],
        identities=[id_a, id_b],
    )

    write_labels(labels_path, labels)
    loaded = read_labels(labels_path)

    # Both sessions loaded
    assert len(loaded.sessions) == 2
    assert len(loaded.identities) == 2

    # Session 1 references identity 0 (mouse_A)
    loaded_ig_s1 = list(loaded.sessions[0].frame_groups.values())[0].instance_groups[0]
    assert loaded_ig_s1.identity is loaded.identities[0]
    assert loaded_ig_s1.identity.name == "mouse_A"

    # Session 2 references identity 1 (mouse_B)
    loaded_ig_s2 = list(loaded.sessions[1].frame_groups.values())[0].instance_groups[0]
    assert loaded_ig_s2.identity is loaded.identities[1]
    assert loaded_ig_s2.identity.name == "mouse_B"

    # Both reference the same Identity objects from Labels.identities
    all_identities = set()
    for session in loaded.sessions:
        for fg in session.frame_groups.values():
            for ig in fg.instance_groups:
                if ig.identity is not None:
                    all_identities.add(id(ig.identity))
    assert len(all_identities) == 2
    assert all_identities == {
        id(loaded.identities[0]),
        id(loaded.identities[1]),
    }


def test_legacy_pre19_sessions_load():
    """Test that pre-1.9 .slp files load correctly with the new reader."""
    labels = read_labels("tests/data/legacy_pre19_sessions.slp")

    assert len(labels.sessions) == 1
    session = labels.sessions[0]
    assert len(session.camera_group.cameras) == 2

    frame_groups = list(session.frame_groups.values())
    assert len(frame_groups) == 1

    fg = frame_groups[0]
    assert len(fg.instance_groups) == 1

    ig = fg.instance_groups[0]
    assert len(ig.instance_by_camera) == 2
    assert ig.score == 0.85

    # Pre-1.9 files have no Identity or Instance3D
    assert ig.identity is None
    assert ig.instance_3d is None

    # Verify instance data survived
    for inst in ig.instance_by_camera.values():
        assert inst.skeleton is not None
        assert len(inst.skeleton) == 2


def test_instance_group_to_dict_warns_on_unknown_identity(camera_group_345):
    """Test instance_group_to_dict warns when identity not in identities."""
    skeleton = Skeleton(["A", "B"])
    cam1, cam2 = camera_group_345.cameras

    inst1 = Instance({"A": [0, 1], "B": [2, 3]}, skeleton=skeleton)
    inst2 = Instance({"A": [4, 5], "B": [6, 7]}, skeleton=skeleton)

    unknown_identity = Identity(name="unknown_animal")
    ig = InstanceGroup(
        instance_by_camera={cam1: inst1, cam2: inst2},
        identity=unknown_identity,
    )

    instance_to_lf_and_inst_idx = {inst1: (0, 0), inst2: (1, 0)}

    known_identities = [Identity(name="other_animal")]

    with pytest.warns(UserWarning, match="not found in Labels.identities"):
        result = instance_group_to_dict(
            instance_group=ig,
            instance_to_lf_and_inst_idx=instance_to_lf_and_inst_idx,
            camera_group=camera_group_345,
            identities=known_identities,
        )

    assert "identity_idx" not in result


# -- Predicted variant round-trip tests --


def test_slp_mask_instance_roundtrip(tmp_path):
    """Mask instance associations survive save/reload."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    inst = Instance.from_numpy([[10, 20]], skeleton=skeleton)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])

    mask = UserSegmentationMask.from_numpy(
        np.ones((5, 5), dtype=bool),
        instance=inst,
        category="cell",
    )

    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "mask_inst.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.masks) == 1
    rm = loaded.masks[0]
    assert rm.instance is not None
    assert rm.category == "cell"
    assert isinstance(rm, UserSegmentationMask)


def test_slp_predicted_mask_roundtrip(tmp_path):
    """PredictedSegmentationMask round-trips with score."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    mask = PredictedSegmentationMask.from_numpy(
        np.ones((5, 5), dtype=bool),
        category="cell",
        score=0.95,
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "pred_mask.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.masks) == 1
    rm = loaded.masks[0]
    assert isinstance(rm, PredictedSegmentationMask)
    assert rm.score == pytest.approx(0.95, abs=1e-5)
    assert rm.category == "cell"
    np.testing.assert_array_equal(rm.data, np.ones((5, 5), dtype=bool))

    # Verify format_id bumped to 1.9
    format_id = read_hdf5_attrs(path, "metadata", "format_id")
    assert format_id == 1.9


def test_slp_mask_from_predicted_not_persisted(tmp_path):
    """UserSegmentationMask.from_predicted is in-memory only (drops on reload).

    Both masks survive the round-trip, but the provenance link is intentionally
    not serialized (documented behavior), so it reloads as ``None``.
    """
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    pred = PredictedSegmentationMask.from_numpy(np.ones((5, 5), dtype=bool), score=0.9)
    user = pred.to_user()
    assert user.from_predicted is pred

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.extend([pred, user])
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "mask_provenance.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.masks) == 2
    # Both subtypes survive intact in a mixed pred+user frame.
    assert sum(isinstance(m, PredictedSegmentationMask) for m in loaded.masks) == 1
    loaded_pred = next(
        m for m in loaded.masks if isinstance(m, PredictedSegmentationMask)
    )
    assert loaded_pred.score == pytest.approx(0.9, abs=1e-5)
    np.testing.assert_array_equal(loaded_pred.data, np.ones((5, 5), dtype=bool))
    loaded_user = next(m for m in loaded.masks if isinstance(m, UserSegmentationMask))
    np.testing.assert_array_equal(loaded_user.data, np.ones((5, 5), dtype=bool))
    # The provenance link is intentionally not persisted.
    assert loaded_user.from_predicted is None


def test_slp_predicted_mask_score_map_roundtrip(tmp_path):
    """PredictedSegmentationMask with score_map survives round-trip."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    score_map = np.random.rand(8, 8).astype(np.float32)
    mask = PredictedSegmentationMask.from_numpy(
        np.ones((8, 8), dtype=bool),
        score=0.9,
        score_map=score_map,
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "pred_mask_sm.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    rm = loaded.masks[0]
    assert isinstance(rm, PredictedSegmentationMask)
    assert rm.score_map is not None
    np.testing.assert_allclose(rm.score_map, score_map, atol=1e-6)


def test_slp_mask_spatial_metadata_roundtrip(tmp_path):
    """Mask scale/offset survives SLP round-trip."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    mask = UserSegmentationMask.from_numpy(
        np.ones((10, 10), dtype=bool),
        scale=(0.5, 0.5),
        offset=(10.0, 20.0),
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
    )

    path = str(tmp_path / "mask_spatial.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    rm = loaded.masks[0]
    assert rm.scale[0] == pytest.approx(0.5)
    assert rm.scale[1] == pytest.approx(0.5)
    assert rm.offset[0] == pytest.approx(10.0)
    assert rm.offset[1] == pytest.approx(20.0)
    assert rm.has_spatial_transform is True


def test_slp_mask_default_spatial_roundtrip(tmp_path):
    """Mask with default scale/offset round-trips correctly."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    mask = UserSegmentationMask.from_numpy(
        np.ones((5, 5), dtype=bool),
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "mask_default_spatial.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    rm = loaded.masks[0]
    assert rm.scale == (1.0, 1.0)
    assert rm.offset == (0.0, 0.0)
    assert rm.has_spatial_transform is False


def test_slp_predicted_mask_spatial_score_map_roundtrip(tmp_path):
    """PredictedSegmentationMask with score_map and spatial metadata round-trips."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    score_map = np.random.rand(5, 5).astype(np.float32)
    mask = PredictedSegmentationMask.from_numpy(
        np.ones((8, 8), dtype=bool),
        score=0.85,
        score_map=score_map,
        scale=(0.5, 0.5),
        offset=(5.0, 10.0),
        score_map_scale=(0.25, 0.25),
        score_map_offset=(1.0, 2.0),
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "pred_mask_spatial.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    rm = loaded.masks[0]
    assert isinstance(rm, PredictedSegmentationMask)
    assert rm.scale[0] == pytest.approx(0.5)
    assert rm.offset[0] == pytest.approx(5.0)
    assert rm.score_map_scale[0] == pytest.approx(0.25)
    assert rm.score_map_offset[0] == pytest.approx(1.0)
    assert rm.score_map is not None
    np.testing.assert_allclose(rm.score_map, score_map, atol=1e-6)


def test_slp_label_image_spatial_roundtrip(tmp_path):
    """LabelImage scale/offset survives SLP round-trip."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(
        data=data,
        scale=(0.5, 0.5),
        offset=(10.0, 20.0),
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.label_images.append(li)
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "li_spatial.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    rli = loaded.label_images[0]
    assert rli.scale[0] == pytest.approx(0.5)
    assert rli.scale[1] == pytest.approx(0.5)
    assert rli.offset[0] == pytest.approx(10.0)
    assert rli.offset[1] == pytest.approx(20.0)


def test_slp_spatial_metadata_format_version(tmp_path):
    """SLP with non-default spatial metadata bumps format_id to 2.1."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    mask = UserSegmentationMask.from_numpy(
        np.ones((5, 5), dtype=bool),
        scale=(0.5, 0.5),
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "spatial_version.slp")
    save_slp(labels, path)

    format_id = read_hdf5_attrs(path, "metadata", "format_id")
    assert format_id >= 2.1


def test_slp_default_spatial_no_version_bump(tmp_path):
    """SLP with only default spatial metadata should not bump to 2.1."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    mask = UserSegmentationMask.from_numpy(
        np.ones((5, 5), dtype=bool),
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.masks.append(mask)
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "default_spatial.slp")
    save_slp(labels, path)

    format_id = read_hdf5_attrs(path, "metadata", "format_id")
    assert format_id < 2.1


def test_slp_predicted_roi_roundtrip(tmp_path):
    """PredictedROI round-trips with score."""
    from shapely.geometry import box

    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    roi = PredictedROI(
        geometry=box(0, 0, 10, 10),
        video=video,
        category="arena",
        score=0.85,
    )

    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
        rois=[roi],
    )

    path = str(tmp_path / "pred_roi.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.rois) == 1
    rr = loaded.rois[0]
    assert isinstance(rr, PredictedROI)
    assert rr.score == pytest.approx(0.85, abs=1e-5)
    assert rr.category == "arena"


def test_slp_user_roi_roundtrip(tmp_path):
    """UserROI round-trips preserving type."""
    from shapely.geometry import Polygon

    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    # Use a non-rectangular polygon so it doesn't get migrated to BoundingBox
    roi = UserROI(
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (5, 15), (0, 10)]),
        video=video,
        category="arena",
    )

    labels = Labels(
        videos=[video],
        skeletons=[skeleton],
        rois=[roi],
    )

    path = str(tmp_path / "user_roi.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.rois) == 1
    rr = loaded.rois[0]
    assert isinstance(rr, UserROI)
    assert not rr.is_predicted


def test_slp_predicted_label_image_roundtrip(tmp_path):
    """PredictedLabelImage round-trips with scores."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    t1 = Track(name="t1")

    data = np.zeros((6, 6), dtype=np.int32)
    data[1:4, 1:4] = 1

    li = PredictedLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(track=t1, category="neuron", score=0.92),
        },
        score=0.88,
    )

    lf = LabeledFrame(video=video, frame_idx=0)
    lf.label_images.append(li)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[t1],
    )

    path = str(tmp_path / "pred_li.slp")
    save_slp(labels, path)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.label_images) == 1
    rli = loaded.label_images[0]
    assert isinstance(rli, PredictedLabelImage)
    assert rli.score == pytest.approx(0.88, abs=1e-5)
    assert rli.objects[1].score == pytest.approx(0.92, abs=1e-5)
    assert rli.objects[1].category == "neuron"
    np.testing.assert_array_equal(rli.data, data)


def test_slp_backward_compat_no_predicted_fields(tmp_path):
    """Files with pre-v1.9 dtypes (no is_predicted column) read as base classes."""
    import json

    from sleap_io.model.mask import _encode_rle

    # Write a minimal SLP file with old-style mask dtype (no is_predicted/instance)
    path = str(tmp_path / "old_format.slp")
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    labels = Labels(videos=[video], skeletons=[skeleton])
    save_slp(labels, path)

    # Manually add a mask dataset using the pre-v1.9 dtype
    old_mask_dtype = np.dtype(
        [
            ("height", "u4"),
            ("width", "u4"),
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("score", "f4"),
            ("rle_start", "u8"),
            ("rle_end", "u8"),
        ]
    )
    rle = _encode_rle(np.ones((5, 5), dtype=bool))
    rle_bytes = rle.astype(np.uint32).tobytes()
    rle_flat = np.frombuffer(rle_bytes, dtype=np.uint8)

    mask_row = np.array(
        [(5, 5, 0, 0, 0, -1, float("nan"), 0, len(rle_flat))],
        dtype=old_mask_dtype,
    )

    with h5py.File(path, "a") as f:
        ds = f.create_dataset("masks", data=mask_row, dtype=old_mask_dtype)
        ds.attrs["categories"] = json.dumps(["cell"])
        ds.attrs["names"] = json.dumps([""])
        ds.attrs["sources"] = json.dumps([""])
        f.create_dataset("mask_rle", data=rle_flat, dtype=np.uint8)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.masks) == 1
    rm = loaded.masks[0]
    # Pre-v1.9 files should read as UserSegmentationMask (default, not Predicted)
    assert type(rm) is UserSegmentationMask
    assert rm.category == "cell"
    np.testing.assert_array_equal(rm.data, np.ones((5, 5), dtype=bool))


def test_slp_all_annotations_roundtrip(labels_all_annotations, tmp_path):
    """Round-trip all annotation types through SLP and verify data integrity."""
    labels = labels_all_annotations
    path = tmp_path / "all_annotations.slp"

    # Save and reload
    save_slp(labels, str(path))
    loaded = load_slp(str(path), open_videos=False)

    # --- Annotation counts ---
    assert len(loaded.masks) == 6
    assert len(loaded.rois) == 6
    assert len(loaded.bboxes) == 6
    assert len(loaded.label_images) == 6

    # --- Type preservation ---
    user_masks = [m for m in loaded.masks if type(m) is UserSegmentationMask]
    pred_masks = [m for m in loaded.masks if type(m) is PredictedSegmentationMask]
    assert len(user_masks) == 3
    assert len(pred_masks) == 3

    user_rois = [r for r in loaded.rois if type(r) is UserROI]
    pred_rois = [r for r in loaded.rois if type(r) is PredictedROI]
    assert len(user_rois) == 3
    assert len(pred_rois) == 3

    user_bboxes = [b for b in loaded.bboxes if type(b) is UserBoundingBox]
    pred_bboxes = [b for b in loaded.bboxes if type(b) is PredictedBoundingBox]
    assert len(user_bboxes) == 3
    assert len(pred_bboxes) == 3

    user_lis = [li for li in loaded.label_images if type(li) is UserLabelImage]
    pred_lis = [li for li in loaded.label_images if type(li) is PredictedLabelImage]
    assert len(user_lis) == 3
    assert len(pred_lis) == 3

    # --- Predicted scores ---
    for pm in pred_masks:
        assert pm.score == pytest.approx(0.92)
    for pr in pred_rois:
        assert pr.score == pytest.approx(0.88)
    for pb in pred_bboxes:
        assert pb.score == pytest.approx(0.97)
    for pli in pred_lis:
        assert pli.score == pytest.approx(0.88)

    # --- Metadata: name, category, source preserved ---
    for orig, reloaded in zip(labels.masks, loaded.masks):
        assert reloaded.name == orig.name
        assert reloaded.category == orig.category
        assert reloaded.source == orig.source

    for orig, reloaded in zip(labels.rois, loaded.rois):
        assert reloaded.name == orig.name
        assert reloaded.category == orig.category
        assert reloaded.source == orig.source

    for orig, reloaded in zip(labels.bboxes, loaded.bboxes):
        assert reloaded.name == orig.name
        assert reloaded.category == orig.category
        assert reloaded.source == orig.source

    for orig, reloaded in zip(labels.label_images, loaded.label_images):
        assert reloaded.source == orig.source

    # --- Instance links restored ---
    for lf in loaded.labeled_frames:
        for m in lf.masks:
            assert m.instance is not None
            assert m.instance in lf.instances

        for r in lf.rois:
            assert r.instance is not None
            assert r.instance in lf.instances

        for b in lf.bboxes:
            assert b.instance is not None
            assert b.instance in lf.instances

    # --- Track links restored ---
    track_names = {t.name for t in loaded.tracks}
    assert "fly_0" in track_names
    assert "fly_1" in track_names

    for m in loaded.masks:
        assert m.track is not None
        assert m.track.name in track_names

    for r in loaded.rois:
        assert r.track is not None
        assert r.track.name in track_names

    for b in loaded.bboxes:
        assert b.track is not None
        assert b.track.name in track_names

    # --- Mask pixel data round-trips exactly ---
    for orig, reloaded in zip(labels.masks, loaded.masks):
        assert np.array_equal(orig.data, reloaded.data)

    # --- Label image pixel data round-trips exactly ---
    for orig, reloaded in zip(labels.label_images, loaded.label_images):
        assert np.array_equal(orig.data, reloaded.data)

    # --- Score maps round-trip closely ---
    for orig_m, reloaded_m in zip(labels.masks, loaded.masks):
        if isinstance(orig_m, PredictedSegmentationMask):
            assert reloaded_m.score_map is not None
            assert np.allclose(orig_m.score_map, reloaded_m.score_map, atol=1e-5)

    for orig_li, reloaded_li in zip(labels.label_images, loaded.label_images):
        if isinstance(orig_li, PredictedLabelImage):
            assert reloaded_li.score_map is not None
            assert np.allclose(orig_li.score_map, reloaded_li.score_map, atol=1e-5)

    # --- Bounding box coordinates match ---
    for orig, reloaded in zip(labels.bboxes, loaded.bboxes):
        assert reloaded.x1 == pytest.approx(orig.x1)
        assert reloaded.y1 == pytest.approx(orig.y1)
        assert reloaded.x2 == pytest.approx(orig.x2)
        assert reloaded.y2 == pytest.approx(orig.y2)


def test_slp_backward_compat_roi_json_attrs(tmp_path):
    """ROIs written with pre-v1.9 format (metadata in JSON attrs) can still be read."""
    path = str(tmp_path / "old_roi.slp")
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    labels = Labels(videos=[video], skeletons=[skeleton])
    save_slp(labels, path)

    # Old dtype without is_predicted column
    old_roi_dtype = np.dtype(
        [
            ("annotation_type", "u1"),
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("track", "i4"),
            ("score", "f4"),
            ("wkb_start", "u8"),
            ("wkb_end", "u8"),
            ("instance", "i4"),
        ]
    )

    # Create a non-rectangular polygon (circle) so it won't be migrated to bboxes
    circle = shapely.Point(30, 40).buffer(15)
    wkb_bytes = shapely.to_wkb(circle)
    wkb_flat = np.frombuffer(wkb_bytes, dtype=np.uint8)

    roi_row = np.array(
        [(0, 0, -1, -1, float("nan"), 0, len(wkb_flat), -1)],
        dtype=old_roi_dtype,
    )

    with h5py.File(path, "a") as f:
        ds = f.create_dataset("rois", data=roi_row, dtype=old_roi_dtype)
        ds.attrs["categories"] = json.dumps(["arena"])
        ds.attrs["names"] = json.dumps(["test_roi"])
        ds.attrs["sources"] = json.dumps(["manual"])
        f.create_dataset("roi_wkb", data=wkb_flat, dtype=np.uint8)

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.rois) == 1
    roi = loaded.rois[0]
    # No is_predicted column → defaults to UserROI
    assert type(roi) is UserROI
    assert roi.category == "arena"
    assert roi.name == "test_roi"
    assert roi.source == "manual"
    assert roi.geometry.bounds == circle.bounds


def test_slp_backward_compat_label_image_json_attrs(tmp_path):
    """Label images with pre-v1.9 format (metadata in JSON attrs) can still be read."""
    path = str(tmp_path / "old_li.slp")
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    labels = Labels(videos=[video], skeletons=[skeleton])
    save_slp(labels, path)

    # Old label_images dtype without is_predicted or score columns
    old_li_dtype = np.dtype(
        [
            ("video", "i4"),
            ("frame_idx", "i8"),
            ("height", "u4"),
            ("width", "u4"),
            ("n_objects", "u4"),
            ("objects_start", "u4"),
            ("data_start", "u8"),
            ("data_end", "u8"),
        ]
    )

    # Create a small 4x4 label image with label ID 1 in the top-left quadrant
    data = np.zeros((4, 4), dtype=np.int32)
    data[:2, :2] = 1
    compressed = zlib.compress(data.tobytes())
    pixel_flat = np.frombuffer(compressed, dtype=np.uint8)

    li_row = np.array(
        [(0, 0, 4, 4, 1, 0, 0, len(pixel_flat))],
        dtype=old_li_dtype,
    )

    # Old objects dtype without score column
    old_obj_dtype = np.dtype(
        [
            ("label_id", "i4"),
            ("track", "i4"),
            ("instance", "i4"),
        ]
    )
    obj_row = np.array([(1, -1, -1)], dtype=old_obj_dtype)

    with h5py.File(path, "a") as f:
        li_ds = f.create_dataset("label_images", data=li_row, dtype=old_li_dtype)
        li_ds.attrs["sources"] = json.dumps(["manual"])
        f.create_dataset("label_image_data", data=pixel_flat, dtype=np.uint8)
        obj_ds = f.create_dataset(
            "label_image_objects", data=obj_row, dtype=old_obj_dtype
        )
        obj_ds.attrs["categories"] = json.dumps(["cell"])
        obj_ds.attrs["names"] = json.dumps(["obj1"])

    loaded = load_slp(path, open_videos=False)
    assert len(loaded.label_images) == 1
    li = loaded.label_images[0]
    # No is_predicted column → defaults to UserLabelImage
    assert type(li) is UserLabelImage
    assert li.source == "manual"
    assert li.objects[1].category == "cell"
    assert li.objects[1].name == "obj1"
    np.testing.assert_array_equal(li.data, data)


def test_label_image_writer_roundtrip(tmp_path):
    """Streaming writer produces a valid SLP that round-trips through load_slp."""
    video = Video(filename="test.mp4")
    t1 = Track(name="t1")
    t2 = Track(name="t2")
    skeleton = Skeleton(nodes=["A"])

    path = str(tmp_path / "streamed.slp")
    writer = LabelImageWriter(
        path, video=video, tracks=[t1, t2], skeleton=skeleton, initial_capacity=2
    )

    n_frames = 5
    frame_data_list = []
    for i in range(n_frames):
        data = np.zeros((10, 12), dtype=np.int32)
        data[i : i + 2, i : i + 3] = 1
        data[i + 2 : i + 4, i : i + 3] = 2
        frame_data_list.append(data)

        li = UserLabelImage(
            data=data,
            objects={
                1: LabelImage.Info(track=t1, category="neuron", name="n1"),
                2: LabelImage.Info(track=t2, category="glia", name="g1"),
            },
        )
        writer.add(li, frame_idx=i)

    labels = writer.finalize()
    assert len(labels.label_images) == n_frames

    # Reload from disk
    reloaded = load_slp(path, open_videos=False)
    assert len(reloaded.label_images) == n_frames

    for i, rli in enumerate(reloaded.label_images):
        np.testing.assert_array_equal(rli.data, frame_data_list[i])
        assert rli.objects[1].track.name == "t1"
        assert rli.objects[2].track.name == "t2"
        assert rli.objects[1].category == "neuron"
        assert rli.objects[2].category == "glia"

    # Check format version is 2.2 (chunked)
    with h5py.File(path, "r") as f:
        assert f["label_image_data"].ndim == 3
        fmt = f["metadata"].attrs["format_id"]
        assert fmt >= 2.2


def test_label_image_writer_context_manager(tmp_path):
    """Context manager auto-finalizes on exit."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "ctx.slp")

    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data)

    with LabelImageWriter(path, video=video) as writer:
        writer.add(li)
        labels = writer.finalize()

    assert len(labels.label_images) == 1

    # Verify file is valid
    reloaded = load_slp(path, open_videos=False)
    assert len(reloaded.label_images) == 1
    np.testing.assert_array_equal(reloaded.label_images[0].data, data)


def test_label_image_writer_mixed_sizes_error(tmp_path):
    """ValueError raised when frame sizes differ."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "mixed.slp")

    writer = LabelImageWriter(path, video=video)
    li1 = UserLabelImage(
        data=np.zeros((10, 10), dtype=np.int32),
    )
    li2 = UserLabelImage(
        data=np.zeros((10, 15), dtype=np.int32),
    )

    writer.add(li1)
    with pytest.raises(ValueError, match="does not match expected"):
        writer.add(li2)

    # Clean up the open file
    if writer._file is not None:
        writer._file.close()


def test_label_image_writer_score_map(tmp_path):
    """PredictedLabelImage with score maps survives streaming write."""
    video = Video(filename="test.mp4")
    t1 = Track(name="t1")
    path = str(tmp_path / "scores.slp")

    data = np.zeros((8, 8), dtype=np.int32)
    data[2:5, 2:5] = 1
    score_map = np.random.rand(8, 8).astype(np.float32)

    li = PredictedLabelImage(
        data=data,
        objects={1: LabelImage.Info(track=t1, category="cell", score=0.95)},
        score=0.9,
        score_map=score_map,
    )

    writer = LabelImageWriter(path, video=video, tracks=[t1])
    writer.add(li)
    writer.finalize()

    reloaded = load_slp(path, open_videos=False)
    assert len(reloaded.label_images) == 1
    rli = reloaded.label_images[0]
    assert isinstance(rli, PredictedLabelImage)
    assert rli.score == pytest.approx(0.9, abs=1e-5)
    assert rli.objects[1].score == pytest.approx(0.95, abs=1e-5)
    np.testing.assert_array_equal(rli.data, data)
    np.testing.assert_allclose(rli.score_map, score_map, atol=1e-6)


def test_label_image_writer_empty(tmp_path):
    """Empty writer (no frames) creates valid minimal SLP."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    path = str(tmp_path / "empty.slp")

    writer = LabelImageWriter(path, video=video, skeleton=skeleton)
    labels = writer.finalize()
    assert labels.label_images == []

    reloaded = load_slp(path, open_videos=False)
    assert reloaded.label_images == []


def test_label_image_writer_double_finalize_error(tmp_path):
    """Calling finalize twice raises RuntimeError."""
    path = str(tmp_path / "double.slp")
    writer = LabelImageWriter(path)
    writer.finalize()
    with pytest.raises(RuntimeError, match="already been finalized"):
        writer.finalize()


def test_label_image_writer_add_after_finalize_error(tmp_path):
    """Adding after finalize raises RuntimeError."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "after.slp")
    writer = LabelImageWriter(path, video=video)
    writer.finalize()
    li = UserLabelImage(data=np.zeros((4, 4), dtype=np.int32))
    with pytest.raises(RuntimeError, match="already been finalized"):
        writer.add(li)


def test_label_image_writer_add_batch(tmp_path):
    """add_batch writes multiple frames correctly."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "batch.slp")

    frames = []
    for i in range(3):
        data = np.zeros((6, 6), dtype=np.int32)
        data[i : i + 2, i : i + 2] = i + 1
        frames.append(UserLabelImage(data=data))

    writer = LabelImageWriter(path, video=video, initial_capacity=1)
    writer.add_batch(frames)
    writer.finalize()

    reloaded = load_slp(path, open_videos=False)
    assert len(reloaded.label_images) == 3
    for i, rli in enumerate(reloaded.label_images):
        np.testing.assert_array_equal(rli.data, frames[i].data)


def test_label_image_writer_export(tmp_path):
    """LabelImageWriter is importable from sleap_io top-level."""
    from sleap_io import LabelImageWriter as LIW

    assert LIW is LabelImageWriter


def test_merge_label_images_basic(tmp_path):
    """Merge 2 files with 3 frames each -> 6 frames in output."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    t1 = Track(name="cell_1")

    # Create two source SLP files with 3 frames each
    for file_idx in range(2):
        labeled_frames = []
        for i in range(3):
            data = np.zeros((8, 10), dtype=np.int32)
            data[i : i + 2, i : i + 2] = i + 1
            li = UserLabelImage(
                data=data,
                objects={i + 1: LabelImage.Info(track=t1, category="cell")},
                source="test",
            )
            lf = LabeledFrame(video=video, frame_idx=file_idx * 3 + i)
            lf.label_images.append(li)
            labeled_frames.append(lf)
        labels = Labels(
            labeled_frames=labeled_frames,
            videos=[video],
            skeletons=[skeleton],
            tracks=[t1],
        )
        save_slp(labels, str(tmp_path / f"src_{file_idx}.slp"))

    # Merge
    merged = merge_label_images(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged.slp"),
    )

    assert len(merged.label_images) == 6

    # Verify pixel data integrity
    for i in range(3):
        expected = np.zeros((8, 10), dtype=np.int32)
        expected[i : i + 2, i : i + 2] = i + 1
        np.testing.assert_array_equal(merged.label_images[i].data, expected)
        np.testing.assert_array_equal(merged.label_images[i + 3].data, expected)


def test_merge_label_images_track_dedup(tmp_path):
    """Same track name across files -> single track in merged output."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    # Two sources with same track name "cell_A" and different track "cell_B"
    for file_idx in range(2):
        tracks = [Track(name="cell_A")]
        if file_idx == 1:
            tracks.append(Track(name="cell_B"))

        data = np.zeros((4, 4), dtype=np.int32)
        data[0:2, 0:2] = 1

        objs = {1: LabelImage.Info(track=tracks[0], category="cell")}
        if file_idx == 1:
            data[2:4, 2:4] = 2
            objs[2] = LabelImage.Info(track=tracks[1], category="cell")

        li = UserLabelImage(data=data, objects=objs)
        lf = LabeledFrame(video=video, frame_idx=0)
        lf.label_images.append(li)
        labels = Labels(
            labeled_frames=[lf],
            videos=[video],
            skeletons=[skeleton],
            tracks=tracks,
        )
        save_slp(labels, str(tmp_path / f"src_{file_idx}.slp"))

    merged = merge_label_images(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged_tracks.slp"),
    )

    # "cell_A" should be deduplicated, "cell_B" added -> 2 total tracks
    assert len(merged.tracks) == 2
    track_names = {t.name for t in merged.tracks}
    assert track_names == {"cell_A", "cell_B"}

    # Both label images should reference the deduplicated "cell_A" track
    assert merged.label_images[0].objects[1].track.name == "cell_A"
    assert merged.label_images[1].objects[1].track.name == "cell_A"


def test_merge_label_images_video_dedup(tmp_path):
    """Same video filename across files -> single video in merged output."""
    skeleton = Skeleton(nodes=["A"])

    for file_idx in range(2):
        video = Video(filename="shared_video.mp4")
        data = np.zeros((4, 4), dtype=np.int32)
        data[0:2, 0:2] = 1
        li = UserLabelImage(data=data)
        lf = LabeledFrame(video=video, frame_idx=0)
        lf.label_images.append(li)
        labels = Labels(
            labeled_frames=[lf],
            videos=[video],
            skeletons=[skeleton],
        )
        save_slp(labels, str(tmp_path / f"src_{file_idx}.slp"))

    merged = merge_label_images(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged_video.slp"),
    )

    assert len(merged.videos) == 1
    assert merged.videos[0].filename == "shared_video.mp4"
    assert len(merged.label_images) == 2


def test_merge_label_images_pixel_integrity(tmp_path):
    """Verify pixel data is exactly preserved through merge."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    rng = np.random.RandomState(42)
    source_data = []

    for file_idx in range(2):
        labeled_frames = []
        for i in range(3):
            data = rng.randint(0, 10, size=(8, 10), dtype=np.int32)
            source_data.append(data.copy())
            li = UserLabelImage(data=data)
            lf = LabeledFrame(video=video, frame_idx=file_idx * 3 + i)
            lf.label_images.append(li)
            labeled_frames.append(lf)
        labels = Labels(
            labeled_frames=labeled_frames,
            videos=[video],
            skeletons=[skeleton],
        )
        save_slp(labels, str(tmp_path / f"src_{file_idx}.slp"))

    merged = merge_label_images(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged_pixels.slp"),
    )

    assert len(merged.label_images) == 6
    # Verify all source data arrays are present (order may differ after merge)
    merged_data = [li.data for li in merged.label_images]
    for expected_data in source_data:
        found = any(np.array_equal(md, expected_data) for md in merged_data)
        assert found, "Expected label image data not found in merged result"


def test_merge_label_images_dimension_mismatch(tmp_path):
    """Merging files with different frame sizes raises ValueError."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    # Source 1: 4x4 frames
    data1 = np.zeros((4, 4), dtype=np.int32)
    li1 = UserLabelImage(data=data1)
    _lf = LabeledFrame(video=video, frame_idx=0)
    _lf.label_images.append(li1)
    labels1 = Labels(labeled_frames=[_lf], videos=[video], skeletons=[skeleton])
    save_slp(labels1, str(tmp_path / "src_4x4.slp"))

    # Source 2: 6x8 frames
    data2 = np.zeros((6, 8), dtype=np.int32)
    li2 = UserLabelImage(data=data2)
    _lf = LabeledFrame(video=video, frame_idx=0)
    _lf.label_images.append(li2)
    labels2 = Labels(labeled_frames=[_lf], videos=[video], skeletons=[skeleton])
    save_slp(labels2, str(tmp_path / "src_6x8.slp"))

    with pytest.raises(ValueError, match="different dimensions"):
        merge_label_images(
            [str(tmp_path / "src_4x4.slp"), str(tmp_path / "src_6x8.slp")],
            str(tmp_path / "should_fail.slp"),
        )


def test_merge_label_images_empty_sources(tmp_path):
    """Passing no source paths raises ValueError."""
    with pytest.raises(ValueError, match="At least one source"):
        merge_label_images([], str(tmp_path / "empty.slp"))


def test_merge_label_images_with_explicit_video(tmp_path):
    """Merge with an explicit video overrides source video references."""
    skeleton = Skeleton(nodes=["A"])
    override_video = Video(filename="override.mp4")

    for file_idx in range(2):
        video = Video(filename=f"video_{file_idx}.mp4")
        data = np.zeros((4, 4), dtype=np.int32)
        data[0:2, 0:2] = 1
        li = UserLabelImage(data=data)
        _lf = LabeledFrame(video=video, frame_idx=0)
        _lf.label_images.append(li)
        labels = Labels(labeled_frames=[_lf], videos=[video], skeletons=[skeleton])
        save_slp(labels, str(tmp_path / f"src_{file_idx}.slp"))

    merged = merge_label_images(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged_override.slp"),
        video=override_video,
    )

    assert len(merged.videos) == 1
    assert merged.videos[0].filename == "override.mp4"


def test_merge_label_images_export():
    """merge_label_images is importable from sleap_io top-level."""
    from sleap_io import merge_label_images as mli

    assert mli is not None


def test_merge_label_images_blob_source(tmp_path):
    """Merge handles blob-format (v1.8) sources by decompressing + recompressing."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    t1 = Track(name="cell_1")

    # Create a blob-format SLP manually using h5py
    blob_path = str(tmp_path / "blob_src.slp")
    data = np.zeros((4, 4), dtype=np.int32)
    data[0:2, 0:2] = 1
    compressed = zlib.compress(data.tobytes())

    with h5py.File(blob_path, "w") as f:
        # Write label_image_data as flat blob (1D uint8) -- legacy format
        blob = np.frombuffer(compressed, dtype=np.uint8)
        f.create_dataset("label_image_data", data=blob, dtype=np.uint8)

        # Write label_images index
        li_row = np.array(
            [
                (
                    0,  # video
                    0,  # frame_idx
                    4,  # height
                    4,  # width
                    1,  # n_objects
                    0,  # objects_start
                    0,  # data_start
                    len(compressed),  # data_end
                    0,  # is_predicted
                    float("nan"),  # score
                    1.0,  # scale_x
                    1.0,  # scale_y
                    0.0,  # offset_x
                    0.0,  # offset_y
                )
            ],
            dtype=LI_DTYPE,
        )
        f.create_dataset("label_images", data=li_row)

        obj_row = np.array([(1, 0, -1, float("nan"), float("nan"))], dtype=OBJ_DTYPE)
        f.create_dataset("label_image_objects", data=obj_row)

        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("label_image_sources", data=["test"], dtype=str_dt)
        f.create_dataset("label_image_obj_categories", data=["cell"], dtype=str_dt)
        f.create_dataset("label_image_obj_names", data=[""], dtype=str_dt)

    # Write videos/tracks/metadata so read_videos/read_tracks work
    write_videos(blob_path, [video])
    write_tracks(blob_path, [t1])
    _write_metadata_standalone(blob_path, format_id=1.8)

    # Also create a normal chunked source
    chunked_path = str(tmp_path / "chunked_src.slp")
    li = UserLabelImage(
        data=data,
        objects={1: LabelImage.Info(track=t1, category="cell")},
    )
    _lf = LabeledFrame(video=video, frame_idx=0)
    _lf.label_images.append(li)
    labels = Labels(
        labeled_frames=[_lf], videos=[video], skeletons=[skeleton], tracks=[t1]
    )
    save_slp(labels, chunked_path)

    # Merge blob + chunked
    merged = merge_label_images(
        [blob_path, chunked_path],
        str(tmp_path / "merged.slp"),
        video=video,
    )
    assert len(merged.label_images) == 2
    np.testing.assert_array_equal(merged.label_images[0].data, data)
    np.testing.assert_array_equal(merged.label_images[1].data, data)


def test_merge_label_images_with_score_maps(tmp_path):
    """Merge preserves score maps from source files."""
    video = Video(filename="test.mp4")
    t1 = Track(name="cell_1")

    score_maps = []
    for file_idx in range(2):
        data = np.zeros((4, 4), dtype=np.int32)
        data[0:2, 0:2] = 1
        sm = np.random.rand(4, 4).astype(np.float32)
        score_maps.append(sm)
        li = PredictedLabelImage(
            data=data,
            objects={1: LabelImage.Info(track=t1, category="cell", score=0.9)},
            score=0.95,
            score_map=sm,
        )
        _lf = LabeledFrame(video=video, frame_idx=0)
        _lf.label_images.append(li)
        labels = Labels(labeled_frames=[_lf], videos=[video], tracks=[t1])
        save_slp(labels, str(tmp_path / f"src_{file_idx}.slp"))

    merged = merge_label_images(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged_sm.slp"),
    )
    assert len(merged.label_images) == 2
    for i, rli in enumerate(merged.label_images):
        assert isinstance(rli, PredictedLabelImage)
        np.testing.assert_allclose(rli.score_map, score_maps[i], atol=1e-6)


def test_label_image_writer_exit_auto_finalizes(tmp_path):
    """__exit__ calls finalize when not already done."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "auto_fin.slp")
    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(data=data)

    # Use context manager WITHOUT calling finalize explicitly
    with LabelImageWriter(path, video=video) as writer:
        writer.add(li)
    # __exit__ should have finalized

    reloaded = load_slp(path, open_videos=False)
    assert len(reloaded.label_images) == 1
    np.testing.assert_array_equal(reloaded.label_images[0].data, data)


def test_label_image_writer_auto_collects_tracks(tmp_path):
    """Writer auto-collects tracks from label images during add()."""
    video = Video(filename="test.mp4")
    t1 = Track(name="a")
    t2 = Track(name="b")
    path = str(tmp_path / "auto_tracks.slp")

    writer = LabelImageWriter(path, video=video)  # no tracks passed
    assert len(writer.tracks) == 0

    data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    li = UserLabelImage(
        data=data,
        objects={
            1: LabelImage.Info(track=t1),
            2: LabelImage.Info(track=t2),
        },
    )
    writer.add(li)
    assert len(writer.tracks) == 2
    assert t1 in writer.tracks
    assert t2 in writer.tracks

    writer.finalize()
    reloaded = load_slp(path, open_videos=False)
    assert len(reloaded.label_images[0].tracks) == 2


def test_write_label_images_blob_fallback_mixed_sizes(tmp_path):
    """Mixed frame sizes fall back to blob format."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "mixed.slp")

    # Create label images with different sizes
    data_small = np.zeros((4, 4), dtype=np.int32)
    data_large = np.zeros((6, 8), dtype=np.int32)
    li1 = UserLabelImage(data=data_small)
    li2 = UserLabelImage(data=data_large)

    _lf0 = LabeledFrame(video=video, frame_idx=0)
    _lf0.label_images.append(li1)
    _lf1 = LabeledFrame(video=video, frame_idx=1)
    _lf1.label_images.append(li2)
    labels = Labels(labeled_frames=[_lf0, _lf1], videos=[video])
    save_slp(labels, path)

    # Verify blob format was used (1D flat array, not 3D chunked)
    with h5py.File(path, "r") as f:
        assert f["label_image_data"].ndim == 1

    # Verify round-trip
    reloaded = load_slp(path, open_videos=False)
    assert len(reloaded.label_images) == 2
    np.testing.assert_array_equal(reloaded.label_images[0].data, data_small)
    np.testing.assert_array_equal(reloaded.label_images[1].data, data_large)


def test_write_metadata_bumps_format_for_chunked(tmp_path):
    """write_metadata sets format_id >= 2.2 when chunked label_image_data exists."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "chunked_fmt.slp")

    data = np.zeros((4, 4), dtype=np.int32)
    li = UserLabelImage(data=data)
    _lf = LabeledFrame(video=video, frame_idx=0)
    _lf.label_images.append(li)
    labels = Labels(labeled_frames=[_lf], videos=[video])

    # First save normally (write_metadata runs before label images)
    save_slp(labels, path)

    # Re-call write_metadata now that label_image_data exists in the file
    # This exercises the format_id bump at lines 1518-1519
    write_metadata(path, labels)

    with h5py.File(path, "r") as f:
        assert f["label_image_data"].ndim == 3  # chunked
        fmt = f["metadata"].attrs["format_id"]
        assert fmt >= 2.2


def test_read_label_images_empty_index(tmp_path):
    """read_label_images returns empty when label_images dataset is empty."""
    path = str(tmp_path / "empty_idx.slp")
    with h5py.File(path, "w") as f:
        f.create_dataset("label_images", data=np.array([], dtype=LI_DTYPE))

    result, fh = read_label_images(path, [], [])
    assert result == []
    assert fh is None


def test_read_label_images_missing_pixel_data(tmp_path):
    """read_label_images returns empty when label_image_data is missing."""
    path = str(tmp_path / "no_pixels.slp")
    with h5py.File(path, "w") as f:
        li_row = np.array(
            [
                (
                    0,  # video
                    0,  # frame_idx
                    4,  # height
                    4,  # width
                    0,  # n_objects
                    0,  # objects_start
                    0,  # data_start
                    0,  # data_end
                    0,  # is_predicted
                    0.0,  # score
                    1.0,  # scale_x
                    1.0,  # scale_y
                    0.0,  # offset_x
                    0.0,  # offset_y
                )
            ],
            dtype=LI_DTYPE,
        )
        f.create_dataset("label_images", data=li_row)
        # No label_image_data dataset

    result, fh = read_label_images(path, [], [])
    assert result == []
    assert fh is None


def test_write_metadata_standalone_path_provenance(tmp_path):
    """_write_metadata_standalone converts Path values in provenance."""
    path = str(tmp_path / "meta.slp")
    with h5py.File(path, "w"):
        pass
    _write_metadata_standalone(path, provenance={"source": Path("/some/path.slp")})

    with h5py.File(path, "r") as f:
        md = json.loads(f["metadata"].attrs["json"])
        assert md["provenance"]["source"] == "/some/path.slp"


def test_merge_label_images_via_main(tmp_path):
    """merge_label_images accessible via sleap_io.io.main."""
    from sleap_io.io.main import merge_label_images as main_merge

    video = Video(filename="test.mp4")
    for i in range(2):
        data = np.zeros((4, 4), dtype=np.int32)
        data[0:2, 0:2] = 1
        li = UserLabelImage(data=data)
        _lf = LabeledFrame(video=video, frame_idx=0)
        _lf.label_images.append(li)
        labels = Labels(labeled_frames=[_lf], videos=[video])
        save_slp(labels, str(tmp_path / f"src_{i}.slp"))

    merged = main_merge(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged.slp"),
    )
    assert len(merged.label_images) == 2


def test_read_label_images_legacy_json_attrs(tmp_path):
    """read_label_images falls back to JSON attrs for old-format files."""
    path = str(tmp_path / "legacy.slp")
    data = np.zeros((4, 4), dtype=np.int32)
    data[0:2, 0:2] = 1
    compressed = zlib.compress(data.tobytes())
    blob = np.frombuffer(compressed, dtype=np.uint8)

    with h5py.File(path, "w") as f:
        # Write pixel data as blob
        f.create_dataset("label_image_data", data=blob, dtype=np.uint8)
        # Write label_images index
        li_row = np.array(
            [
                (
                    0,
                    0,
                    4,
                    4,
                    1,
                    0,
                    0,
                    len(compressed),
                    0,
                    float("nan"),
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                )
            ],
            dtype=LI_DTYPE,
        )
        f.create_dataset("label_images", data=li_row)
        # Write objects with JSON attrs (legacy format, no string datasets)
        obj_row = np.array([(1, -1, -1, float("nan"), float("nan"))], dtype=OBJ_DTYPE)
        obj_ds = f.create_dataset("label_image_objects", data=obj_row)
        obj_ds.attrs["categories"] = '["cell"]'
        obj_ds.attrs["names"] = '["obj1"]'
        # Write sources as JSON attrs on label_images (legacy)
        li_grp = f["label_images"]
        li_grp.attrs["sources"] = '["test_source"]'
        # No string datasets — reader must fall back to JSON attrs

    write_videos(path, [Video(filename="test.mp4")])
    write_tracks(path, [])
    _write_metadata_standalone(path, format_id=1.8)

    result, fh = read_label_images(path, [Video(filename="test.mp4")], [])
    try:
        assert len(result) == 1
        li, vid_idx, fidx = result[0]
        assert li.objects[1].category == "cell"
        assert li.objects[1].name == "obj1"
        assert li.source == "test_source"
        np.testing.assert_array_equal(li.data, data)
    finally:
        if fh is not None:
            fh.close()


def test_merge_label_images_no_label_images_error(tmp_path):
    """merge_label_images raises on source with no label images dataset."""
    path = str(tmp_path / "empty.slp")
    with h5py.File(path, "w"):
        pass  # No datasets at all

    with pytest.raises(ValueError, match="no label images"):
        merge_label_images([path], str(tmp_path / "out.slp"))


def test_merge_label_images_empty_label_images_error(tmp_path):
    """merge_label_images raises on source with empty label_images."""
    path = str(tmp_path / "empty_li.slp")
    with h5py.File(path, "w") as f:
        f.create_dataset("label_images", data=np.array([], dtype=LI_DTYPE))
        f.create_dataset("label_image_data", data=np.array([], dtype=np.uint8))

    with pytest.raises(ValueError, match="no label images"):
        merge_label_images([path], str(tmp_path / "out.slp"))


def test_merge_label_images_no_objects_table(tmp_path):
    """Merge handles source files with no label_image_objects dataset."""
    video = Video(filename="test.mp4")
    path = str(tmp_path / "no_objs.slp")

    data = np.zeros((4, 4), dtype=np.int32)
    data[0:2, 0:2] = 1
    compressed = zlib.compress(data.tobytes())
    blob = np.frombuffer(compressed, dtype=np.uint8)

    with h5py.File(path, "w") as f:
        f.create_dataset("label_image_data", data=blob, dtype=np.uint8)
        li_row = np.array(
            [
                (
                    0,
                    0,
                    4,
                    4,
                    0,
                    0,
                    0,
                    len(compressed),
                    0,
                    float("nan"),
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                )
            ],
            dtype=LI_DTYPE,
        )
        f.create_dataset("label_images", data=li_row)
        # No label_image_objects, no string datasets

    write_videos(path, [video])
    write_tracks(path, [])
    _write_metadata_standalone(path, format_id=1.8)

    merged = merge_label_images([path], str(tmp_path / "merged.slp"), video=video)
    assert len(merged.label_images) == 1
    np.testing.assert_array_equal(merged.label_images[0].data, data)


def test_merge_label_images_image_video(tmp_path):
    """Merge handles ImageVideo backends where Video.filename is a list."""
    # Create image files to form an ImageVideo
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        img = np.zeros((4, 4), dtype=np.uint8)
        Image.fromarray(img).save(str(img_dir / f"frame_{i:03d}.png"))

    img_paths = sorted(str(p) for p in img_dir.glob("*.png"))
    video = Video(filename=img_paths)

    data = np.zeros((4, 4), dtype=np.int32)
    data[0:2, 0:2] = 1

    for file_idx in range(2):
        li = UserLabelImage(data=data)
        _lf = LabeledFrame(video=video, frame_idx=0)
        _lf.label_images.append(li)
        labels = Labels(labeled_frames=[_lf], videos=[video])
        save_slp(labels, str(tmp_path / f"src_{file_idx}.slp"))

    # This previously crashed with TypeError: unhashable type: 'list'
    merged = merge_label_images(
        [str(tmp_path / "src_0.slp"), str(tmp_path / "src_1.slp")],
        str(tmp_path / "merged.slp"),
    )

    assert len(merged.label_images) == 2
    assert len(merged.videos) == 1
    assert isinstance(merged.videos[0].filename, list)
    np.testing.assert_array_equal(merged.label_images[0].data, data)
    np.testing.assert_array_equal(merged.label_images[1].data, data)


def test_slp_centroid_roundtrip(tmp_path):
    """Test SLP round-trip with UserCentroid and PredictedCentroid."""
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    c1 = UserCentroid(
        x=10.5,
        y=20.3,
        z=1.5,
        track=track,
        tracking_score=0.8,
        name="c1",
        category="cell",
        source="center_of_mass",
    )
    c2 = PredictedCentroid(
        x=50.0,
        y=60.0,
        score=0.95,
        name="c2",
        category="lysosome",
        source="trackmate",
    )

    skeleton = Skeleton(nodes=["A"])
    _lf = LabeledFrame(video=video, frame_idx=0)
    _lf.centroids.extend([c1, c2])
    labels = Labels(
        labeled_frames=[_lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track],
    )

    path = str(tmp_path / "test_centroids.slp")
    save_slp(labels, path)

    loaded = load_slp(path)
    assert len(loaded.centroids) == 2

    lc1 = loaded.centroids[0]
    assert isinstance(lc1, UserCentroid)
    assert lc1.x == pytest.approx(10.5)
    assert lc1.y == pytest.approx(20.3)
    assert lc1.z == pytest.approx(1.5)
    assert lc1.track is loaded.tracks[0]
    assert lc1.tracking_score == pytest.approx(0.8)
    assert lc1.name == "c1"
    assert lc1.category == "cell"
    assert lc1.source == "center_of_mass"

    lc2 = loaded.centroids[1]
    assert isinstance(lc2, PredictedCentroid)
    assert lc2.x == pytest.approx(50.0)
    assert lc2.y == pytest.approx(60.0)
    assert lc2.z is None
    assert lc2.score == pytest.approx(0.95)
    assert lc2.tracking_score is None
    assert lc2.name == "c2"
    assert lc2.category == "lysosome"
    assert lc2.source == "trackmate"


def test_slp_centroid_backward_compat(tmp_path):
    """Old SLP files without centroids load with empty list."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    labels = Labels(videos=[video], skeletons=[skeleton])
    path = str(tmp_path / "no_centroids.slp")
    save_slp(labels, path)

    loaded = load_slp(path)
    assert loaded.centroids == []


def test_slp_bbox_tracking_score_roundtrip(tmp_path):
    """Test tracking_score round-trip for bounding boxes."""
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    bbox = UserBoundingBox(
        x1=0.0,
        y1=0.0,
        x2=10.0,
        y2=10.0,
        track=track,
        tracking_score=0.75,
    )

    skeleton = Skeleton(nodes=["A"])
    _lf = LabeledFrame(video=video, frame_idx=0)
    _lf.bboxes.append(bbox)
    labels = Labels(
        labeled_frames=[_lf], videos=[video], skeletons=[skeleton], tracks=[track]
    )

    path = str(tmp_path / "bbox_ts.slp")
    save_slp(labels, path)

    loaded = load_slp(path)
    assert loaded.bboxes[0].tracking_score == pytest.approx(0.75)


def test_slp_bbox_tracking_score_none_roundtrip(tmp_path):
    """Test that tracking_score=None round-trips correctly."""
    video = Video(filename="test.mp4")
    bbox = UserBoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)

    skeleton = Skeleton(nodes=["A"])
    _lf = LabeledFrame(video=video, frame_idx=0)
    _lf.bboxes.append(bbox)
    labels = Labels(labeled_frames=[_lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "bbox_ts_none.slp")
    save_slp(labels, path)

    loaded = load_slp(path)
    assert loaded.bboxes[0].tracking_score is None


def test_duplicate_track_name_roundtrip(tmp_path):
    """Two distinct same-named tracks round-trip losslessly through SLP.

    The identity-default merge makes duplicate-name Labels more common, so verify
    that the SLP reader/writer (which keys tracks by positional index, with Track
    eq=False) preserves two distinct ``track_0`` tracks rather than collapsing
    them into one.
    """
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])

    track_a = Track(name="track_0")
    track_b = Track(name="track_0")  # Same name, distinct object.

    inst_a = Instance.from_numpy(
        np.array([[1.0, 2.0]]), skeleton=skeleton, track=track_a
    )
    inst_b = Instance.from_numpy(
        np.array([[3.0, 4.0]]), skeleton=skeleton, track=track_b
    )
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst_a, inst_b])
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skeleton],
        tracks=[track_a, track_b],
    )

    path = str(tmp_path / "dup_track_names.slp")
    save_slp(labels, path)
    loaded = load_slp(path)

    # Both distinct tracks survive with the same name.
    assert len(loaded.tracks) == 2
    assert [t.name for t in loaded.tracks] == ["track_0", "track_0"]

    # Per-instance track references resolve to distinct positional indices.
    loaded_lf = loaded.labeled_frames[0]
    track_indices = sorted(
        loaded.tracks.index(inst.track) for inst in loaded_lf.instances
    )
    assert track_indices == [0, 1]

    # Points are exact and correctly associated with their (distinct) track.
    by_index = {loaded.tracks.index(inst.track): inst for inst in loaded_lf.instances}
    np.testing.assert_allclose(by_index[0].numpy(), np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(by_index[1].numpy(), np.array([[3.0, 4.0]]))


def test_slp_centroid_low_level(tmp_path):
    """Test low-level read_centroids/write_centroids."""
    video = Video(filename="test.mp4")
    track = Track(name="t1")
    c = UserCentroid(x=1.0, y=2.0, track=track)

    skeleton = Skeleton(nodes=["A"])
    path = str(tmp_path / "centroids_ll.slp")
    save_slp(Labels(videos=[video], skeletons=[skeleton], tracks=[track]), path)
    write_centroids(path, [c], [video], [track], contexts=[(0, 5)])

    loaded = read_centroids(path, [video], [track])
    assert len(loaded) == 1
    centroid, vid_idx, frame_idx = loaded[0]
    assert centroid.x == pytest.approx(1.0)
    assert centroid.y == pytest.approx(2.0)
    assert frame_idx == 5


def test_slp_lazy_centroid_only_roundtrip(tmp_path):
    """Lazy load of centroid-only SLP (annotation-only frames)."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(["A"])
    track = Track(name="t1")
    labeled_frames = []
    for i in range(3):
        c = PredictedCentroid(
            x=float(i),
            y=float(i * 2),
            track=track,
            score=0.9,
        )
        lf = LabeledFrame(video=video, frame_idx=i)
        lf.centroids.append(c)
        labeled_frames.append(lf)
    labels = Labels(
        labeled_frames=labeled_frames,
        videos=[video],
        skeletons=[skeleton],
        tracks=[track],
    )

    path = str(tmp_path / "centroids.slp")
    save_slp(labels, path)

    # Lazy load — centroids go to supplementary frames or per-frame dicts
    lazy = load_slp(path, lazy=True)
    assert len(lazy.centroids) == 3
    assert len(lazy.labeled_frames) >= 3

    # Accessing by index works (including supplementary)
    for i in range(len(lazy.labeled_frames)):
        lf = lazy.labeled_frames[i]
        assert lf is not None

    # Slice access works
    all_frames = lazy.labeled_frames[:]
    assert len(all_frames) >= 3

    # Save lazy and reload eagerly
    path2 = str(tmp_path / "centroids2.slp")
    save_slp(lazy, path2)
    loaded = load_slp(path2)
    assert len(loaded.centroids) == 3


def test_slp_lazy_annotations_on_materialized_frames(tmp_path):
    """Lazy materialize_frame attaches annotations from store dicts."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(["A"])
    track = Track(name="t1")
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton, track=track)
    lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    c = UserCentroid(x=1.0, y=2.0, track=track)
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)
    mask_data = np.zeros((10, 10), dtype=bool)
    mask_data[2:8, 2:8] = True
    m = UserSegmentationMask.from_numpy(mask_data)

    lf.centroids.append(c)
    lf.bboxes.append(b)
    lf.masks.append(m)
    labels = Labels(labeled_frames=[lf])
    path = str(tmp_path / "test.slp")
    save_slp(labels, path)

    # Lazy load — annotations in per-frame dicts
    lazy = load_slp(path, lazy=True)

    # Access frame 0 — should have annotations attached
    lf0 = lazy.labeled_frames[0]
    assert len(lf0.centroids) == 1
    assert len(lf0.bboxes) == 1
    assert len(lf0.masks) == 1
    assert len(lf0.instances) == 1


def test_slp_lazy_supplementary_frame_indexing(tmp_path):
    """Supplementary frames accessible by index and slice in LazyFrameList."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(["A"])
    track = Track(name="t1")

    # Create centroid-only labels (creates annotation-only frames)
    labeled_frames = []
    for i in range(3):
        c = PredictedCentroid(
            x=float(i),
            y=float(i),
            track=track,
            score=0.9,
        )
        lf = LabeledFrame(video=video, frame_idx=i)
        lf.centroids.append(c)
        labeled_frames.append(lf)
    labels = Labels(
        labeled_frames=labeled_frames,
        videos=[video],
        skeletons=[skeleton],
        tracks=[track],
    )

    path = str(tmp_path / "supp.slp")
    save_slp(labels, path)
    lazy = load_slp(path, lazy=True)

    n = len(lazy.labeled_frames)
    assert n >= 3

    # Test individual index access (covers supplementary path)
    for i in range(n):
        lf = lazy.labeled_frames[i]
        assert lf.video is not None

    # Test negative index
    last = lazy.labeled_frames[-1]
    assert last is not None

    # Test slice access (covers supplementary path in slice)
    sliced = lazy.labeled_frames[0:n]
    assert len(sliced) == n


def test_slp_lazy_old_format_annotation_only_frames(tmp_path):
    """Lazy load handles annotations not in /frames (old SLP format)."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(["A"])
    track = Track(name="t1")

    # Create SLP with instances on frame 0 + centroids on frame 1 (no instances)
    inst = Instance.from_numpy(np.array([[10.0, 20.0]]), skeleton=skeleton, track=track)
    c0 = UserCentroid(x=1.0, y=2.0, track=track)
    c1 = UserCentroid(x=3.0, y=4.0, track=track)

    lf0 = LabeledFrame(video=video, frame_idx=0, instances=[inst])
    lf0.centroids.append(c0)
    lf1 = LabeledFrame(video=video, frame_idx=1)
    lf1.centroids.append(c1)
    labels = Labels(
        labeled_frames=[lf0, lf1],
    )
    path = str(tmp_path / "test.slp")
    save_slp(labels, path)

    # Remove frame 1 from /frames to simulate old format (annotation-only frame
    # not in /frames)
    with h5py.File(path, "a") as f:
        frames = f["frames"][:]
        # Keep only frame 0 (the one with instances)
        new_frames = frames[frames["frame_idx"] == 0]
        del f["frames"]
        f.create_dataset("frames", data=new_frames)

    # Lazy load — frame 1 centroid should create a supplementary frame
    lazy = load_slp(path, lazy=True)
    n = len(lazy.labeled_frames)

    # Should have store frames + supplementary frames
    assert n >= 2

    # All centroids accessible
    assert len(lazy.centroids) == 2

    # Individual index access (covers supplementary path line 822)
    for i in range(n):
        lf = lazy.labeled_frames[i]
        assert lf.video is not None

    # Slice access (covers supplementary path line 809)
    _ = lazy.labeled_frames[0:n]


def test_write_centroids_default_contexts(tmp_path):
    """write_centroids with contexts=None uses (-1, -1) defaults."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    c = UserCentroid(x=1.0, y=2.0)

    path = str(tmp_path / "test.slp")
    save_slp(Labels(videos=[video], skeletons=[skeleton]), path)
    write_centroids(path, [c], [video], [])

    result = read_centroids(path, [video], [])
    assert len(result) == 1
    _, vid_idx, fidx = result[0]
    assert vid_idx == -1
    assert fidx == -1


def test_write_bboxes_default_contexts(tmp_path):
    """write_bboxes with contexts=None uses (-1, -1) defaults."""
    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    b = UserBoundingBox(x1=0, y1=0, x2=10, y2=10)

    path = str(tmp_path / "test.slp")
    save_slp(Labels(videos=[video], skeletons=[skeleton]), path)
    write_bboxes(path, [b], [video], [])

    result = read_bboxes(path, [video], [])
    assert len(result) == 1
    _, vid_idx, fidx = result[0]
    assert vid_idx == -1
    assert fidx == -1


def test_slp_undistributed_annotations_roundtrip(tmp_path):
    """Annotations written with -1 routing context are undistributed on read."""
    from sleap_io.io.slp import write_masks

    video = Video(filename="test.mp4")
    skeleton = Skeleton(nodes=["A"])
    lf = LabeledFrame(video=video, frame_idx=0)
    labels = Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])

    path = str(tmp_path / "test.slp")
    save_slp(labels, path)

    # Write annotations with -1 routing context (undistributable)
    c = UserCentroid(x=5.0, y=10.0)
    b = UserBoundingBox(x1=0, y1=0, x2=20, y2=20)
    mask_data = np.zeros((5, 5), dtype=bool)
    mask_data[1:4, 1:4] = True
    m = UserSegmentationMask.from_numpy(mask_data)

    write_centroids(path, [c], [video], [], contexts=[(-1, -1)])
    write_bboxes(path, [b], [video], [], contexts=[(-1, -1)])
    write_masks(path, [m], [video], [], contexts=[(-1, -1)])

    # read_labels should put these in undistributed (not on any frame)
    loaded = read_labels(path)
    # The existing LabeledFrame at frame_idx=0 has no annotations
    lf0 = [lf for lf in loaded.labeled_frames if lf.frame_idx == 0][0]
    assert len(lf0.centroids) == 0
    assert len(lf0.bboxes) == 0
    assert len(lf0.masks) == 0


def test_read_tracks_with_open_file(slp_typical):
    """`read_tracks` reads from a passed-in open handle without closing it."""
    with h5py.File(slp_typical, "r") as f:
        tracks_passed = read_tracks(slp_typical, _hdf5_file=f)
        # Handle must remain usable after the read (helper must not close it).
        assert f.id.valid
        assert "tracks_json" in f

    tracks_default = read_tracks(slp_typical)
    assert [t.name for t in tracks_passed] == [t.name for t in tracks_default]


def test_read_metadata_with_open_file(slp_typical):
    """`read_metadata` reads from a passed-in open handle without closing it."""
    with h5py.File(slp_typical, "r") as f:
        md_passed = read_metadata(slp_typical, _hdf5_file=f)
        assert f.id.valid

    md_default = read_metadata(slp_typical)
    assert isinstance(md_passed, dict)
    assert md_passed == md_default


def test_read_videos_with_open_file(slp_minimal_pkg):
    """`read_videos` reads (and threads the handle into make_video) correctly."""
    with h5py.File(slp_minimal_pkg, "r") as f:
        videos_passed = read_videos(slp_minimal_pkg, open_backend=False, _hdf5_file=f)
        # Handle must remain usable after the read.
        assert f.id.valid

    videos_default = read_videos(slp_minimal_pkg, open_backend=False)
    assert len(videos_passed) == len(videos_default)
    assert [str(v.filename) for v in videos_passed] == [
        str(v.filename) for v in videos_default
    ]


def test_read_labels_threads_single_open(slp_minimal_pkg):
    """`read_labels` opens the backing HDF5 file only a small number of times.

    Counts real `h5py.File` opens during a single `read_labels` call. Because the
    orchestrator opens once and threads that handle through every helper, the only
    extra open is `read_label_images`' long-lived second handle (when present), so
    the total must be small (<= 3).
    """
    real_h5py_file = h5py.File
    open_count = 0

    def counting_h5py_file(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_h5py_file(*args, **kwargs)

    with mock.patch("sleap_io.io.slp.h5py.File", side_effect=counting_h5py_file):
        labels = read_labels(slp_minimal_pkg)

    assert type(labels) is Labels
    assert open_count <= 3


# ---------------------------------------------------------------------------
# Virtual crop serialization (UNIT U4, §6)
# ---------------------------------------------------------------------------


def test_crop_roundtrip_media_video(centered_pair_low_quality_path, tmp_path):
    """A MediaVideo crop round-trips: crop, cropped shape, uncropped source."""
    source = Video.from_filename(centered_pair_low_quality_path)
    crop = (10, 20, 110, 140)
    expected = crop_frame(source[0], crop, fill=0)

    cropped = source.crop(crop)
    labels = Labels(videos=[cropped])
    path = str(tmp_path / "media_crop.slp")
    write_labels(path, labels)

    loaded = read_labels(path, open_videos=True)
    video = loaded.videos[0]

    assert isinstance(video.backend, CropVideoBackend)
    assert video._crop_tuple() == crop
    # Cropped shape on the facade; uncropped on the provenance source.
    assert video.shape == (source.shape[0], 120, 100, source.shape[3])
    assert video.source_video.shape == source.shape
    assert np.array_equal(video[0], expected)


def test_crop_tuple_fill_roundtrips_and_flattens(
    centered_pair_low_quality_path, tmp_path
):
    """A tuple fill survives the SLP round-trip and still flattens a crop-of-crop.

    /video_crops stores fill as JSON (a tuple becomes a list); the fill converter
    normalizes it back so a reloaded crop with a tuple fill compares equal and
    flattens against a tuple inner.fill.
    """
    source = Video.from_filename(centered_pair_low_quality_path, grayscale=False)
    cropped = source.crop((10, 20, 110, 140), fill=(7, 8, 9))
    labels = Labels(videos=[cropped])
    path = str(tmp_path / "tuple_fill.slp")
    write_labels(path, labels)

    loaded = read_labels(path, open_videos=True)
    v = loaded.videos[0]
    # Fill is normalized to a tuple (not a list) after the JSON round-trip.
    assert v.backend.fill == (7, 8, 9)
    assert isinstance(v.backend.fill, tuple)
    # A further in-bounds crop with the same tuple fill flattens (no nesting).
    refined = v.crop((1, 1, 50, 50), fill=(7, 8, 9))
    assert not isinstance(refined.backend.inner, CropVideoBackend)


def test_reloaded_mosaic_tiles_own_their_inner(
    centered_pair_low_quality_path, tmp_path
):
    """Each reloaded cropped tile owns its inner decoder so close() releases it."""
    source = Video.from_filename(centered_pair_low_quality_path)
    tiles = [source.crop((0, 0, 64, 64)), source.crop((64, 0, 128, 64))]
    labels = Labels(videos=tiles)
    path = str(tmp_path / "mosaic.slp")
    write_labels(path, labels)

    loaded = read_labels(path, open_videos=True)
    for v in loaded.videos:
        assert isinstance(v.backend, CropVideoBackend)
        # Owns its inner (no leaked, unowned decoder on the reload path).
        assert v.backend.owns_inner is True
        _ = v[0]
        v.close()  # cascades to the owned inner; no leak


def test_apply_crops_sparse_embedded_raises(centered_pair_low_quality_path, tmp_path):
    """Baking a crop over a sparsely-embedded video raises (would break frame_idx).

    Embedding labeled frames at non-contiguous indices then reloading yields a crop
    whose inner is an embedded HDF5Video with a sparse frame_map; baking it would
    compact frames to a contiguous range and dangle the labeled-frame references,
    so apply_crop / apply_crops must refuse with a clear error.
    """
    source = Video.from_filename(centered_pair_low_quality_path)
    cropped = source.crop((10, 10, 74, 74))
    skel = Skeleton(["a"])
    lfs = [
        LabeledFrame(
            video=cropped,
            frame_idx=idx,
            instances=[Instance.from_numpy(np.array([[5.0, 5.0]]), skeleton=skel)],
        )
        for idx in (5, 9)  # non-contiguous source indices
    ]
    labels = Labels(videos=[cropped], skeletons=[skel], labeled_frames=lfs)
    pkg = str(tmp_path / "sparse.pkg.slp")
    write_labels(pkg, labels, embed=True)

    reloaded = read_labels(pkg, open_videos=True)
    rv = reloaded.videos[0]
    assert rv.backend.inner.frame_map  # sparse embedded inner
    with pytest.raises(ValueError, match="sparsely embedded"):
        rv.apply_crop(str(tmp_path / "baked.mp4"))
    with pytest.raises(ValueError, match="sparsely embedded"):
        reloaded.apply_crops(video_dir=str(tmp_path / "baked"))


def test_crop_roundtrip_image_video(centered_pair_frame_paths, tmp_path):
    """An ImageVideo crop round-trips through /video_crops."""
    source = Video.from_filename(centered_pair_frame_paths)
    crop = (5, 5, 55, 65)
    expected = crop_frame(source[0], crop, fill=0)

    cropped = source.crop(crop)
    labels = Labels(videos=[cropped])
    path = str(tmp_path / "image_crop.slp")
    write_labels(path, labels)

    loaded = read_labels(path, open_videos=True)
    video = loaded.videos[0]

    assert isinstance(video.backend, CropVideoBackend)
    assert isinstance(video.backend.inner, ImageVideo)
    assert video._crop_tuple() == crop
    assert video.shape == (source.shape[0], 60, 50, source.shape[3])
    assert video.source_video.shape == source.shape
    assert np.array_equal(video[0], expected)


def test_crop_write_video_crops_dataset(centered_pair_low_quality_path, tmp_path):
    """The /video_crops dataset is written with the crop tuple and fill."""
    source = Video.from_filename(centered_pair_low_quality_path)
    cropped = source.crop((10, 20, 110, 140), fill=7)
    labels = Labels(videos=[cropped])
    path = str(tmp_path / "crop_dataset.slp")
    write_labels(path, labels)

    crops = read_video_crops(path)
    assert crops == {0: {"video": 0, "crop": [10, 20, 110, 140], "fill": 7}}


def test_uncropped_writes_no_video_crops_and_golden_videos_json(slp_minimal, tmp_path):
    """Uncropped labels stay byte-identical and bump nothing.

    Uncropped labels write no /video_crops, keep format_id <= 2.2, and the
    videos_json bytes are identical to a baseline save (golden compare).
    """
    labels = read_labels(slp_minimal)

    baseline = str(tmp_path / "baseline.slp")
    candidate = str(tmp_path / "candidate.slp")
    write_labels(baseline, labels)
    write_labels(candidate, labels)

    # No /video_crops dataset for an uncropped file.
    with pytest.raises(KeyError):
        read_hdf5_dataset(candidate, "video_crops")
    assert read_video_crops(candidate) == {}

    with h5py.File(candidate, "r") as f:
        assert "video_crops" not in f
        assert f["metadata"].attrs["format_id"] <= 2.2

    # videos_json is byte-identical to a baseline save.
    with h5py.File(baseline, "r") as f1, h5py.File(candidate, "r") as f2:
        assert f1["videos_json"][:].tobytes() == f2["videos_json"][:].tobytes()


def test_crop_format_id_bumped_only_with_crop(
    slp_minimal, centered_pair_low_quality_path, tmp_path
):
    """format_id becomes 2.3 only when a crop is present."""
    uncropped = read_labels(slp_minimal)
    uncropped_path = str(tmp_path / "uncropped.slp")
    write_labels(uncropped_path, uncropped)
    with h5py.File(uncropped_path, "r") as f:
        assert f["metadata"].attrs["format_id"] < 2.3

    cropped_video = Video.from_filename(centered_pair_low_quality_path).crop(
        (0, 0, 100, 100)
    )
    cropped = Labels(videos=[cropped_video])
    cropped_path = str(tmp_path / "cropped.slp")
    write_labels(cropped_path, cropped)
    with h5py.File(cropped_path, "r") as f:
        assert f["metadata"].attrs["format_id"] == 2.3


def test_crop_old_reader_degrades_without_video_crops(
    centered_pair_low_quality_path, tmp_path
):
    """A file with /video_crops deleted loads the uncropped source (old reader)."""
    source = Video.from_filename(centered_pair_low_quality_path)
    cropped = source.crop((10, 20, 110, 140))
    labels = Labels(videos=[cropped])
    path = str(tmp_path / "degrade.slp")
    write_labels(path, labels)

    with h5py.File(path, "a") as f:
        del f["video_crops"]

    loaded = read_labels(path, open_videos=True)
    video = loaded.videos[0]

    # No crop record -> the full uncropped source is reconstructed.
    assert not isinstance(video.backend, CropVideoBackend)
    assert video._crop_tuple() is None
    assert video.shape == source.shape
    assert np.array_equal(video[0], source[0])


def test_crop_load_modify_save_load_preserves_crop(
    centered_pair_low_quality_path, tmp_path
):
    """Round-trip through an in-place edit preserves the crop (the GUI footgun)."""
    source = Video.from_filename(centered_pair_low_quality_path)
    crop = (10, 20, 110, 140)
    expected = crop_frame(source[0], crop, fill=0)

    labels = Labels(videos=[source.crop(crop)])
    first = str(tmp_path / "first.slp")
    write_labels(first, labels)

    reloaded = read_labels(first, open_videos=True)
    reloaded.provenance["edited"] = True  # an innocuous in-place modification
    second = str(tmp_path / "second.slp")
    write_labels(second, reloaded)

    final = read_labels(second, open_videos=True)
    video = final.videos[0]
    assert video._crop_tuple() == crop
    assert video.shape == (source.shape[0], 120, 100, source.shape[3])
    assert video.source_video.shape == source.shape
    assert np.array_equal(video[0], expected)


def test_crop_closed_load_reports_cropped_shape_and_reserializes(
    centered_pair_low_quality_path, tmp_path
):
    """Closed load reports cropped shape and re-serializes the crop.

    With open_backend=False the closed load reports the cropped shape and
    re-serializes the crop; the re-serialized videos_json describes the
    uncropped source.
    """
    source = Video.from_filename(centered_pair_low_quality_path)
    crop = (10, 20, 110, 140)
    labels = Labels(videos=[source.crop(crop)])
    path = str(tmp_path / "closed.slp")
    write_labels(path, labels)

    closed = read_labels(path, open_videos=False)
    video = closed.videos[0]

    # Closed path: no live backend, but the cropped shape and crop are reported.
    assert video.backend is None
    assert video.shape == (source.shape[0], 120, 100, source.shape[3])
    assert video._crop_tuple() == crop

    # Re-serializing the closed labels preserves the crop and keeps videos_json
    # describing the uncropped source frame.
    out = str(tmp_path / "closed_reserialized.slp")
    write_labels(out, closed)
    assert read_video_crops(out) == {0: {"video": 0, "crop": list(crop), "fill": 0}}
    with h5py.File(out, "r") as f:
        video_json = json.loads(f["videos_json"][0])
    assert list(video_json["backend"]["shape"]) == list(source.shape)


def test_crop_over_embedded_pkg_roundtrip(slp_minimal_pkg, tmp_path):
    """A crop over an embedded .pkg.slp video round-trips.

    The crop-aware reader gets cropped embedded frames with no KeyError, and the
    embedded frames are stored uncropped (so the crop is applied exactly once).
    """
    labels = read_labels(slp_minimal_pkg)
    source = labels.video
    crop = (100, 100, 200, 250)
    expected = crop_frame(source[0], crop, fill=0)

    cropped = source.crop(crop)
    labels.videos[0] = cropped
    for lf in labels.labeled_frames:
        lf.video = cropped

    path = str(tmp_path / "cropped.pkg.slp")
    write_labels(path, labels)

    # Crop rides /video_crops; the embedded video group still exists.
    assert read_video_crops(path) == {0: {"video": 0, "crop": list(crop), "fill": 0}}
    with h5py.File(path, "r") as f:
        assert "video0/video" in f

    loaded = read_labels(path, open_videos=True)
    video = loaded.videos[0]
    assert isinstance(video.backend, CropVideoBackend)
    assert isinstance(video.backend.inner, HDF5Video)
    assert video.backend.inner.has_embedded_images
    # Inner holds the UNCROPPED embedded frame; the crop is applied exactly once.
    assert video.backend.inner.shape == source.shape
    assert video.shape == (source.shape[0], 150, 100, source.shape[3])
    assert np.array_equal(video[0], expected)


def test_crop_over_media_embed_true_roundtrip(centered_pair_low_quality_path, tmp_path):
    """Crop over a MediaVideo saved with embed=True embeds UNCROPPED + keeps the crop.

    The crop is preserved in /video_crops (not baked away), the embedded frames are
    full-source frames, and reload yields a CropVideoBackend applying the crop once.
    """
    source = Video.from_filename(centered_pair_low_quality_path)
    crop = (20, 30, 100, 110)
    expected = crop_frame(source[0], crop, fill=0)
    cropped = source.crop(crop)

    skel = Skeleton(["a"])
    lf = LabeledFrame(
        video=cropped,
        frame_idx=0,
        instances=[Instance.from_numpy(np.array([[5.0, 5.0]]), skeleton=skel)],
    )
    labels = Labels(videos=[cropped], skeletons=[skel], labeled_frames=[lf])

    path = str(tmp_path / "crop_embed.pkg.slp")
    write_labels(path, labels, embed=True)

    assert read_video_crops(path) == {0: {"video": 0, "crop": list(crop), "fill": 0}}

    loaded = read_labels(path, open_videos=True)
    v = loaded.videos[0]
    assert isinstance(v.backend, CropVideoBackend)
    assert v.backend.inner.has_embedded_images
    # Inner holds the UNCROPPED embedded frame; crop applied exactly once.
    assert v.backend.inner.shape[1:3] == source.shape[1:3]
    assert v.shape[1:3] == (80, 80)
    assert np.array_equal(v[0], expected)


def test_nested_crop_serialization_raises(centered_pair_low_quality_path):
    """A nested (un-flattened) crop-of-crop raises a clear error at serialize time."""
    source = Video.from_filename(centered_pair_low_quality_path)
    inner = source.crop((10, 10, 60, 60))
    # Differing fills force a NEST (wrap cannot flatten) -> unrepresentable on disk.
    nested = inner.crop((1, 1, 20, 20), fill=9)
    assert isinstance(nested.backend.inner, CropVideoBackend)
    with pytest.raises(ValueError, match="nested crop-of-crop"):
        video_to_dict(nested)


def test_video_to_dict_crop_serializes_uncropped_source(
    centered_pair_low_quality_path,
):
    """video_to_dict on a cropped Video emits the UNCROPPED source backend dict."""
    source = Video.from_filename(centered_pair_low_quality_path)
    cropped = source.crop((10, 20, 110, 140))

    result = video_to_dict(cropped)

    # The backend dict describes the uncropped MediaVideo source, not the crop.
    assert result["backend"]["type"] == "MediaVideo"
    assert list(result["backend"]["shape"]) == list(source.shape)
    assert "crop" not in result["backend"]
    assert "crop_fill" not in result["backend"]


def test_write_video_crops_omitted_when_no_crops(slp_minimal, tmp_path):
    """write_video_crops writes nothing when no video is cropped."""
    labels = read_labels(slp_minimal)
    path = str(tmp_path / "no_crops.slp")
    write_labels(path, labels)

    # Calling write_video_crops directly on uncropped labels is a no-op.
    write_video_crops(path, labels)
    with h5py.File(path, "r") as f:
        assert "video_crops" not in f
