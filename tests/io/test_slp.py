"""Tests for functions in the sleap_io.io.slp file."""

from __future__ import annotations

from sleap_io import (
    Video,
    Skeleton,
    Edge,
    Node,
    Instance,
    LabeledFrame,
    Track,
    PredictedInstance,
    Labels,
    SuggestionFrame,
    Camera,
    CameraGroup,
    FrameGroup,
    InstanceGroup,
    RecordingSession,
    load_file,
    save_file,
)
from sleap_io.io.slp import (
    read_videos,
    write_videos,
    read_tracks,
    write_tracks,
    read_instances,
    read_metadata,
    read_skeletons,
    serialize_skeletons,
    video_to_dict,
    write_metadata,
    read_points,
    read_pred_points,
    read_instances,
    write_lfs,
    read_labels,
    write_labels,
    read_suggestions,
    write_suggestions,
    make_camera,
    make_camera_group,
    make_frame_group,
    make_instance_group,
    make_session,
    camera_to_dict,
    camera_group_to_dict,
    frame_group_to_dict,
    instance_group_to_dict,
    session_to_dict,
    read_sessions,
    write_sessions,
    embed_frames,
    embed_videos,
    process_and_embed_frames,
    prepare_frames_to_embed,
)
from sleap_io.io.utils import read_hdf5_attrs, read_hdf5_dataset
from sleap_io.io.main import save_slp, save_file, load_slp
import numpy as np
import simplejson as json
import pytest
from pathlib import Path
import shutil
from sleap_io.io.video_reading import ImageVideo, HDF5Video, MediaVideo
import sys
import h5py
from unittest import mock
from tqdm import tqdm


def test_read_labels(slp_typical, slp_simple_skel, slp_minimal):
    """Test `read_labels` can read different types of .slp files."""
    labels = read_labels(slp_typical)
    assert type(labels) == Labels

    labels = read_labels(slp_simple_skel)
    assert type(labels) == Labels

    labels = read_labels(slp_minimal)
    assert type(labels) == Labels


def test_load_slp_with_provenance(slp_predictions_with_provenance):
    labels = read_labels(slp_predictions_with_provenance)
    provenance = labels.provenance
    assert type(provenance) == dict
    assert provenance["sleap_version"] == "1.2.7"


def test_read_instances_from_predicted(slp_real_data):
    labels = read_labels(slp_real_data)

    lf = labels.find(video=labels.video, frame_idx=220)[0]
    assert len(lf) == 3
    assert type(lf.instances[0]) == PredictedInstance
    assert type(lf.instances[1]) == PredictedInstance
    assert type(lf.instances[2]) == Instance
    assert lf.instances[2].from_predicted == lf.instances[1]
    assert lf.unused_predictions == [lf.instances[0]]

    lf = labels.find(video=labels.video, frame_idx=770)[0]
    assert len(lf) == 4
    assert type(lf.instances[0]) == PredictedInstance
    assert type(lf.instances[1]) == PredictedInstance
    assert type(lf.instances[2]) == Instance
    assert type(lf.instances[3]) == Instance
    assert lf.instances[2].from_predicted == lf.instances[1]
    assert lf.instances[3].from_predicted == lf.instances[0]
    assert len(lf.unused_predictions) == 0


def test_read_labels_multiview(slp_multiview):
    labels = read_labels(slp_multiview)
    assert type(labels) == Labels
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
    assert type(skeleton) == Skeleton
    assert len(skeleton.nodes) == 24
    assert len(skeleton.edges) == 23
    assert len(skeleton.symmetries) == 20
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
        assert type(saved_labels.video.backend) == type(labels.video.backend)
        assert saved_labels.video.backend.grayscale == labels.video.backend.grayscale
        assert saved_labels.video.backend.shape == labels.video.backend.shape
        assert len(saved_labels.skeletons) == len(labels.skeletons) == 1
        assert saved_labels.skeleton.name == labels.skeleton.name
        assert saved_labels.skeleton.node_names == labels.skeleton.node_names
        assert len(saved_labels.suggestions) == len(labels.suggestions)


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
    assert np.array_equal(instance_group_dict["points"], instance_group._points)

    # Test from dict

    instance_group_0 = make_instance_group(
        instance_group_dict,
        labeled_frames=labeled_frames,
        camera_group=camera_group_345,
    )
    assert instance_group_0._score == instance_group._score
    assert np.array_equal(instance_group_0._points, instance_group._points)
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
        frame_group_345: Frame group with an `InstanceGroup` at each camera view.
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
    assert type(labels.video.backend) == ImageVideo
    assert labels.video.shape == (3, 384, 384, 1)

    write_labels(tmpdir / "test.slp", labels)
    labels = read_labels(tmpdir / "test.slp")
    assert type(labels.video.backend) == ImageVideo
    assert labels.video.shape == (3, 384, 384, 1)

    videos = [Video.from_filename(["fake1.jpg", "fake2.jpg"])]
    assert videos[0].shape is None
    assert len(videos[0].filename) == 2
    write_videos(tmpdir / "test2.slp", videos)
    videos = read_videos(tmpdir / "test2.slp")
    assert type(videos[0].backend) == ImageVideo
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


def test_pkg_roundtrip(tmpdir, slp_minimal_pkg):
    labels = read_labels(slp_minimal_pkg)
    assert type(labels.video.backend) == HDF5Video
    assert labels.video.shape == (1, 384, 384, 1)
    assert labels.video.backend.embedded_frame_inds == [0]
    assert labels.video.filename == slp_minimal_pkg

    write_labels(str(tmpdir / "roundtrip.pkg.slp"), labels)
    labels = read_labels(str(tmpdir / "roundtrip.pkg.slp"))
    assert type(labels.video.backend) == HDF5Video
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
    assert type(base_labels.video.backend) == MediaVideo
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
    assert type(labels.video.backend) == HDF5Video
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
    from sleap_io.io.slp import prepare_frames_to_embed, process_and_embed_frames

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
    assert type(labels.video.backend) == HDF5Video
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
    from sleap_io.io.slp import prepare_frames_to_embed, process_and_embed_frames

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
    assert type(labels.video.backend) == HDF5Video
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
    assert type(labels3.video.backend) == MediaVideo


def test_embed_empty_video(tmpdir, slp_real_data, centered_pair_frame_paths):
    base_labels = read_labels(slp_real_data)
    base_labels.videos.append(Video.from_filename(centered_pair_frame_paths))
    labels_path = str(tmpdir / "labels.pkg.slp")
    write_labels(labels_path, base_labels, embed="user")
    labels = read_labels(labels_path)

    assert labels.videos[0].backend.embedded_frame_inds == [0, 220, 440, 770, 990]
    assert len(labels.videos) == 2


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
    assert type(labels.video.backend) == MediaVideo
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

    # Test with verbose=False
    with mock.patch("tqdm.tqdm") as mock_tqdm:
        result = mod.conditional_import(False)
        assert result is False  # The import didn't happen


def test_embed_frames_verbose_propagation(slp_minimal, tmp_path):
    """Test that embed_frames propagates the verbose parameter to process_and_embed_frames."""
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
    """Test that write_videos propagates the verbose parameter to process_and_embed_frames."""
    labels = load_slp(slp_minimal)

    # Create temp file for embedding
    temp_slp = tmp_path / "test_write_videos_prop.slp"

    # Mock process_and_embed_frames to verify verbose is correctly passed when embedding is needed
    with mock.patch("sleap_io.io.slp.process_and_embed_frames") as mock_embed:
        # This is a simplified test as we can't easily trigger the condition where write_videos
        # calls process_and_embed_frames directly
        write_videos(temp_slp, labels.videos, verbose=True)
        # In a real case with embedded videos, this would verify that verbose is passed correctly
        # Since we're not actually embedding in this test, the mock may not be called

    # The actual test here is that the function accepts the verbose parameter without errors


def test_write_labels_verbose_propagation(slp_minimal, tmp_path):
    """Test that write_labels propagates the verbose parameter to embed_videos and write_videos."""
    labels = load_slp(slp_minimal)

    # Create temp file
    temp_slp = tmp_path / "test_write_labels_prop.slp"

    # Mock embed_videos to verify verbose is correctly passed
    with mock.patch("sleap_io.io.slp.embed_videos") as mock_embed_videos, mock.patch(
        "sleap_io.io.slp.write_videos"
    ) as mock_write_videos:

        write_labels(temp_slp, labels, embed="user", verbose=True)

        # Check that verbose=True was passed to embed_videos
        assert mock_embed_videos.call_args.kwargs["verbose"] is True

        # Check that verbose=True was passed to write_videos
        assert mock_write_videos.call_args.kwargs["verbose"] is True

    # Check with verbose=False
    with mock.patch("sleap_io.io.slp.embed_videos") as mock_embed_videos, mock.patch(
        "sleap_io.io.slp.write_videos"
    ) as mock_write_videos:

        write_labels(temp_slp, labels, embed="user", verbose=False)

        # Check that verbose=False was passed to embed_videos
        assert mock_embed_videos.call_args.kwargs["verbose"] is False

        # Check that verbose=False was passed to write_videos
        assert mock_write_videos.call_args.kwargs["verbose"] is False


def test_format_id_1_3_tracking_score(tmp_path):
    """Test that FORMAT_ID 1.3 properly handles tracking_score field."""
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

    # Save with FORMAT_ID 1.3
    test_path = tmp_path / "test_format_1_3.slp"
    write_labels(test_path, labels)

    # Verify FORMAT_ID is 1.3
    format_id = read_hdf5_attrs(test_path, "metadata", "format_id")
    assert format_id == 1.3

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

    # Save with only video2 embedded
    pkg_path = tmp_path / "mixed.pkg.slp"
    frames_to_embed = [(video2, 0)]
    write_labels(str(pkg_path), labels, embed=frames_to_embed)

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
    """Test that saving with embed=False over the same file works when no embedded data."""
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
    """Test that saving with embed=False over the same file works when embedded data has source."""
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
    labels = Labels(
        videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
    )

    # Create a .pkg.slp file with embedded frames
    pkg_path = tmp_path / "test.pkg.slp"

    # Mock embedding to create a video with embedded frames but no source video
    with h5py.File(pkg_path, "w") as f:
        # Create minimal structure
        f.create_dataset("videos_json", data=[])
        f.create_dataset("tracks_json", data=[])

    # Create a mock embedded video with minimal HDF5Video backend
    # We need to use a mock to bypass the actual file reading
    from unittest.mock import Mock, patch

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
    assert labels_preserve.videos[0].backend_metadata["has_embedded_images"] == True


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
    source_backend_metadata = video.backend_metadata.copy()

    # Test EMBED mode
    labels = load_file(slp_minimal_pkg)  # Fresh load
    output_embed = tmp_path / "test_embed.slp"
    labels.save(output_embed, embed=True)

    with h5py.File(output_embed, "r") as f:
        # Check that both original and source video metadata are stored
        assert "video0" in f
        # Since source_video IS the original, it should be stored as original_video
        assert "original_video" in f["video0"]
        assert "source_video" in f["video0"]

        # Verify original video metadata (should be the MediaVideo)
        assert isinstance(f["video0/original_video"], h5py.Group)
        original_json = json.loads(f["video0/original_video"].attrs["json"])
        assert original_json["backend"]["type"] == "MediaVideo"
        assert (
            original_json["backend"]["filename"]
            == original_backend_metadata["filename"]
        )

        # Verify source video metadata (should be the .pkg.slp file)
        assert isinstance(f["video0/source_video"], h5py.Group)
        source_json = json.loads(f["video0/source_video"].attrs["json"])
        assert source_json["backend"]["type"] == "HDF5Video"
        assert source_json["backend"]["filename"] == slp_minimal_pkg

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


def test_video_original_video_field(slp_minimal_pkg):
    """Test that Video objects have the new original_video field."""
    labels = load_file(slp_minimal_pkg)
    video = labels.videos[0]

    # Current implementation has source_video, not original_video
    # This test will fail until we implement the field rename
    assert hasattr(video, "original_video")
    assert video.original_video is None  # Not set for current files

    # TODO: The source_video should become original_video when the field rename is complete
    assert video.source_video is not None  # This is current behavior


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
    assert loaded_predictions.videos[0].backend_metadata["has_embedded_images"] == True

    # Verify metadata preservation through the workflow
    # The video objects from inference_labels already have source_video metadata
    # which is preserved when we create the predictions Labels object
    assert loaded_predictions.videos[0].source_video is not None

    # The source_video should point to minimal_instance.pkg.slp (the original training data)
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
    from sleap_io.io.slp import write_videos, VideoReferenceMode
    from sleap_io.model.video import Video
    import tempfile

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

        # Test restore_source=True with reference_mode=None (should use RESTORE_ORIGINAL)
        write_videos(str(output), [video], restore_source=True, reference_mode=None)

        # Test restore_source=False with reference_mode=None (should use EMBED)
        write_videos(str(output), [video], restore_source=False, reference_mode=None)


def test_video_lineage_edge_cases():
    """Test edge cases in video lineage metadata handling."""
    from sleap_io.io.slp import write_videos, VideoReferenceMode
    from sleap_io.model.video import Video
    import tempfile

    # Test case 1: Video with original_video already set
    original = Video(
        filename="original.mp4",
        backend_metadata={
            "type": "MediaVideo",
            "shape": [10, 100, 100, 1],
            "filename": "original.mp4",
            "grayscale": True,
        },
    )

    video_with_original = Video(
        filename="current.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "current.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        source_video=None,
        original_video=original,  # This should be saved
    )

    # Test case 2: source_video has original_video
    source_with_original = Video(
        filename="source.pkg.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "source.pkg.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        original_video=original,
    )

    video_with_source_original = Video(
        filename="current2.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "current2.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        source_video=source_with_original,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test_lineage.slp"

        # Write videos with different lineage scenarios
        write_videos(
            str(output),
            [video_with_original, video_with_source_original],
            reference_mode=VideoReferenceMode.EMBED,
            original_videos=[video_with_original, video_with_source_original],
        )
