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
)
from sleap_io.io.utils import read_hdf5_attrs, read_hdf5_dataset
import numpy as np
import simplejson as json
import pytest
from pathlib import Path
import shutil
from sleap_io.io.video_reading import ImageVideo, HDF5Video, MediaVideo
import sys


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
