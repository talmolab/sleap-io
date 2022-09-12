import pytest
from pathlib import Path
import datetime

import numpy as np
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule

from sleap_io import load_slp
from sleap_io import write_labels_to_nwb, append_labels_data_to_nwb


@pytest.fixture
def nwbfile():
    session_description: str = "Testing session for nwb"
    session_start_time = datetime.datetime.now(datetime.timezone.utc)
    identifier = "identifier"

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
    )

    return nwbfile


def test_typical_case_append(nwbfile, slp_typical):
    labels = load_slp(slp_typical)
    nwbfile = append_labels_data_to_nwb(labels, nwbfile)

    # Test matching number of processing modules
    number_of_videos = len(labels.videos)
    assert len(nwbfile.processing) == number_of_videos

    # Test processing module naming
    video_index = 0
    video = labels.videos[video_index]
    video_path = Path(video.filename)
    processing_module_name = f"SLEAP_VIDEO_{video_index:03}_{video_path.stem}"
    assert processing_module_name in nwbfile.processing

    processing_module = nwbfile.processing[processing_module_name]
    all_containers = processing_module.data_interfaces
    # Test name of PoseEstimation containers
    # In this case the predicted instances are not tracked.
    container_name = "track=untracked"
    assert container_name in all_containers

    # Test that the skeleton nodes are store as nodes in containers
    pose_estimation_container = all_containers[container_name]
    expected_node_names = [node.name for node in labels.skeletons[0]]
    assert expected_node_names == pose_estimation_container.nodes

    # Test that each PoseEstimationSeries is named as a node
    for node_name in pose_estimation_container.nodes:
        assert node_name in pose_estimation_container.pose_estimation_series


def test_typical_case_append_with_metadata_propagation(nwbfile, slp_typical):
    labels = load_slp(slp_typical)

    pose_estimation_metadata = {
        "source_software": "1.2.3",  # Sleap-version, I chosen a random one for the test
        "dimensions": [
            [384, 384]
        ],  # The dimensions of the video frame extracted using ffmpeg probe
    }

    nwbfile = append_labels_data_to_nwb(labels, nwbfile, pose_estimation_metadata)

    # Test processing module naming
    video_index = 0
    video = labels.videos[video_index]
    video_path = Path(video.filename)
    processing_module_name = f"SLEAP_VIDEO_{video_index:03}_{video_path.stem}"

    processing_module = nwbfile.processing[processing_module_name]
    pose_estimation_container = processing_module.data_interfaces["track=untracked"]

    # Test pose estimation metadata propagation
    extracted_source_software = pose_estimation_container.source_software
    expected_source_software = pose_estimation_metadata["source_software"]
    assert extracted_source_software == expected_source_software

    extracted_dimensions = pose_estimation_container.dimensions
    expected_dimensions = pose_estimation_metadata["dimensions"]
    assert extracted_dimensions == expected_dimensions


def test_complex_case_append(nwbfile, slp_predictions):
    labels = load_slp(slp_predictions)
    nwbfile = append_labels_data_to_nwb(labels, nwbfile)

    # Test matching number of processing modules
    number_of_videos = len(labels.videos)
    assert len(nwbfile.processing) == number_of_videos

    # Test processing module naming
    video_index = 0
    video = labels.videos[video_index]
    video_path = Path(video.filename)
    processing_module_name = f"SLEAP_VIDEO_{video_index:03}_{video_path.stem}"
    assert processing_module_name in nwbfile.processing

    # For this case we have as many containers as tracks
    processing_module = nwbfile.processing[processing_module_name]
    all_containers = processing_module.data_interfaces
    assert len(all_containers) == len(labels.tracks)

    # Test name of PoseEstimation containers
    extracted_container_names = all_containers.keys()
    for track in labels.tracks:
        expected_track_name = f"track={track.name}"
        assert expected_track_name in extracted_container_names

    # Test one PoseEstimation container
    container_name = "track=1"
    pose_estimation_container = all_containers[container_name]
    # Test that the skeleton nodes are store as nodes in containers
    expected_node_names = [node.name for node in labels.skeletons[0]]
    assert expected_node_names == pose_estimation_container.nodes

    # Test that each PoseEstimationSeries is named as a node
    for node_name in pose_estimation_container.nodes:
        assert node_name in pose_estimation_container.pose_estimation_series


def test_complex_case_append_with_timestamps_metadata(nwbfile, slp_predictions):
    labels = load_slp(slp_predictions)

    number_of_frames = 1100  # extracted using ffmpeg probe
    video_sample_rate = 15.0  # 15 Hz extracted using ffmpeg probe for the video stream
    video_timestamps = np.arange(number_of_frames) / video_sample_rate

    pose_estimation_metadata = {
        "video_timestamps": video_timestamps,
    }

    nwbfile = append_labels_data_to_nwb(labels, nwbfile, pose_estimation_metadata)

    # Test processing module naming
    video_index = 0
    video = labels.videos[video_index]
    video_path = Path(video.filename)
    processing_module_name = f"SLEAP_VIDEO_{video_index:03}_{video_path.stem}"

    processing_module = nwbfile.processing[processing_module_name]
    # Test one PoseEstimationContainer
    container_name = "track=1"
    pose_estimation_container = processing_module.data_interfaces[container_name]

    # Test sampling rate propagation. In this case the timestamps are uniform so
    # The sampling rate should be stored instead of them
    expected_rate = video_sample_rate
    for node_name in pose_estimation_container.nodes:
        pose_estimation_series = pose_estimation_container.pose_estimation_series[
            node_name
        ]

        # Some store rate and it should be the video_sample_rate
        if pose_estimation_series.rate:
            extracted_rate = pose_estimation_series.rate
            assert extracted_rate == expected_rate, f"{node_name}"

        # Other store timestamps and the timestmaps should be a subset of the videotimestamps
        else:
            extracted_timestamps = pose_estimation_series.timestamps
            assert np.in1d(
                extracted_timestamps, video_timestamps, assume_unique=True
            ).all()


def test_assertion_with_no_predicted_instance(nwbfile, slp_minimal):
    labels = load_slp(slp_minimal)
    with pytest.raises(
        ValueError, match="No predicted instances found in labels object"
    ):
        nwbfile = append_labels_data_to_nwb(labels, nwbfile)


def test_typical_case_write(slp_typical, tmp_path):
    labels = load_slp(slp_typical)

    nwbfile_path = tmp_path / "write_to_nwb_typical_case.nwb"
    write_labels_to_nwb(labels=labels, nwbfile_path=nwbfile_path)

    with NWBHDF5IO(str(nwbfile_path), "r") as io:
        nwbfile = io.read()

        # Test matching number of processing modules
        number_of_videos = len(labels.videos)
        assert len(nwbfile.processing) == number_of_videos
