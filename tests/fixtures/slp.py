"""Fixtures that return paths to .slp files."""

import pytest


@pytest.fixture
def slp_typical():
    """Typical SLP file including."""
    return "tests/data/slp/typical.slp"


@pytest.fixture
def slp_simple_skel():
    """SLP file missing the py/reduce in the skeleton dict."""
    return "tests/data/slp/reduce.slp"


@pytest.fixture
def slp_minimal():
    """SLP project with minimal real data."""
    return "tests/data/slp/minimal_instance.slp"


@pytest.fixture
def slp_minimal_pkg():
    """SLP project with minimal real data and embedded images."""
    return "tests/data/slp/minimal_instance.pkg.slp"


@pytest.fixture
def centered_pair():
    """Example with predicted instances from multiple tracks and a single video.

    This project:
    - Has 1 grayscale video with 1100 frames, cropped to 384x384 with 2 flies
    - Has a 24 node skeleton with edges and symmetries
    - Has 0 user instances and 2274 predicted instances
    - Has 2 correct tracks and 25 extraneous tracks
    """
    return "tests/data/slp/centered_pair_predictions.slp"


@pytest.fixture
def slp_predictions_with_provenance():
    """The slp file generated with the colab tutorial and sleap version 1.2.7."""
    return "tests/data/slp/predictions_1.2.7_provenance_and_tracking.slp"


@pytest.fixture
def skeleton_json_minimal():
    """Minimal skeleton JSON file with 2 nodes and 1 edge.

    This skeleton:
    - Has 2 nodes: head and abdomen
    - Has 1 edge connecting head to abdomen
    - Uses jsonpickle format with py/object and py/state tags
    """
    return "tests/data/slp/labels.v002.rel_paths.skeleton.json"


@pytest.fixture
def skeleton_json_flies():
    """Complex fly skeleton JSON file with 13 nodes, edges and symmetries.

    This skeleton:
    - Has 13 nodes representing fly body parts
    - Has multiple edges connecting body parts
    - Has symmetry relationships between left/right body parts
    - Uses jsonpickle format with py/id references
    """
    return "tests/data/slp/flies13.skeleton.json"


@pytest.fixture
def skeleton_yaml_flies():
    """Complex fly skeleton YAML file with 13 nodes, edges and symmetries.

    This skeleton:
    - Same skeleton as skeleton_json_flies but in YAML format
    - Uses simplified human-readable YAML format
    - Has skeleton names as top-level keys
    - Nodes are simple name lists
    - Edges and symmetries use node names for references
    """
    return "tests/data/slp/flies13.skeleton.yml"


@pytest.fixture
def slp_real_data():
    """A real data example containing predicted and user instances.

    This project:
    - Was generated using SLEAP v1.3.1 on an M1 Mac
    - Contains 1 video (centered_pair_low_quality.mp4)
    - Has a 2 node skeleton of "head" and "abdomen"
    - Has 5 labeled frames
    - Has 10 suggested frames, of which 7 have predictions
    - Has 2 frames with user instances created from predictions:
        - frame_idx 220 has 1 user instance from prediction and 1 predicted instance
        - frame_idx 770 has 2 user instances from predictions
    - Does not have tracks

    Note: There are two versions of these labels, one with absolute paths and one with
    relative paths. The relative paths version is used here:

        >> sio.load_slp("tests/data/slp/labels.v002.rel_paths.slp").video.filename
        "tests/data/videos/centered_pair_low_quality.mp4"
        >> sio.load_slp("tests/data/slp/labels.v002.slp").video.filename
        "/Users/talmo/sleap-io/tests/data/videos/centered_pair_low_quality.mp4"
    """
    return "tests/data/slp/labels.v002.rel_paths.slp"


@pytest.fixture
def slp_imgvideo():
    """SLP project with a single image video."""
    return "tests/data/slp/imgvideo.slp"


@pytest.fixture
def slp_multiview():
    """SLP project with multiple views/videos/cameras of the same scene.

    Labels(
        labeled_frames=24, videos=8, skeletons=1, tracks=2, suggestions=0, sessions=1
    )
    Skeleton(
        nodes=[
            "Nose", "Ear_R", "Ear_L", "TTI", "TailTip", "Head", "Trunk", "Tail_0",
            "Tail_1", "Tail_2", "Shoulder_left", "Shoulder_right", "Haunch_left",
            "Haunch_right", "Neck"
        ],
        edges=[
            (3, 5), (3, 7), (3, 8), (3, 9), (3, 12), (3, 13), (3, 6), (3, 4), (5, 0),
            (5, 14), (5, 10), (5, 11), (5, 1), (5, 2)
        ]
    )
    """
    return "tests/data/slp/multiview.slp"


@pytest.fixture
def skeleton_json_fly32():
    """Fly skeleton JSON file with 32 nodes and 25 edges.

    This skeleton:
    - Has 32 nodes representing detailed fly body parts
    - Has 25 edges connecting body parts
    - Uses jsonpickle format with py/id references
    - Nodes are ordered differently in the nodes array than their py/id assignments
    """
    return "tests/data/slp/fly32.skeleton.json"


@pytest.fixture
def skeleton_yaml_fly32():
    """Fly skeleton YAML file with 32 nodes and 25 edges.

    This skeleton:
    - Same skeleton as skeleton_json_fly32 but in YAML format
    - Uses simplified human-readable YAML format
    - Has skeleton name as top-level key
    - Nodes are simple name lists
    - Edges use node names for references
    """
    return "tests/data/slp/fly32.skeleton.yaml"


@pytest.fixture
def training_config_fly32():
    """Training configuration JSON file with embedded skeleton.

    This file:
    - Contains training configuration for SLEAP
    - Has an embedded skeleton in the 'skeletons' field
    - The skeleton uses jsonpickle format with py/id references
    - Demonstrates how skeletons are embedded in training configs
    """
    return "tests/data/slp/fly32.training_config.json"
