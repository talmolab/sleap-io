"""Tests for functions in the sleap_io.io.main file."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from sleap_io import Labels, LabelsSet, RemoteIOError, Video, clear_remote_cache
from sleap_io.io._remote import _require_package
from sleap_io.io.main import (
    load_file,
    load_jabs,
    load_labels_set,
    load_labelstudio,
    load_nwb,
    load_slp,
    load_video,
    save_file,
    save_jabs,
    save_labelstudio,
    save_nwb,
    save_video,
)


def test_load_slp(slp_typical):
    """Test `load_slp` loads a .slp to a `Labels` object."""
    assert type(load_slp(slp_typical)) is Labels
    assert type(load_file(slp_typical)) is Labels


def test_nwb(tmp_path, slp_typical, slp_real_data):
    # Test with predictions (slp_typical has predictions)
    labels_pred = load_slp(slp_typical)
    save_nwb(labels_pred, tmp_path / "test_nwb_pred.nwb")
    loaded_labels = load_nwb(tmp_path / "test_nwb_pred.nwb")
    assert type(loaded_labels) is Labels
    assert type(load_file(tmp_path / "test_nwb_pred.nwb")) is Labels

    # Test with annotations (slp_real_data has user instances)
    labels_ann = load_slp(slp_real_data)
    save_nwb(labels_ann, tmp_path / "test_nwb_ann.nwb")
    loaded_labels = load_nwb(tmp_path / "test_nwb_ann.nwb")
    assert type(loaded_labels) is Labels

    # Test overwriting (no append)
    save_nwb(labels_pred, tmp_path / "test_nwb_pred.nwb")  # Overwrites
    loaded_labels = load_nwb(tmp_path / "test_nwb_pred.nwb")
    assert type(loaded_labels) is Labels


def test_labelstudio(tmp_path, slp_typical):
    labels = load_slp(slp_typical)
    save_labelstudio(labels, tmp_path / "test_labelstudio.json")
    loaded_labels = load_labelstudio(tmp_path / "test_labelstudio.json")
    assert type(loaded_labels) is Labels
    assert type(load_file(tmp_path / "test_labelstudio.json")) is Labels
    assert len(loaded_labels) == len(labels)


def test_jabs(tmp_path, jabs_real_data_v2, jabs_real_data_v5):
    labels_single = load_jabs(jabs_real_data_v2)
    assert isinstance(labels_single, Labels)
    save_jabs(labels_single, 2, tmp_path)
    labels_single_written = load_jabs(str(tmp_path / jabs_real_data_v2))
    # Confidence field is not preserved, so just check number of labels
    assert len(labels_single) == len(labels_single_written)
    assert len(labels_single.videos) == len(labels_single_written.videos)
    assert type(load_file(jabs_real_data_v2)) is Labels

    labels_multi = load_jabs(jabs_real_data_v5)
    assert isinstance(labels_multi, Labels)
    save_jabs(labels_multi, 3, tmp_path)
    save_jabs(labels_multi, 4, tmp_path)
    save_jabs(labels_multi, 5, tmp_path)
    labels_v5_written = load_jabs(str(tmp_path / jabs_real_data_v5))
    # v5 contains all v4 and v3 data, so only need to check v5
    # Confidence field and ordering of identities is not preserved, so just check
    # number of labels
    assert len(labels_v5_written) == len(labels_multi)
    assert len(labels_v5_written.videos) == len(labels_multi.videos)


def test_load_video(centered_pair_low_quality_path):
    assert load_video(centered_pair_low_quality_path).shape == (1100, 384, 384, 1)
    assert load_file(centered_pair_low_quality_path).shape == (1100, 384, 384, 1)


def test_load_video_MP4(centered_pair_low_quality_path, tmp_path):
    """Test loading video with uppercase extension (.MP4)."""
    # Copy the existing fixture to a temp file with uppercase extension
    uppercase_video_path = tmp_path / "centered_pair.MP4"
    shutil.copy(centered_pair_low_quality_path, uppercase_video_path)

    # Test with string path
    video = load_video(str(uppercase_video_path))
    assert video.shape == (1100, 384, 384, 1)

    # Test with Path object
    video = load_video(uppercase_video_path)
    assert video.shape == (1100, 384, 384, 1)


@pytest.mark.parametrize("format", ["slp", "nwb", "labelstudio", "jabs"])
def test_load_save_file(format, tmp_path, slp_typical, jabs_real_data_v5):
    if format == "slp":
        labels = load_slp(slp_typical)
        save_file(labels, tmp_path / "test.slp")
        assert type(load_file(tmp_path / "test.slp")) is Labels
    elif format == "nwb":
        labels = load_slp(slp_typical)
        save_file(labels, tmp_path / "test.nwb")
        assert type(load_file(tmp_path / "test.nwb")) is Labels
    elif format == "labelstudio":
        labels = load_slp(slp_typical)
        save_file(labels, tmp_path / "test.json")
        assert type(load_file(tmp_path / "test.json")) is Labels
    elif format == "jabs":
        labels = load_jabs(jabs_real_data_v5)
        save_file(labels, tmp_path, pose_version=5)
        assert type(load_file(tmp_path / jabs_real_data_v5)) is Labels

        save_file(labels, tmp_path, format="jabs")
        assert type(load_file(tmp_path / jabs_real_data_v5)) is Labels


def test_load_save_file_invalid():
    with pytest.raises(ValueError):
        load_file("invalid_file.ext")

    with pytest.raises(ValueError):
        save_file(Labels(), "invalid_file.ext")


def test_save_video(centered_pair_low_quality_video, tmp_path):
    imgs = centered_pair_low_quality_video[:4]
    save_video(imgs, tmp_path / "output.mp4")
    vid = load_video(tmp_path / "output.mp4")
    assert vid.shape == (4, 384, 384, 1)
    save_video(vid, tmp_path / "output2.mp4")
    vid2 = load_video(tmp_path / "output2.mp4")
    assert vid2.shape == (4, 384, 384, 1)


def test_save_file_ultralytics_autodetect(tmp_path, slp_typical):
    """Test ultralytics format auto-detection in save_file."""
    labels = load_slp(slp_typical)

    # Test with directory path (should auto-detect ultralytics)
    output_dir = tmp_path / "ultralytics_output"
    output_dir.mkdir()
    save_file(labels, str(output_dir))  # No format specified

    # Check that files were created in ultralytics format
    assert (output_dir / "data.yaml").exists()
    # Default split creates train/val directories
    assert (output_dir / "train").exists()
    assert (output_dir / "val").exists()

    # Test with split_ratios kwarg (should auto-detect ultralytics)
    output_dir2 = tmp_path / "ultralytics_output2"
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    save_file(labels, str(output_dir2), split_ratios=splits)

    assert (output_dir2 / "data.yaml").exists()
    # With 3-way split, creates train/val/test
    assert (output_dir2 / "train").exists()
    assert (output_dir2 / "val").exists()
    assert (output_dir2 / "test").exists()


def test_load_labels_set_slp_directory(tmp_path, slp_minimal):
    """Test load_labels_set with SLP directory."""
    labels = load_slp(slp_minimal)

    # Create test directory with SLP files
    test_dir = tmp_path / "splits"
    test_dir.mkdir()
    labels.save(test_dir / "train.slp", embed=False)
    labels.save(test_dir / "val.slp", embed=False)

    # Load without format specification (should auto-detect SLP)
    labels_set = load_labels_set(test_dir)

    assert isinstance(labels_set, LabelsSet)
    assert len(labels_set) == 2
    assert "train" in labels_set
    assert "val" in labels_set


def test_load_labels_set_slp_list(tmp_path, slp_minimal):
    """Test load_labels_set with list of SLP files."""
    labels = load_slp(slp_minimal)

    # Create test files
    file1 = tmp_path / "split1.slp"
    file2 = tmp_path / "split2.slp"
    labels.save(file1, embed=False)
    labels.save(file2, embed=False)

    # Load with list (should auto-detect SLP)
    labels_set = load_labels_set([file1, file2])

    assert len(labels_set) == 2
    assert "split1" in labels_set
    assert "split2" in labels_set


def test_load_labels_set_slp_dict(tmp_path, slp_minimal):
    """Test load_labels_set with dictionary of SLP files."""
    labels = load_slp(slp_minimal)

    # Create test files
    train_file = tmp_path / "train_data.slp"
    val_file = tmp_path / "val_data.slp"
    labels.save(train_file, embed=False)
    labels.save(val_file, embed=False)

    # Load with dict
    labels_set = load_labels_set({"training": train_file, "validation": val_file})

    assert len(labels_set) == 2
    assert "training" in labels_set
    assert "validation" in labels_set


def test_load_labels_set_ultralytics(tmp_path):
    """Test load_labels_set with Ultralytics dataset."""
    import numpy as np
    import yaml

    from sleap_io import LabelsSet

    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    # Create minimal Ultralytics structure
    data_config = {
        "path": str(dataset_path),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": [2, 2],
        "names": ["animal"],
    }

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Create train split
    train_path = dataset_path / "train"
    images_path = train_path / "images"
    labels_path = train_path / "labels"
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    # Create a dummy image and label
    import imageio.v3 as iio

    img_file = images_path / "img_000.jpg"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    iio.imwrite(img_file, dummy_img)

    label_file = labels_path / "img_000.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.4 0.4 0.4 2 0.6 0.6 2\n")

    # Load with auto-detection (should detect ultralytics)
    labels_set = load_labels_set(dataset_path)

    assert isinstance(labels_set, LabelsSet)
    assert "train" in labels_set


def test_load_labels_set_explicit_format(tmp_path, slp_minimal):
    """Test load_labels_set with explicit format specification."""
    labels = load_slp(slp_minimal)

    # Create test file
    test_file = tmp_path / "test.slp"
    labels.save(test_file, embed=False)

    # Load with explicit format
    labels_set = load_labels_set([test_file], format="slp")

    assert len(labels_set) == 1
    assert "test" in labels_set


def test_load_labels_set_invalid_format():
    """Test load_labels_set with invalid format."""
    with pytest.raises(ValueError, match="Unknown format"):
        load_labels_set("dummy_path", format="invalid_format")


def test_load_labels_set_ultralytics_invalid_input():
    """Test load_labels_set with invalid input for ultralytics."""
    with pytest.raises(ValueError, match="requires a directory path"):
        load_labels_set(["file1", "file2"], format="ultralytics")


def test_load_labels_set_kwargs_passing(tmp_path):
    """Test that kwargs are properly passed to format-specific loaders."""
    import numpy as np
    import yaml

    from sleap_io import Node, Skeleton

    dataset_path = tmp_path / "yolo_dataset"
    dataset_path.mkdir()

    # Create val split only
    val_path = dataset_path / "val"
    images_path = val_path / "images"
    labels_path = val_path / "labels"
    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    # Create dummy data
    import imageio.v3 as iio

    img_file = images_path / "img_000.jpg"
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    iio.imwrite(img_file, dummy_img)

    label_file = labels_path / "img_000.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.4 0.5 0.5 2\n")

    # Create minimal data.yaml
    data_config = {
        "path": str(dataset_path),
        "val": "val/images",
        "kpt_shape": [1, 2],
        "names": ["animal"],
    }

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Load with specific splits and skeleton via kwargs
    skeleton = Skeleton([Node("custom_node")])
    labels_set = load_labels_set(
        dataset_path,
        format="ultralytics",
        splits=["val"],
        skeleton=skeleton,
    )

    assert len(labels_set) == 1
    assert "val" in labels_set
    assert labels_set["val"].skeletons[0].nodes[0].name == "custom_node"


def test_load_labels_set_format_detection_edge_cases(tmp_path, slp_minimal):
    """Test edge cases in format detection for load_labels_set."""
    labels = load_slp(slp_minimal)

    # Test single SLP file wrapped in a list
    single_file = tmp_path / "single.slp"
    labels.save(single_file, embed=False)

    # Should auto-detect SLP format from file extension when in list
    labels_set = load_labels_set([str(single_file)])
    assert len(labels_set) == 1
    assert "single" in labels_set

    # Test non-directory path with explicit format
    # This tests the edge case where a single file path is provided but
    # format is explicit
    try:
        # This should fail because SLP format expects directory/list/dict
        load_labels_set(str(single_file), format="slp")
    except ValueError as e:
        assert "Path must be a directory" in str(e)


# ---------------------------------------------------------------------------
# Remote URL loading (PR 1)
# ---------------------------------------------------------------------------


def test_load_slp_url_via_httpserver(httpserver, slp_minimal):
    """`load_slp` loads a `.slp` served over HTTP into a `Labels` object.

    The fixture is served with a plain 200 (no Range support), so the
    ``download`` stream mode (full read into memory) is used.
    """
    file_bytes = Path(slp_minimal).read_bytes()
    httpserver.expect_request("/labels.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.slp")

    labels = load_slp(url, stream_mode="download")
    assert type(labels) is Labels
    assert len(labels) == 1


def test_load_slp_url_preserves_lazy(httpserver, slp_minimal):
    """`load_slp(url, lazy=True)` produces a lazy `Labels` over HTTP."""
    file_bytes = Path(slp_minimal).read_bytes()
    httpserver.expect_request("/labels.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.slp")

    lazy_labels = load_slp(url, lazy=True, stream_mode="download")
    assert type(lazy_labels) is Labels
    assert lazy_labels.is_lazy
    # Lazy labels still expose frames.
    assert len(lazy_labels) == 1


def test_load_slp_url_invalid_stream_mode(httpserver, slp_minimal):
    """An unrecognized `stream_mode` raises `ValueError` (not RemoteIOError)."""
    file_bytes = Path(slp_minimal).read_bytes()
    httpserver.expect_request("/labels.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.slp")

    with pytest.raises(ValueError, match="Invalid stream_mode"):
        load_slp(url, stream_mode="not-a-real-mode")


def _serve_with_range(httpserver, path, data):
    """Serve `data` at `path` honoring HTTP Range requests (206 partial content)."""
    from werkzeug.wrappers import Response

    def handler(request):
        rng = request.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng.split("=", 1)[1].split(",")[0]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(data) - 1
            end = min(end, len(data) - 1)
            chunk = data[start : end + 1]
            resp = Response(chunk, status=206)
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
        else:
            resp = Response(data, status=200)
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    httpserver.expect_request(path).respond_with_handler(handler)


def test_load_slp_url_embedded_pkg_keeps_clean_url_and_reads_frame(
    httpserver, slp_minimal_pkg
):
    """Embedded ``pkg.slp`` over a URL keeps a clean URL and streams frames.

    Regression test: ``make_video`` previously ran the embedded video's URL
    through ``Path``, collapsing ``https://`` to ``https:/`` so the embedded
    ``HDF5Video`` could not be reopened over the network.
    """
    data = Path(slp_minimal_pkg).read_bytes()
    _serve_with_range(httpserver, "/minimal.pkg.slp", data)
    url = httpserver.url_for("/minimal.pkg.slp")

    labels = load_slp(url)
    video = labels.videos[0]

    # The embedded video filename must be the unmangled URL (scheme "//" intact).
    assert str(video.filename) == url
    assert "://" in str(video.filename)
    assert type(video.backend).__name__ == "HDF5Video"

    # The embedded frame must be readable over the URL (range-streamed).
    frame = video[0]
    assert frame.shape == (384, 384, 1)
    assert frame.dtype.name == "uint8"


def _make_label_image_pkg(tmp_path):
    """Build a `.pkg.slp` containing a single `UserLabelImage` and return path.

    Returns the path and the raw label-image pixel data so callers can assert a
    pixel-perfect round-trip.
    """
    import numpy as np

    from sleap_io import LabeledFrame, Skeleton, Track, Video
    from sleap_io.io.main import save_slp
    from sleap_io.model.label_image import LabelImage, UserLabelImage

    data = np.zeros((8, 10), dtype=np.int32)
    data[1:3, 2:5] = 1
    data[5:7, 6:9] = 2

    video = Video(filename="remote_label_image_test.mp4")
    track = Track(name="cell_1")
    li = UserLabelImage(
        data=data,
        objects={1: LabelImage.Info(track=track, category="neuron", name="n1")},
        source="cellpose",
    )
    lf = LabeledFrame(video=video, frame_idx=0)
    lf.label_images.append(li)
    labels = Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[Skeleton(nodes=["A"])],
        tracks=[track],
    )
    path = str(tmp_path / "label_image.pkg.slp")
    save_slp(labels, path)
    return path, data


@pytest.mark.parametrize("lazy", [False, True])
def test_load_slp_url_label_image_roundtrip(httpserver, tmp_path, lazy):
    """A remote `.pkg.slp` with a `LabelImage` loads and reads pixels over HTTP.

    Regression test for finding 1: the long-lived lazy-access handle in
    `read_label_images` was opened unconditionally as
    `h5py.File(labels_path, "r")`, which raised `OSError` for URL loads (where
    `labels_path` is the raw URL). It must now open a fresh fsspec file-like.
    """
    import numpy as np

    path, data = _make_label_image_pkg(tmp_path)
    file_bytes = Path(path).read_bytes()
    _serve_with_range(httpserver, "/label_image.pkg.slp", file_bytes)
    url = httpserver.url_for("/label_image.pkg.slp")

    labels = load_slp(url, lazy=lazy, stream_mode="download")
    assert len(labels.label_images) == 1

    rli = labels.label_images[0]
    assert rli.source == "cellpose"
    assert rli.objects[1].category == "neuron"
    # The pixel data must be readable over HTTP via the lazy loader.
    np.testing.assert_array_equal(rli.data, data)


def test_load_video_url_reads_first_frame(httpserver, centered_pair_low_quality_path):
    """`load_video(url)` reads a first frame matching the local file's frame."""
    file_bytes = Path(centered_pair_low_quality_path).read_bytes()
    httpserver.expect_request("/movie.mp4").respond_with_data(
        file_bytes, content_type="video/mp4"
    )
    url = httpserver.url_for("/movie.mp4")

    local = load_video(centered_pair_low_quality_path)
    remote = load_video(url)
    assert isinstance(remote, Video)

    local_frame = local[0]
    remote_frame = remote[0]
    assert remote_frame.shape == local_frame.shape
    assert remote_frame.dtype == local_frame.dtype
    assert np.array_equal(remote_frame, local_frame)


def test_load_file_url_routes_to_video(httpserver, centered_pair_low_quality_path):
    """`load_file` routes a `.mp4` URL to `load_video`, returning a `Video`."""
    file_bytes = Path(centered_pair_low_quality_path).read_bytes()
    httpserver.expect_request("/movie.mp4").respond_with_data(
        file_bytes, content_type="video/mp4"
    )
    url = httpserver.url_for("/movie.mp4")

    video = load_file(url)
    assert isinstance(video, Video)
    assert video[0].shape == (384, 384, 1)


def test_load_file_url_video_with_query_string(
    httpserver, centered_pair_low_quality_path
):
    """`load_file` routes a query-stringed `.mp4` URL to `load_video`.

    Regression: the video-extension fallback previously matched against the raw
    URL, so a presigned/CDN-tokenized URL (which does not end in ``.mp4``) fell
    through to a `ValueError`, while `load_video` on the same URL succeeded.
    """
    file_bytes = Path(centered_pair_low_quality_path).read_bytes()
    httpserver.expect_request("/movie.mp4").respond_with_data(
        file_bytes, content_type="video/mp4"
    )
    url = httpserver.url_for("/movie.mp4") + "?token=secret&x=1"

    video = load_file(url)
    assert isinstance(video, Video)
    assert video[0].shape == (384, 384, 1)


def test_load_file_url_video_with_fragment(httpserver, centered_pair_low_quality_path):
    """`load_file` routes a fragment-bearing `.mp4` URL to `load_video`."""
    file_bytes = Path(centered_pair_low_quality_path).read_bytes()
    httpserver.expect_request("/movie.mp4").respond_with_data(
        file_bytes, content_type="video/mp4"
    )
    url = httpserver.url_for("/movie.mp4") + "#t=5"

    video = load_file(url)
    assert isinstance(video, Video)


def test_load_file_url_cloud_scheme_video_raises_not_implemented():
    """A cloud-scheme media URL is rejected with a clean `NotImplementedError`.

    The documented contract is http/https-only for remote video; cloud schemes
    must not reach the decoder. No network request is issued.
    """
    url = "s3://AKIA:secretkey@bucket/video.mp4?X-Amz-Security-Token=topsecret"
    with pytest.raises(NotImplementedError) as exc_info:
        load_file(url)
    message = str(exc_info.value)
    assert "http/https" in message
    # The raw credentials/token must not leak in the error message.
    assert "secretkey" not in message
    assert "topsecret" not in message


def test_load_file_url_nwb_raises_not_implemented():
    """A `.nwb` URL raises a clean `NotImplementedError`, not a raw `OSError`.

    No network request is issued: the extension routes directly to the
    (unimplemented) `nwb` format before any I/O.
    """
    url = "https://user:s3cr3t@example.invalid/data.nwb?token=TOPSECRET"
    with pytest.raises(NotImplementedError) as exc_info:
        load_file(url)
    message = str(exc_info.value)
    assert "not yet implemented" in message
    assert "nwb" in message
    # Credentials are redacted out of the surfaced URL.
    assert "s3cr3t" not in message
    assert "TOPSECRET" not in message
    assert "example.invalid" in message


def test_load_file_url_slp_still_loads(httpserver, slp_minimal):
    """A `.slp` URL still loads after the non-slp gate was added (finding 4)."""
    file_bytes = Path(slp_minimal).read_bytes()
    httpserver.expect_request("/labels.slp").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.slp")
    labels = load_file(url, stream_mode="download")
    assert type(labels) is Labels
    assert len(labels) == 1


def test_load_file_url_sniff_routes_slp(httpserver, slp_minimal):
    """`load_file` sniffs an ambiguous `.h5` URL and routes it to `load_slp`."""
    file_bytes = Path(slp_minimal).read_bytes()
    httpserver.expect_request("/labels.h5").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.h5")

    # `.h5` is an ambiguous extension; sniffing the HDF5 magic + group
    # structure should route to the slp loader.
    labels = load_file(url, stream_mode="download")
    assert type(labels) is Labels
    assert len(labels) == 1


def test_load_file_url_sniff_false_ambiguous_raises(httpserver, slp_minimal):
    """`load_file(url, sniff=False)` on an ambiguous extension raises."""
    file_bytes = Path(slp_minimal).read_bytes()
    httpserver.expect_request("/labels.h5").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.h5")

    with pytest.raises(ValueError, match="ambiguous extension"):
        load_file(url, sniff=False)


def test_load_file_url_explicit_format_skips_sniff(httpserver, slp_minimal):
    """An explicit `format` dispatches a URL directly without sniffing."""
    file_bytes = Path(slp_minimal).read_bytes()
    httpserver.expect_request("/labels.h5").respond_with_data(
        file_bytes, content_type="application/octet-stream"
    )
    url = httpserver.url_for("/labels.h5")

    labels = load_file(url, format="slp", stream_mode="download")
    assert type(labels) is Labels
    assert len(labels) == 1


def test_load_file_url_unsupported_format_redacts_credentials():
    """An explicit unsupported `format` on a URL raises with a redacted URL.

    The `ValueError` from the URL dispatcher must not leak userinfo or
    token-like query parameters (S4).
    """
    url = "https://user:s3cr3t@example.com/data.slp?token=TOPSECRET"
    with pytest.raises(ValueError) as exc_info:
        load_file(url, format="not_a_real_format")
    message = str(exc_info.value)
    assert "s3cr3t" not in message
    assert "TOPSECRET" not in message
    assert "example.com" in message


def test_load_file_url_ambiguous_sniff_false_redacts_credentials():
    """`load_file(url, sniff=False)` on an ambiguous extension redacts the URL.

    No network request is issued: the error is raised before any I/O.
    """
    url = "https://user:s3cr3t@example.com/data.h5?token=TOPSECRET"
    with pytest.raises(ValueError) as exc_info:
        load_file(url, sniff=False)
    message = str(exc_info.value)
    assert "s3cr3t" not in message
    assert "TOPSECRET" not in message
    assert "example.com" in message


def test_load_file_url_unknown_extension_redacts_credentials():
    """A URL with an unrecognized extension raises with a redacted URL.

    No network request is issued: the error is raised before any I/O.
    """
    url = "https://user:s3cr3t@example.com/data.unknownext?token=TOPSECRET"
    with pytest.raises(ValueError) as exc_info:
        load_file(url)
    message = str(exc_info.value)
    assert "s3cr3t" not in message
    assert "TOPSECRET" not in message
    assert "example.com" in message


def test_load_file_url_unrecognized_content_redacts_credentials(httpserver):
    """A sniffed-but-unsupported body raises with a redacted URL (S4).

    The body is binary that the magic sniffer classifies as `unknown`, so the
    ambiguous-extension sniff path reaches the unsupported-content branch.
    """
    httpserver.expect_request("/data.h5").respond_with_data(
        b"\xff\xfe\x00\x01nonsense-binary-body",
        content_type="application/octet-stream",
    )
    base = httpserver.url_for("/data.h5")
    # Inject credentials + token into the loopback URL the loader surfaces.
    scheme, rest = base.split("://", 1)
    url = f"{scheme}://user:s3cr3t@{rest}?token=TOPSECRET"
    with pytest.raises(ValueError) as exc_info:
        load_file(url)
    message = str(exc_info.value)
    assert "s3cr3t" not in message
    assert "TOPSECRET" not in message


def test_load_slp_url_cloud_missing_extra():
    """A missing cloud adapter raises `ImportError` with the `[cloud]` hint.

    Exercised via `_require_package` with a genuinely-absent package name so no
    mocking is needed (CI installs the real cloud adapters).
    """
    with pytest.raises(ImportError) as exc_info:
        _require_package("definitely_not_installed_pkg_xyz", scheme="s3")
    message = str(exc_info.value)
    assert "sleap-io[cloud]" in message
    assert "definitely_not_installed_pkg_xyz" in message


def test_imports_clear_remote_cache_at_top_level():
    """`clear_remote_cache` and `RemoteIOError` are importable from the top.

    Also verifies the lazy re-exports resolve to the underlying objects.
    """
    import sleap_io as sio

    assert sio.clear_remote_cache is clear_remote_cache
    assert sio.RemoteIOError is RemoteIOError
    assert issubclass(RemoteIOError, OSError)
    assert callable(clear_remote_cache)
