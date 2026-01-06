"""Tests for lazy loading infrastructure."""

import numpy as np
import pytest

import sleap_io as sio
from sleap_io.io.slp_lazy import LazyDataStore, LazyFrameList
from sleap_io.model.labeled_frame import LabeledFrame  # noqa: F401

# === LazyDataStore Tests ===


class TestLazyDataStore:
    """Tests for LazyDataStore class."""

    def test_lazy_data_store_init(self, slp_real_data):
        """LazyDataStore is created when loading with lazy=True."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        assert labels._lazy_store is not None
        assert isinstance(labels._lazy_store, LazyDataStore)

    def test_lazy_data_store_len(self, slp_real_data):
        """__len__ returns frame count."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        # Check LazyDataStore length
        assert len(labels._lazy_store) == len(labels)
        # Also check against eager loading
        eager_labels = sio.load_slp(slp_real_data, lazy=False)
        assert len(labels._lazy_store) == len(eager_labels)

    def test_lazy_data_store_copy(self, slp_real_data):
        """copy() creates independent copy with separate arrays."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        store1 = labels._lazy_store
        store2 = store1.copy()

        # Arrays should be copies (not same object)
        assert store1.frames_data is not store2.frames_data
        assert store1.instances_data is not store2.instances_data
        assert store1.points_data is not store2.points_data
        assert store1.pred_points_data is not store2.pred_points_data

        # But arrays should have same contents
        assert np.array_equal(store1.frames_data, store2.frames_data)
        # For structured arrays with NaN, compare byte-level representation
        assert store1.instances_data.tobytes() == store2.instances_data.tobytes()

        # Metadata objects should be shared (same reference)
        assert store1.videos is store2.videos
        assert store1.skeletons is store2.skeletons
        assert store1.tracks is store2.tracks

    def test_lazy_data_store_source_path(self, slp_real_data):
        """Source path is accessible."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        assert labels._lazy_store._source_path == slp_real_data


# === LazyFrameList Tests ===


class TestLazyFrameList:
    """Tests for LazyFrameList class."""

    def test_lazy_frame_list_len(self, slp_real_data):
        """__len__ returns correct count."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        eager_labels = sio.load_slp(slp_real_data, lazy=False)
        assert len(labels.labeled_frames) == len(eager_labels.labeled_frames)

    def test_lazy_frame_list_type(self, slp_real_data):
        """labeled_frames is a LazyFrameList."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        assert isinstance(labels.labeled_frames, LazyFrameList)

    def test_lazy_frame_list_getitem_positive(self, slp_real_data):
        """Positive indexing returns LabeledFrame."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        frame = labels.labeled_frames[0]
        assert isinstance(frame, LabeledFrame)

    def test_lazy_frame_list_getitem_negative(self, slp_real_data):
        """Negative indexing returns correct frame."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        eager_labels = sio.load_slp(slp_real_data, lazy=False)

        lazy_frame = labels.labeled_frames[-1]
        eager_frame = eager_labels.labeled_frames[-1]

        assert lazy_frame.frame_idx == eager_frame.frame_idx
        assert lazy_frame.video.filename == eager_frame.video.filename

    def test_lazy_frame_list_getitem_out_of_range(self, slp_real_data):
        """Out of range index raises IndexError."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        n = len(labels)
        with pytest.raises(IndexError):
            labels.labeled_frames[n + 100]

    def test_lazy_frame_list_slice(self, slp_real_data):
        """Slicing returns list of LabeledFrames."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        frames = labels.labeled_frames[0:3]
        assert isinstance(frames, list)
        assert len(frames) == 3
        for frame in frames:
            assert isinstance(frame, LabeledFrame)

    def test_lazy_frame_list_iter(self, slp_real_data):
        """Iteration yields LabeledFrame objects."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        count = 0
        for frame in labels.labeled_frames:
            assert isinstance(frame, LabeledFrame)
            count += 1
            if count >= 3:  # Only test first few frames
                break

    def test_lazy_frame_list_append_blocked(self, slp_real_data):
        """append() raises RuntimeError with guidance."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        dummy_frame = LabeledFrame(video=labels.videos[0], frame_idx=9999)
        with pytest.raises(RuntimeError, match="Cannot append"):
            labels.labeled_frames.append(dummy_frame)

    def test_lazy_frame_list_extend_blocked(self, slp_real_data):
        """extend() raises RuntimeError with guidance."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        with pytest.raises(RuntimeError, match="Cannot extend"):
            labels.labeled_frames.extend([])

    def test_lazy_frame_list_insert_blocked(self, slp_real_data):
        """insert() raises RuntimeError with guidance."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        dummy_frame = LabeledFrame(video=labels.videos[0], frame_idx=9999)
        with pytest.raises(RuntimeError, match="Cannot insert"):
            labels.labeled_frames.insert(0, dummy_frame)

    def test_lazy_frame_list_setitem_blocked(self, slp_real_data):
        """Item assignment raises RuntimeError with guidance."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        dummy_frame = LabeledFrame(video=labels.videos[0], frame_idx=9999)
        with pytest.raises(RuntimeError, match="Cannot __setitem__"):
            labels.labeled_frames[0] = dummy_frame

    def test_lazy_frame_list_delitem_blocked(self, slp_real_data):
        """Item deletion raises RuntimeError with guidance."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        with pytest.raises(RuntimeError, match="Cannot __delitem__"):
            del labels.labeled_frames[0]

    def test_lazy_frame_list_repr(self, slp_real_data):
        """__repr__ shows useful info."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        repr_str = repr(labels.labeled_frames)
        assert "LazyFrameList" in repr_str
        assert "n_frames" in repr_str


# === Labels Lazy State Tests ===


class TestLabelsLazyState:
    """Tests for Labels lazy state."""

    def test_labels_is_lazy_true(self, slp_real_data):
        """is_lazy is True for lazy-loaded Labels."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        assert labels.is_lazy is True

    def test_labels_is_lazy_false(self, slp_real_data):
        """is_lazy is False for normally loaded Labels."""
        labels = sio.load_slp(slp_real_data, lazy=False)
        assert labels.is_lazy is False

    def test_labels_is_lazy_default(self):
        """is_lazy is False for normally constructed Labels."""
        labels = sio.Labels()
        assert labels.is_lazy is False

    def test_labels_lazy_store_not_in_repr(self, slp_real_data):
        """_lazy_store is not included in repr."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        repr_str = repr(labels)
        assert "_lazy_store" not in repr_str

    def test_labels_metadata_loaded_eagerly(self, slp_real_data):
        """Videos, skeletons, tracks are eagerly loaded."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        eager_labels = sio.load_slp(slp_real_data, lazy=False)

        # Videos should be loaded
        assert len(labels.videos) == len(eager_labels.videos)
        assert labels.videos[0].filename == eager_labels.videos[0].filename

        # Skeletons should be loaded
        assert len(labels.skeletons) == len(eager_labels.skeletons)

        # Tracks should be loaded (may be empty)
        assert len(labels.tracks) == len(eager_labels.tracks)

    def test_labels_len(self, slp_real_data):
        """len() works on lazy Labels."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        eager_labels = sio.load_slp(slp_real_data, lazy=False)
        assert len(labels) == len(eager_labels)


# === load_slp API Tests ===


class TestLoadSlpLazy:
    """Tests for load_slp lazy parameter."""

    def test_load_slp_lazy_false_default(self, slp_real_data):
        """load_slp defaults to eager loading."""
        labels = sio.load_slp(slp_real_data)
        assert labels.is_lazy is False
        assert not isinstance(labels.labeled_frames, LazyFrameList)

    def test_load_slp_lazy_true(self, slp_real_data):
        """load_slp(lazy=True) returns lazy Labels."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        assert labels.is_lazy is True
        assert isinstance(labels.labeled_frames, LazyFrameList)


# === Lazy vs Eager Equivalence Tests ===


class TestLazyEagerEquivalence:
    """Tests verifying lazy and eager produce identical results."""

    def test_frame_count_equivalent(self, slp_real_data):
        """Frame count is identical."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)
        assert len(lazy) == len(eager)

    def test_frame_content_equivalent(self, slp_real_data):
        """Individual frames have identical content."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        for i in range(min(5, len(lazy))):
            lf_lazy = lazy[i]
            lf_eager = eager[i]
            assert lf_lazy.frame_idx == lf_eager.frame_idx
            assert lf_lazy.video.filename == lf_eager.video.filename
            assert len(lf_lazy) == len(lf_eager)

    def test_instance_types_equivalent(self, slp_real_data):
        """Instance types (Instance/PredictedInstance) are correct."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        for i in range(min(5, len(lazy))):
            lf_lazy = lazy[i]
            lf_eager = eager[i]
            for j, (inst_lazy, inst_eager) in enumerate(
                zip(lf_lazy.instances, lf_eager.instances)
            ):
                assert type(inst_lazy) is type(inst_eager), (
                    f"Frame {i}, instance {j}: "
                    f"type mismatch {type(inst_lazy)} vs {type(inst_eager)}"
                )

    def test_points_equivalent(self, slp_real_data):
        """Instance points are correctly constructed."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        for i in range(min(5, len(lazy))):
            lf_lazy = lazy[i]
            lf_eager = eager[i]
            for j, (inst_lazy, inst_eager) in enumerate(
                zip(lf_lazy.instances, lf_eager.instances)
            ):
                # Check point coordinates
                np.testing.assert_allclose(
                    inst_lazy.points["xy"],
                    inst_eager.points["xy"],
                    err_msg=f"Frame {i}, instance {j}: points mismatch",
                )
                # Check node names
                assert list(inst_lazy.points["name"]) == list(inst_eager.points["name"])


# === Predictions Tests ===


class TestLazyPredictions:
    """Tests for lazy loading of files with predictions."""

    def test_load_predictions(self, centered_pair):
        """Lazy loading works with prediction-heavy files."""
        labels = sio.load_slp(centered_pair, lazy=True)
        assert labels.is_lazy
        assert len(labels) > 0

    def test_predictions_frame_access(self, centered_pair):
        """Can access frames from prediction files."""
        labels = sio.load_slp(centered_pair, lazy=True)
        frame = labels[0]
        assert isinstance(frame, LabeledFrame)
        assert len(frame) > 0  # Has instances

    def test_predictions_tracks(self, centered_pair):
        """Tracks are correctly loaded."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)
        assert len(lazy.tracks) == len(eager.tracks)


# === NumPy Output Correctness Tests ===


class TestLazyNumpyEquivalence:
    """Tests verifying lazy numpy() produces identical output to eager."""

    def test_numpy_default_params(self, slp_real_data):
        """Default params produce identical output."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        lazy_arr = lazy.numpy()
        eager_arr = eager.numpy()

        assert lazy_arr.shape == eager_arr.shape, (
            f"Shape mismatch: lazy={lazy_arr.shape}, eager={eager_arr.shape}"
        )
        np.testing.assert_allclose(
            lazy_arr, eager_arr, equal_nan=True, err_msg="numpy() output mismatch"
        )

    def test_numpy_untracked(self, slp_real_data):
        """untracked=True produces identical output."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        lazy_arr = lazy.numpy(untracked=True)
        eager_arr = eager.numpy(untracked=True)

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_return_confidence(self, slp_real_data):
        """return_confidence=True produces identical output."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        lazy_arr = lazy.numpy(return_confidence=True)
        eager_arr = eager.numpy(return_confidence=True)

        assert lazy_arr.shape == eager_arr.shape
        assert lazy_arr.shape[-1] == 3  # x, y, confidence
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_user_instances_false(self, centered_pair):
        """user_instances=False produces identical output."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)

        lazy_arr = lazy.numpy(user_instances=False)
        eager_arr = eager.numpy(user_instances=False)

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_with_video_index(self, slp_real_data):
        """Video index parameter works correctly."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        # Test with video index
        lazy_arr = lazy.numpy(video=0)
        eager_arr = eager.numpy(video=0)

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_with_video_object(self, slp_real_data):
        """Video object parameter works correctly."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        # Test with video object
        lazy_arr = lazy.numpy(video=lazy.videos[0])
        eager_arr = eager.numpy(video=eager.videos[0])

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_predictions_file(self, centered_pair):
        """numpy() works correctly for prediction-heavy files."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)

        lazy_arr = lazy.numpy()
        eager_arr = eager.numpy()

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_predictions_with_confidence(self, centered_pair):
        """numpy(return_confidence=True) works for prediction files."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)

        lazy_arr = lazy.numpy(return_confidence=True)
        eager_arr = eager.numpy(return_confidence=True)

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)


# === Labels.materialize() Tests ===


class TestLabelsMaterialize:
    """Tests for Labels.materialize() method."""

    def test_materialize_returns_labels(self, slp_real_data):
        """materialize() returns a Labels object."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        materialized = lazy.materialize()
        assert isinstance(materialized, sio.Labels)

    def test_materialize_not_lazy(self, slp_real_data):
        """materialize() returns non-lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        materialized = lazy.materialize()
        assert materialized.is_lazy is False

    def test_materialize_eager_returns_self(self, slp_real_data):
        """materialize() on eager Labels returns self."""
        eager = sio.load_slp(slp_real_data, lazy=False)
        result = eager.materialize()
        assert result is eager

    def test_materialize_preserves_frame_count(self, slp_real_data):
        """Materialized Labels has same frame count."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        materialized = lazy.materialize()
        assert len(materialized) == len(lazy)

    def test_materialize_preserves_metadata(self, slp_real_data):
        """Materialized Labels has copied metadata."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        materialized = lazy.materialize()

        assert len(materialized.videos) == len(lazy.videos)
        assert len(materialized.skeletons) == len(lazy.skeletons)
        assert len(materialized.tracks) == len(lazy.tracks)

    def test_materialize_numpy_equivalent(self, slp_real_data):
        """Materialized Labels produces same numpy output."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        materialized = lazy.materialize()

        lazy_arr = lazy.numpy()
        mat_arr = materialized.numpy()

        assert lazy_arr.shape == mat_arr.shape
        np.testing.assert_allclose(lazy_arr, mat_arr, equal_nan=True)

    def test_materialize_allows_mutations(self, slp_real_data):
        """Mutations work on materialized Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        materialized = lazy.materialize()

        # Should not raise
        initial_len = len(materialized)
        new_frame = LabeledFrame(video=materialized.videos[0], frame_idx=99999)
        materialized.append(new_frame)
        assert len(materialized) == initial_len + 1


# === Save Round-Trip Tests ===


class TestLazySaveRoundTrip:
    """Tests for lazy save/load round-trip correctness."""

    def test_lazy_save_creates_valid_file(self, slp_real_data, tmp_path):
        """Lazy save creates a valid SLP file."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        sio.save_slp(lazy, str(out_path))

        assert out_path.exists()
        # Should be loadable
        reloaded = sio.load_slp(str(out_path))
        assert len(reloaded) == len(lazy)

    def test_lazy_save_preserves_frame_data(self, slp_real_data, tmp_path):
        """Lazy save preserves all frame/instance data."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        sio.save_slp(lazy, str(out_path))
        reloaded = sio.load_slp(str(out_path))

        # numpy() output should match
        np.testing.assert_allclose(
            reloaded.numpy(),
            lazy.numpy(),
            equal_nan=True,
            err_msg="Frame data not preserved through round-trip",
        )

    def test_lazy_save_preserves_metadata(self, slp_real_data, tmp_path):
        """Lazy save preserves videos/skeletons/tracks."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        sio.save_slp(lazy, str(out_path))
        reloaded = sio.load_slp(str(out_path))

        assert len(reloaded.videos) == len(lazy.videos)
        assert len(reloaded.skeletons) == len(lazy.skeletons)
        assert len(reloaded.tracks) == len(lazy.tracks)

        # Check skeleton structure
        for skel_orig, skel_reload in zip(lazy.skeletons, reloaded.skeletons):
            assert skel_orig.name == skel_reload.name
            assert len(skel_orig.nodes) == len(skel_reload.nodes)

        # Check track names
        for track_orig, track_reload in zip(lazy.tracks, reloaded.tracks):
            assert track_orig.name == track_reload.name

    def test_roundtrip_lazy_save_eager_load(self, slp_real_data, tmp_path):
        """Lazy save -> eager load preserves data."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        sio.save_slp(lazy, str(out_path))
        eager = sio.load_slp(str(out_path), lazy=False)

        np.testing.assert_allclose(lazy.numpy(), eager.numpy(), equal_nan=True)

    def test_roundtrip_lazy_save_lazy_load(self, slp_real_data, tmp_path):
        """Lazy save -> lazy load preserves data."""
        lazy1 = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        sio.save_slp(lazy1, str(out_path))
        lazy2 = sio.load_slp(str(out_path), lazy=True)

        np.testing.assert_allclose(lazy1.numpy(), lazy2.numpy(), equal_nan=True)

    def test_roundtrip_with_predictions(self, centered_pair, tmp_path):
        """Round-trip works with prediction files."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        out_path = tmp_path / "output.slp"

        sio.save_slp(lazy, str(out_path))
        reloaded = sio.load_slp(str(out_path))

        assert len(reloaded) == len(lazy)
        # Compare numpy output (handles predictions correctly)
        np.testing.assert_allclose(
            reloaded.numpy(user_instances=False),
            lazy.numpy(user_instances=False),
            equal_nan=True,
        )

    def test_lazy_save_with_embed_false(self, slp_real_data, tmp_path):
        """Lazy save with embed=False works."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        # embed=False should use the fast path
        sio.save_slp(lazy, str(out_path), embed=False)

        reloaded = sio.load_slp(str(out_path))
        assert len(reloaded) == len(lazy)

    def test_lazy_save_with_embed_source(self, slp_real_data, tmp_path):
        """Lazy save with embed='source' works."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        # embed="source" should use the fast path
        sio.save_slp(lazy, str(out_path), embed="source")

        reloaded = sio.load_slp(str(out_path))
        assert len(reloaded) == len(lazy)

    def test_lazy_save_with_embedding_materializes(self, slp_real_data, tmp_path):
        """Lazy save with embed=True triggers materialization."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        # embed=True requires materialization internally
        # This should work (not raise) and produce valid output
        sio.save_slp(lazy, str(out_path), embed="user")

        reloaded = sio.load_slp(str(out_path))
        # Frame count should be preserved
        assert len(reloaded) == len(lazy)

    def test_roundtrip_preserves_suggestions(self, slp_real_data, tmp_path):
        """Round-trip preserves suggestions."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        out_path = tmp_path / "output.slp"

        sio.save_slp(lazy, str(out_path))
        reloaded = sio.load_slp(str(out_path))

        assert len(reloaded.suggestions) == len(lazy.suggestions)

    def test_double_roundtrip(self, slp_real_data, tmp_path):
        """Double round-trip preserves data."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        original_numpy = lazy.numpy()

        # First round-trip
        out1 = tmp_path / "output1.slp"
        sio.save_slp(lazy, str(out1))
        lazy2 = sio.load_slp(str(out1), lazy=True)

        # Second round-trip
        out2 = tmp_path / "output2.slp"
        sio.save_slp(lazy2, str(out2))
        lazy3 = sio.load_slp(str(out2), lazy=True)

        np.testing.assert_allclose(
            original_numpy,
            lazy3.numpy(),
            equal_nan=True,
            err_msg="Data not preserved through double round-trip",
        )


# === Mutation Guard Tests ===


class TestLazyMutationBlocking:
    """Tests for mutation blocking on lazy Labels."""

    def test_append_blocked(self, slp_real_data):
        """append() raises RuntimeError on lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        new_frame = LabeledFrame(video=lazy.videos[0], frame_idx=99999)
        with pytest.raises(RuntimeError, match="Cannot append"):
            lazy.append(new_frame)

    def test_extend_blocked(self, slp_real_data):
        """extend() raises RuntimeError on lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        with pytest.raises(RuntimeError, match="Cannot extend"):
            lazy.extend([])

    def test_clean_blocked(self, slp_real_data):
        """clean() raises RuntimeError on lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        with pytest.raises(RuntimeError, match="Cannot clean"):
            lazy.clean()

    def test_remove_predictions_blocked(self, slp_real_data):
        """remove_predictions() raises RuntimeError on lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        with pytest.raises(RuntimeError, match="Cannot remove_predictions"):
            lazy.remove_predictions()

    def test_merge_blocked(self, slp_real_data):
        """merge() raises RuntimeError on lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        other = sio.Labels()
        with pytest.raises(RuntimeError, match="Cannot merge"):
            lazy.merge(other)

    def test_error_message_includes_materialize_guidance(self, slp_real_data):
        """Error messages include materialize() guidance."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        new_frame = LabeledFrame(video=lazy.videos[0], frame_idx=99999)
        with pytest.raises(RuntimeError) as exc_info:
            lazy.append(new_frame)
        assert "materialize()" in str(exc_info.value)


# === Labels.find() Lazy Tests ===


class TestLabelsFindLazy:
    """Tests for Labels.find() with lazy loading."""

    def test_find_existing_frame(self, slp_real_data):
        """find() returns existing frame from lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        video = lazy.videos[0]
        # Get a known frame index
        first_frame = lazy[0]
        frame_idx = first_frame.frame_idx

        result = lazy.find(video, frame_idx)
        assert len(result) == 1
        assert result[0].frame_idx == frame_idx

    def test_find_nonexistent_returns_empty(self, slp_real_data):
        """find() returns empty list for nonexistent frame."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        video = lazy.videos[0]
        result = lazy.find(video, 999999)
        assert result == []

    def test_find_return_new_true(self, slp_real_data):
        """find(return_new=True) returns new frame for nonexistent."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        video = lazy.videos[0]
        result = lazy.find(video, 999999, return_new=True)
        assert len(result) == 1
        assert result[0].frame_idx == 999999

    def test_find_lazy_matches_eager(self, slp_real_data):
        """Lazy find() produces same results as eager."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        # Use respective video objects for each Labels
        lazy_video = lazy.videos[0]
        eager_video = eager.videos[0]
        first_frame = lazy[0]
        frame_idx = first_frame.frame_idx

        lazy_result = lazy.find(lazy_video, frame_idx)
        eager_result = eager.find(eager_video, frame_idx)

        assert len(lazy_result) == len(eager_result)
        assert lazy_result[0].frame_idx == eager_result[0].frame_idx

    def test_find_all_frames_for_video(self, slp_real_data):
        """find(video) returns all frames for that video."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        # Use respective video objects for each Labels
        lazy_video = lazy.videos[0]
        eager_video = eager.videos[0]
        lazy_result = lazy.find(lazy_video)
        eager_result = eager.find(eager_video)

        assert len(lazy_result) == len(eager_result)


# === Labels.user_labeled_frames Lazy Tests ===


class TestLabelsUserLabeledFramesLazy:
    """Tests for Labels.user_labeled_frames with lazy loading."""

    def test_user_labeled_frames_returns_list(self, slp_real_data):
        """user_labeled_frames returns list for lazy Labels."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        result = lazy.user_labeled_frames
        assert isinstance(result, list)

    def test_user_labeled_frames_lazy_matches_eager(self, slp_real_data):
        """Lazy user_labeled_frames matches eager."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        lazy_result = lazy.user_labeled_frames
        eager_result = eager.user_labeled_frames

        assert len(lazy_result) == len(eager_result)
        for lf_lazy, lf_eager in zip(lazy_result, eager_result):
            assert lf_lazy.frame_idx == lf_eager.frame_idx


# === Labels.copy() Lazy Tests ===


class TestLabelsCopyLazy:
    """Tests for Labels.copy() with lazy loading."""

    def test_copy_returns_labels(self, slp_real_data):
        """copy() returns Labels object."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        copy = lazy.copy()
        assert isinstance(copy, sio.Labels)

    def test_copy_is_lazy(self, slp_real_data):
        """Copy of lazy Labels is also lazy."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        copy = lazy.copy()
        assert copy.is_lazy

    def test_copy_is_independent(self, slp_real_data):
        """Copy has independent arrays."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        copy = lazy.copy()

        # Arrays should be different objects
        assert lazy._lazy_store.frames_data is not copy._lazy_store.frames_data

    def test_copy_lazy_matches_eager(self, slp_real_data):
        """Lazy copy produces same data as eager copy."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        copy = lazy.copy()

        np.testing.assert_allclose(lazy.numpy(), copy.numpy(), equal_nan=True)


# === Labels.__repr__() Lazy Tests ===


class TestLabelsReprLazy:
    """Tests for Labels.__repr__() with lazy loading."""

    def test_lazy_repr_shows_lazy_true(self, slp_real_data):
        """Lazy Labels repr shows lazy=True."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        repr_str = repr(lazy)
        assert "lazy=True" in repr_str

    def test_eager_repr_no_lazy(self, slp_real_data):
        """Eager Labels repr doesn't show lazy."""
        eager = sio.load_slp(slp_real_data, lazy=False)
        repr_str = repr(eager)
        assert "lazy=True" not in repr_str


# === End-to-End Integration Tests ===


class TestLazyLoadingIntegration:
    """End-to-end integration tests."""

    def test_load_numpy_workflow(self, slp_real_data):
        """Load lazy -> numpy() workflow."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        arr = labels.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 4

    def test_load_save_workflow(self, slp_real_data, tmp_path):
        """Load lazy -> save workflow."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        out = tmp_path / "out.slp"
        sio.save_slp(labels, str(out))
        assert out.exists()

    def test_load_modify_save_workflow(self, slp_real_data, tmp_path):
        """Load lazy -> materialize -> modify -> save workflow."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        labels = labels.materialize()
        # Make a modification
        labels.provenance["modified"] = True
        out = tmp_path / "out.slp"
        sio.save_slp(labels, str(out))

        reloaded = sio.load_slp(str(out))
        assert reloaded.provenance.get("modified") is True

    def test_lazy_iteration(self, slp_real_data):
        """Lazy iteration yields correct frames."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        for lf_lazy, lf_eager in zip(lazy, eager):
            assert lf_lazy.frame_idx == lf_eager.frame_idx
            assert len(lf_lazy) == len(lf_eager)

    def test_lazy_indexing(self, slp_real_data):
        """Lazy indexing returns correct frames."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        for i in [0, len(lazy) // 2, -1]:
            lf_lazy = lazy[i]
            lf_eager = eager[i]
            assert lf_lazy.frame_idx == lf_eager.frame_idx

    def test_labeled_frames_access_pattern(self, slp_real_data):
        """Common labeled_frames access pattern works."""
        labels = sio.load_slp(slp_real_data, lazy=True)

        # Common pattern in existing code
        for lf in labels.labeled_frames:
            _ = lf.frame_idx
            break  # Just test first frame
