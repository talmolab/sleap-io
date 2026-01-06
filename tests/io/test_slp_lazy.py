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
