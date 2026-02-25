"""Tests for lazy loading infrastructure."""

import h5py
import numpy as np
import pytest

import sleap_io as sio
from sleap_io import Instance, PredictedInstance, Skeleton, Track, Video
from sleap_io.io.slp import InstanceType, write_labels
from sleap_io.io.slp_lazy import LazyDataStore, LazyFrameList
from sleap_io.model.labeled_frame import LabeledFrame  # noqa: F401

# === LazyDataStore Validation Tests ===


class TestLazyDataStoreValidation:
    """Tests for LazyDataStore validation logic."""

    def _make_minimal_store(
        self,
        n_frames=1,
        n_instances=1,
        n_points=2,
        n_pred_points=2,
        frame_inst_start=0,
        frame_inst_end=1,
        inst_point_start=0,
        inst_point_end=2,
        inst_type=InstanceType.USER,
        skeleton=None,
    ):
        """Helper to create a LazyDataStore with controlled data."""
        # Create minimal skeleton
        if skeleton is None:
            skeleton = Skeleton(name="test", nodes=["a", "b"])

        # Create frames_data
        frames_dtype = np.dtype(
            [
                ("frame_id", "<i8"),
                ("video", "<i8"),
                ("frame_idx", "<u8"),
                ("instance_id_start", "<u8"),
                ("instance_id_end", "<u8"),
            ]
        )
        frames_data = np.zeros(n_frames, dtype=frames_dtype)
        if n_frames > 0:
            frames_data[0] = (0, 0, 0, frame_inst_start, frame_inst_end)

        # Create instances_data
        instances_dtype = np.dtype(
            [
                ("instance_id", "<i8"),
                ("instance_type", "<u1"),
                ("frame_id", "<i8"),
                ("skeleton", "<i4"),
                ("track", "<i4"),
                ("from_predicted", "<i8"),
                ("instance_score", "<f8"),
                ("point_id_start", "<u8"),
                ("point_id_end", "<u8"),
                ("tracking_score", "<f8"),
            ]
        )
        instances_data = np.zeros(n_instances, dtype=instances_dtype)
        if n_instances > 0:
            instances_data[0] = (
                0,
                int(inst_type),
                0,
                0,
                -1,
                -1,
                1.0,
                inst_point_start,
                inst_point_end,
                0.0,
            )

        # Create points_data
        points_dtype = np.dtype(
            [("x", "<f8"), ("y", "<f8"), ("visible", "?"), ("complete", "?")]
        )
        points_data = np.zeros(n_points, dtype=points_dtype)
        for i in range(n_points):
            points_data[i] = (float(i), float(i), True, True)

        # Create pred_points_data
        pred_points_dtype = np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("visible", "?"),
                ("complete", "?"),
                ("score", "<f8"),
            ]
        )
        pred_points_data = np.zeros(n_pred_points, dtype=pred_points_dtype)
        for i in range(n_pred_points):
            pred_points_data[i] = (float(i), float(i), True, True, 1.0)

        return LazyDataStore(
            frames_data=frames_data,
            instances_data=instances_data,
            pred_points_data=pred_points_data,
            points_data=points_data,
            videos=[],
            skeletons=[skeleton],
            tracks=[],
            format_id=1.2,
            source_path=None,
        )

    def test_validate_empty_frames_data(self):
        """Validation passes for empty frames_data (early return)."""
        store = self._make_minimal_store(n_frames=0, n_instances=0)
        # Should not raise - empty data is valid
        store.validate()

    def test_validate_frames_with_no_instances(self):
        """Validation passes for frames with no instances."""
        store = self._make_minimal_store(
            n_frames=1, n_instances=0, frame_inst_start=0, frame_inst_end=0
        )
        # Should not raise - no instances means no point validation needed
        store.validate()

    def test_validate_invalid_instance_bounds(self):
        """Validation raises ValueError for out-of-bounds instance references."""
        with pytest.raises(ValueError, match="Frame references instance index"):
            # Frame references instance index 5 but only 1 instance exists
            self._make_minimal_store(
                n_frames=1,
                n_instances=1,
                frame_inst_start=0,
                frame_inst_end=5,  # Out of bounds!
            )

    def test_validate_invalid_user_point_bounds(self):
        """Validation raises ValueError for out-of-bounds user point references."""
        with pytest.raises(ValueError, match="User instance references point index"):
            # User instance references point index 10 but only 2 points exist
            self._make_minimal_store(
                n_frames=1,
                n_instances=1,
                n_points=2,
                inst_point_start=0,
                inst_point_end=10,  # Out of bounds!
                inst_type=InstanceType.USER,
            )

    def test_validate_invalid_pred_point_bounds(self):
        """Validation raises ValueError for out-of-bounds pred_point references."""
        with pytest.raises(
            ValueError, match="Predicted instance references pred_point index"
        ):
            # Predicted instance references pred_point index 10 but only 2 exist
            self._make_minimal_store(
                n_frames=1,
                n_instances=1,
                n_pred_points=2,
                inst_point_start=0,
                inst_point_end=10,  # Out of bounds!
                inst_type=InstanceType.PREDICTED,
            )


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

    def test_materialize_creates_deep_copies(self, slp_real_data):
        """Materialized Labels has independent (deep copied) metadata objects."""
        lazy = sio.load_slp(slp_real_data, lazy=True)
        materialized = lazy.materialize()

        # Videos should be different objects
        assert materialized.videos[0] is not lazy.videos[0]

        # Skeletons should be different objects
        assert materialized.skeletons[0] is not lazy.skeletons[0]

        # Modifying materialized should not affect lazy
        original_filename = lazy.videos[0].filename
        materialized.videos[0].replace_filename("modified.mp4", open=False)
        assert lazy.videos[0].filename == original_filename

        # Frames should reference the new (materialized) video objects
        assert materialized[0].video is materialized.videos[0]
        assert materialized[0].video is not lazy.videos[0]

        # Instances should reference the new (materialized) skeleton objects
        if len(materialized[0].instances) > 0:
            assert materialized[0].instances[0].skeleton is materialized.skeletons[0]
            assert materialized[0].instances[0].skeleton is not lazy.skeletons[0]


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


# === Additional Edge Case Tests ===


class TestLazyNumpyEdgeCases:
    """Tests for edge cases in LazyDataStore.to_numpy()."""

    def test_numpy_tracked_mode_with_predictions(self, centered_pair):
        """Test tracked mode with prediction file (has tracks)."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)

        # Default mode should use tracked organization when tracks exist
        lazy_arr = lazy.numpy()
        eager_arr = eager.numpy()

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_user_instances_false_tracked_mode(self, centered_pair):
        """Test user_instances=False with tracked mode."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)

        # user_instances=False should only include predicted instances
        lazy_arr = lazy.numpy(user_instances=False)
        eager_arr = eager.numpy(user_instances=False)

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_untracked_with_predictions(self, centered_pair):
        """Test untracked mode with prediction file."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)

        lazy_arr = lazy.numpy(untracked=True)
        eager_arr = eager.numpy(untracked=True)

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)

    def test_numpy_all_params_combined_predictions(self, centered_pair):
        """Test all numpy params combined with prediction file."""
        lazy = sio.load_slp(centered_pair, lazy=True)
        eager = sio.load_slp(centered_pair, lazy=False)

        lazy_arr = lazy.numpy(
            untracked=True, return_confidence=True, user_instances=False
        )
        eager_arr = eager.numpy(
            untracked=True, return_confidence=True, user_instances=False
        )

        assert lazy_arr.shape == eager_arr.shape
        np.testing.assert_allclose(lazy_arr, eager_arr, equal_nan=True)


class TestLazyUserFrameIndices:
    """Tests for LazyDataStore.get_user_frame_indices()."""

    def test_get_user_frame_indices_with_user_instances(self, slp_real_data):
        """get_user_frame_indices returns frame indices with user instances."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        store = labels._lazy_store

        indices = store.get_user_frame_indices()
        assert isinstance(indices, list)
        # Should return sorted list
        assert indices == sorted(indices)

    def test_get_user_frame_indices_predictions_only(self, centered_pair):
        """get_user_frame_indices on predictions-only file."""
        labels = sio.load_slp(centered_pair, lazy=True)
        store = labels._lazy_store

        # Check if there are any user instances
        indices = store.get_user_frame_indices()

        # This file may or may not have user instances
        # Just verify the return type and that it doesn't crash
        assert isinstance(indices, list)


class TestLazyEmptyFrameList:
    """Tests for LazyFrameList with empty or minimal data."""

    def test_lazy_frame_list_empty_slice(self, slp_real_data):
        """Empty slice returns empty list."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        # Slice that returns nothing
        result = labels.labeled_frames[100:100]
        assert result == []

    def test_lazy_frame_list_step_slice(self, slp_real_data):
        """Step slicing works correctly."""
        labels = sio.load_slp(slp_real_data, lazy=True)
        eager = sio.load_slp(slp_real_data, lazy=False)

        # Slice with step
        lazy_frames = labels.labeled_frames[0:6:2]
        eager_frames = eager.labeled_frames[0:6:2]

        assert len(lazy_frames) == len(eager_frames)
        for lf, ef in zip(lazy_frames, eager_frames):
            assert lf.frame_idx == ef.frame_idx


# === Legacy Format Tests ===


class TestLazyLegacyFormats:
    """Tests for lazy loading of files with legacy format_id values."""

    def _create_legacy_format_file(
        self, tmp_path, format_id, include_tracking_score=True
    ):
        """Helper to create an SLP file and modify it to simulate legacy format."""
        skeleton = Skeleton(["A", "B"])
        track = Track("track1")

        # Create instances with known coordinates
        inst = Instance([[1.5, 2.5], [3.5, 4.5]], skeleton=skeleton, track=track)
        pred_inst = PredictedInstance(
            [[5.5, 6.5], [7.5, 8.5]], skeleton=skeleton, track=track, score=0.9
        )

        video = Video.from_filename("fake.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst, pred_inst])
        labels = sio.Labels(
            videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
        )

        # Save the file
        test_path = tmp_path / f"test_format_{format_id}.slp"
        write_labels(test_path, labels)

        # Modify the format_id and optionally remove tracking_score
        with h5py.File(test_path, "r+") as f:
            f["metadata"].attrs["format_id"] = format_id

            if not include_tracking_score:
                # Remove tracking_score field for format_id < 1.2
                instances_data = f["instances"][:]
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
                old_instances = np.zeros(len(instances_data), dtype=old_dtype)
                for i, inst_data in enumerate(instances_data):
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
                del f["instances"]
                f.create_dataset("instances", data=old_instances, dtype=old_dtype)

        return test_path

    def test_format_id_less_than_1_2_lazy_load(self, tmp_path):
        """Lazy loading handles format_id < 1.2 correctly (no tracking_score)."""
        test_path = self._create_legacy_format_file(
            tmp_path, format_id=1.1, include_tracking_score=False
        )

        # Load lazily and access frame to trigger materialization
        labels = sio.load_slp(str(test_path), lazy=True)
        assert labels.is_lazy

        # Access a frame to trigger the format_id < 1.2 branch
        frame = labels[0]
        assert len(frame.instances) == 2

        # Verify tracking_score defaults to 0.0
        inst = frame.instances[0]
        pred_inst = frame.instances[1]
        assert inst.tracking_score == 0.0
        assert pred_inst.tracking_score == 0.0

    def test_format_id_less_than_1_1_coordinate_adjustment(self, tmp_path):
        """Lazy loading applies -0.5 coordinate adjustment for format_id < 1.1."""
        skeleton = Skeleton(["A", "B"])
        track = Track("track1")

        # Create instances with known coordinates
        original_user_coords = [[2.0, 3.0], [4.0, 5.0]]
        original_pred_coords = [[6.0, 7.0], [8.0, 9.0]]
        inst = Instance(original_user_coords, skeleton=skeleton, track=track)
        pred_inst = PredictedInstance(
            original_pred_coords, skeleton=skeleton, track=track, score=0.9
        )

        video = Video.from_filename("fake.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst, pred_inst])
        labels = sio.Labels(
            videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
        )

        # Save the file
        test_path = tmp_path / "test_format_1_0.slp"
        write_labels(test_path, labels)

        # Modify to format_id 1.0 and remove tracking_score
        with h5py.File(test_path, "r+") as f:
            f["metadata"].attrs["format_id"] = 1.0

            # Remove tracking_score field
            instances_data = f["instances"][:]
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
            old_instances = np.zeros(len(instances_data), dtype=old_dtype)
            for i, inst_data in enumerate(instances_data):
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
            del f["instances"]
            f.create_dataset("instances", data=old_instances, dtype=old_dtype)

        # Load lazily and access frame to trigger coordinate adjustment
        labels = sio.load_slp(str(test_path), lazy=True)
        frame = labels[0]

        # Verify coordinates were adjusted by -0.5
        user_inst = frame.instances[0]
        pred_inst = frame.instances[1]

        # User instance coordinates should be original - 0.5
        np.testing.assert_allclose(
            user_inst.points["xy"],
            np.array(original_user_coords) - 0.5,
        )
        # Predicted instance coordinates should be original - 0.5
        np.testing.assert_allclose(
            pred_inst.points["xy"],
            np.array(original_pred_coords) - 0.5,
        )

    def test_format_id_less_than_1_1_numpy_coordinate_adjustment(self, tmp_path):
        """numpy() applies -0.5 coordinate adjustment for format_id < 1.1."""
        skeleton = Skeleton(["A", "B"])
        track = Track("track1")

        # Create instances with known coordinates
        original_coords = [[2.0, 3.0], [4.0, 5.0]]
        inst = Instance(original_coords, skeleton=skeleton, track=track)

        video = Video.from_filename("fake.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = sio.Labels(
            videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
        )

        # Save and modify to format_id 1.0
        test_path = tmp_path / "test_format_1_0_numpy.slp"
        write_labels(test_path, labels)

        with h5py.File(test_path, "r+") as f:
            f["metadata"].attrs["format_id"] = 1.0
            instances_data = f["instances"][:]
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
            old_instances = np.zeros(len(instances_data), dtype=old_dtype)
            for i, inst_data in enumerate(instances_data):
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
            del f["instances"]
            f.create_dataset("instances", data=old_instances, dtype=old_dtype)

        # Load lazily and call numpy() to trigger coordinate adjustment in fast path
        labels = sio.load_slp(str(test_path), lazy=True)
        arr = labels.numpy()

        # Verify coordinates were adjusted by -0.5
        expected = np.array(original_coords) - 0.5
        np.testing.assert_allclose(arr[0, 0, :, :2], expected)


# === Video Not in Labels Tests ===


class TestLazyFindVideoNotInLabels:
    """Tests for Labels.find() when video is not in the labels."""

    def test_find_video_not_in_labels_returns_empty(self, slp_real_data):
        """find() returns empty list when video is not in labels."""
        labels = sio.load_slp(slp_real_data, lazy=True)

        # Create a different video that's not in labels
        other_video = Video.from_filename("nonexistent.mp4")

        # Should return empty list
        result = labels.find(other_video, frame_idx=0)
        assert result == []

    def test_find_video_not_in_labels_return_new_single_frame(self, slp_real_data):
        """find(return_new=True) returns new frames for unknown video."""
        labels = sio.load_slp(slp_real_data, lazy=True)

        # Create a different video that's not in labels
        other_video = Video.from_filename("nonexistent.mp4")

        # With return_new=True, should return new frame
        result = labels.find(other_video, frame_idx=42, return_new=True)
        assert len(result) == 1
        assert result[0].video is other_video
        assert result[0].frame_idx == 42

    def test_find_video_not_in_labels_return_new_multiple_frames(self, slp_real_data):
        """find(return_new=True) returns multiple new frames for unknown video."""
        labels = sio.load_slp(slp_real_data, lazy=True)

        # Create a different video that's not in labels
        other_video = Video.from_filename("nonexistent.mp4")

        # With return_new=True and list of frame indices
        result = labels.find(other_video, frame_idx=[10, 20, 30], return_new=True)
        assert len(result) == 3
        assert all(f.video is other_video for f in result)
        assert [f.frame_idx for f in result] == [10, 20, 30]


# === Empty Video Numpy Tests ===


class TestLazyNumpyEmptyVideo:
    """Tests for numpy() edge case with empty video."""

    def test_numpy_empty_video_returns_empty_array(self, tmp_path):
        """numpy(video=X) returns empty array when video has no frames."""
        # Create labels with two videos, second one has no frames
        skeleton = Skeleton(["A", "B"])
        video1 = Video.from_filename("video1.mp4")
        video2 = Video.from_filename("video2.mp4")  # No frames for this one

        inst = Instance([[1, 2], [3, 4]], skeleton=skeleton)
        lf = LabeledFrame(video=video1, frame_idx=0, instances=[inst])

        labels = sio.Labels(
            videos=[video1, video2],
            skeletons=[skeleton],
            labeled_frames=[lf],
        )

        # Save and reload lazily
        test_path = tmp_path / "multi_video.slp"
        write_labels(test_path, labels)
        labels = sio.load_slp(str(test_path), lazy=True)

        # numpy() for video2 should return empty array
        arr = labels.numpy(video=1)
        assert arr.shape[0] == 0  # No frames
        assert arr.shape[2] == 2  # n_nodes
        assert arr.shape[3] == 2  # x, y


# === Track Overlap Tests ===


class TestLazyNumpyTrackOverlap:
    """Tests for numpy() track overlap detection."""

    def test_numpy_user_predicted_same_track_skips_predicted(self, tmp_path):
        """numpy() skips predicted instance when user instance has same track."""
        skeleton = Skeleton(["A", "B"])
        track = Track("track1")

        # Create user and predicted instances with same track
        user_coords = [[1.0, 2.0], [3.0, 4.0]]
        pred_coords = [[10.0, 20.0], [30.0, 40.0]]
        user_inst = Instance(user_coords, skeleton=skeleton, track=track)
        pred_inst = PredictedInstance(pred_coords, skeleton=skeleton, track=track)

        video = Video.from_filename("video.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
        labels = sio.Labels(
            videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
        )

        # Save and reload lazily
        test_path = tmp_path / "track_overlap.slp"
        write_labels(test_path, labels)
        labels = sio.load_slp(str(test_path), lazy=True)

        # In untracked mode, should use user instance, not predicted
        arr = labels.numpy(untracked=True)
        # Only the user instance should be included (user takes precedence)
        np.testing.assert_allclose(arr[0, 0, :, :2], np.array(user_coords))


# === Legacy Format with Confidence Tests ===


class TestLazyLegacyFormatWithConfidence:
    """Tests for legacy format coordinate adjustment with return_confidence=True."""

    def test_format_id_less_than_1_1_numpy_with_confidence(self, tmp_path):
        """numpy(return_confidence=True) with format_id < 1.1 adjusts coordinates."""
        skeleton = Skeleton(["A", "B"])
        track = Track("track1")

        # Create both user and predicted instances
        user_coords = [[2.0, 3.0], [4.0, 5.0]]
        pred_coords = [[6.0, 7.0], [8.0, 9.0]]
        user_inst = Instance(user_coords, skeleton=skeleton, track=track)
        pred_inst = PredictedInstance(
            pred_coords, skeleton=skeleton, track=track, score=0.9
        )

        video = Video.from_filename("video.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[user_inst, pred_inst])
        labels = sio.Labels(
            videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
        )

        # Save and modify to format_id 1.0
        test_path = tmp_path / "legacy_conf.slp"
        write_labels(test_path, labels)

        with h5py.File(test_path, "r+") as f:
            f["metadata"].attrs["format_id"] = 1.0
            instances_data = f["instances"][:]
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
            old_instances = np.zeros(len(instances_data), dtype=old_dtype)
            for i, inst_data in enumerate(instances_data):
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
            del f["instances"]
            f.create_dataset("instances", data=old_instances, dtype=old_dtype)

        # Load lazily and call numpy() with return_confidence
        labels = sio.load_slp(str(test_path), lazy=True)
        arr = labels.numpy(return_confidence=True)

        # Verify coordinates were adjusted by -0.5
        expected_user = np.array(user_coords) - 0.5
        np.testing.assert_allclose(arr[0, 0, :, :2], expected_user)
        # User instances should have confidence of 1.0
        np.testing.assert_allclose(arr[0, 0, :, 2], 1.0)

    def test_format_id_less_than_1_1_numpy_predicted_only(self, tmp_path):
        """numpy(user_instances=False) with format_id < 1.1 adjusts pred coords."""
        skeleton = Skeleton(["A", "B"])
        track = Track("track1")

        # Create only predicted instances
        pred_coords = [[6.0, 7.0], [8.0, 9.0]]
        pred_inst = PredictedInstance(
            pred_coords, skeleton=skeleton, track=track, score=0.85
        )

        video = Video.from_filename("video.mp4")
        lf = LabeledFrame(video=video, frame_idx=0, instances=[pred_inst])
        labels = sio.Labels(
            videos=[video], skeletons=[skeleton], tracks=[track], labeled_frames=[lf]
        )

        # Save and modify to format_id 1.0
        test_path = tmp_path / "legacy_pred.slp"
        write_labels(test_path, labels)

        with h5py.File(test_path, "r+") as f:
            f["metadata"].attrs["format_id"] = 1.0
            instances_data = f["instances"][:]
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
            old_instances = np.zeros(len(instances_data), dtype=old_dtype)
            for i, inst_data in enumerate(instances_data):
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
            del f["instances"]
            f.create_dataset("instances", data=old_instances, dtype=old_dtype)

            # Also set point-level scores in pred_points for proper testing
            pred_points = f["pred_points"][:]
            pred_points["score"] = 0.85  # Set point-level scores
            del f["pred_points"]
            f.create_dataset("pred_points", data=pred_points)

        # Load lazily and call numpy() with predicted only
        labels = sio.load_slp(str(test_path), lazy=True)
        arr = labels.numpy(user_instances=False, return_confidence=True)

        # Verify predicted coordinates were adjusted by -0.5
        expected_pred = np.array(pred_coords) - 0.5
        np.testing.assert_allclose(arr[0, 0, :, :2], expected_pred)
        # Confidence should be from the predicted instance point scores
        np.testing.assert_allclose(arr[0, 0, :, 2], 0.85, rtol=0.01)
