"""Fixtures that return `Labels` objects."""

import numpy as np
import pytest
from shapely.geometry import box as shapely_box

import sleap_io
from sleap_io import (
    Instance,
    LabeledFrame,
    Skeleton,
    Track,
    Video,
)
from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.label_image import (
    LabelImage,
    PredictedLabelImage,
    UserLabelImage,
)
from sleap_io.model.mask import PredictedSegmentationMask, UserSegmentationMask
from sleap_io.model.roi import PredictedROI, UserROI


@pytest.fixture
def labels_predictions(centered_pair):
    """Labels object with predicted instances, multiple tracks and a single video."""
    return sleap_io.load_slp(centered_pair)


def make_labels_all_annotations() -> sleap_io.Labels:
    """Build Labels with all annotation types in both User and Predicted variants.

    Creates 3 frames with 2 instances each. Each instance has:
    - A segmentation mask (User for instance 0, Predicted with score_map for instance 1)
    - An ROI (User for instance 0, Predicted with score for instance 1)
    - A bounding box (User for instance 0, Predicted with score for instance 1)

    Each frame has:
    - A UserLabelImage and a PredictedLabelImage (with score_map and per-object scores)

    All annotations are linked to video, frame_idx, track, and instance.
    """
    skeleton = Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    video = Video(filename="test_video.mp4", open_backend=False)
    tracks = [Track(name="fly_0"), Track(name="fly_1")]

    labeled_frames = []
    all_masks = []
    all_rois = []
    all_bboxes = []
    all_label_images = []

    for fi in range(3):
        instances = []
        frame_masks = []

        for ii in range(2):
            track = tracks[ii]

            # Create instance with offset points per frame/instance
            x_off = 50 + ii * 120
            y_off = 40 + fi * 30
            inst = Instance(
                points={
                    "head": [x_off, y_off],
                    "thorax": [x_off + 10, y_off + 20],
                    "abdomen": [x_off + 5, y_off + 40],
                },
                skeleton=skeleton,
                track=track,
            )
            instances.append(inst)

            # --- Segmentation Mask ---
            mask_data = np.zeros((80, 200), dtype=bool)
            mask_data[y_off : y_off + 15, x_off : x_off + 20] = True

            if ii == 0:
                mask = UserSegmentationMask.from_numpy(
                    mask_data,
                    name=f"mask_f{fi}_i{ii}",
                    category="fly",
                    source="manual",
                    video=video,
                    frame_idx=fi,
                    track=track,
                    instance=inst,
                )
            else:
                score_map = np.where(mask_data, 0.95, 0.05).astype(np.float32)
                mask = PredictedSegmentationMask.from_numpy(
                    mask_data,
                    name=f"mask_f{fi}_i{ii}",
                    category="fly",
                    source="model_v1",
                    video=video,
                    frame_idx=fi,
                    track=track,
                    instance=inst,
                    score=0.92,
                    score_map=score_map,
                )
            all_masks.append(mask)
            frame_masks.append(mask)

            # --- ROI ---
            roi_geom = shapely_box(x_off - 5, y_off - 5, x_off + 25, y_off + 45)
            if ii == 0:
                roi = UserROI(
                    geometry=roi_geom,
                    name=f"roi_f{fi}_i{ii}",
                    category="fly",
                    source="manual",
                    video=video,
                    frame_idx=fi,
                    track=track,
                    instance=inst,
                )
            else:
                roi = PredictedROI(
                    geometry=roi_geom,
                    name=f"roi_f{fi}_i{ii}",
                    category="fly",
                    source="model_v1",
                    video=video,
                    frame_idx=fi,
                    track=track,
                    instance=inst,
                    score=0.88,
                )
            all_rois.append(roi)

            # --- Bounding Box ---
            if ii == 0:
                bbox = UserBoundingBox(
                    x1=float(x_off - 5),
                    y1=float(y_off - 5),
                    x2=float(x_off + 25),
                    y2=float(y_off + 45),
                    video=video,
                    frame_idx=fi,
                    track=track,
                    instance=inst,
                    category="fly",
                    name=f"bbox_f{fi}_i{ii}",
                    source="manual",
                )
            else:
                bbox = PredictedBoundingBox(
                    x1=float(x_off - 5),
                    y1=float(y_off - 5),
                    x2=float(x_off + 25),
                    y2=float(y_off + 45),
                    video=video,
                    frame_idx=fi,
                    track=track,
                    instance=inst,
                    category="fly",
                    name=f"bbox_f{fi}_i{ii}",
                    source="model_v1",
                    score=0.97,
                )
            all_bboxes.append(bbox)

        # --- Label Images ---
        # Compose label image from the two masks
        li_data = np.zeros((80, 200), dtype=np.int32)
        for label_id, m in enumerate(frame_masks, start=1):
            li_data[m.data] = label_id

        # User label image
        user_li = UserLabelImage(
            data=li_data.copy(),
            objects={
                1: LabelImage.Info(
                    track=tracks[0],
                    category="fly",
                    name="fly_0",
                    instance=instances[0],
                ),
                2: LabelImage.Info(
                    track=tracks[1],
                    category="fly",
                    name="fly_1",
                    instance=instances[1],
                ),
            },
            video=video,
            frame_idx=fi,
            source="manual",
        )
        all_label_images.append(user_li)

        # Predicted label image with score_map and per-object scores
        pred_score_map = np.where(li_data > 0, 0.9, 0.05).astype(np.float32)
        pred_li = PredictedLabelImage(
            data=li_data.copy(),
            objects={
                1: LabelImage.Info(
                    track=tracks[0],
                    category="fly",
                    name="fly_0",
                    instance=instances[0],
                    score=0.85,
                ),
                2: LabelImage.Info(
                    track=tracks[1],
                    category="fly",
                    name="fly_1",
                    instance=instances[1],
                    score=0.90,
                ),
            },
            video=video,
            frame_idx=fi,
            source="model_v1",
            score=0.88,
            score_map=pred_score_map,
        )
        all_label_images.append(pred_li)

        labeled_frames.append(
            LabeledFrame(video=video, frame_idx=fi, instances=instances)
        )

    return sleap_io.Labels(
        labeled_frames=labeled_frames,
        videos=[video],
        skeletons=[skeleton],
        tracks=tracks,
        masks=all_masks,
        rois=all_rois,
        bboxes=all_bboxes,
        label_images=all_label_images,
    )


@pytest.fixture
def labels_all_annotations():
    """Labels with all annotation types (masks, ROIs, bboxes, label_images).

    Includes both User and Predicted variants with scores and score maps.
    All annotations linked to video, tracks, and instances.
    """
    return make_labels_all_annotations()
