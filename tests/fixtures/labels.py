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
from sleap_io.model.centroid import PredictedCentroid, UserCentroid
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

    for fi in range(3):
        instances = []
        frame_masks = []
        frame_rois = []
        frame_bboxes = []
        frame_centroids = []

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
                    track=track,
                    instance=inst,
                    score=0.92,
                    score_map=score_map,
                )
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
                    track=track,
                    instance=inst,
                    score=0.88,
                )
            frame_rois.append(roi)

            # --- Bounding Box ---
            if ii == 0:
                bbox = UserBoundingBox(
                    x1=float(x_off - 5),
                    y1=float(y_off - 5),
                    x2=float(x_off + 25),
                    y2=float(y_off + 45),
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
                    track=track,
                    instance=inst,
                    category="fly",
                    name=f"bbox_f{fi}_i{ii}",
                    source="model_v1",
                    score=0.97,
                )
            frame_bboxes.append(bbox)

            # --- Centroid ---
            if ii == 0:
                centroid = UserCentroid(
                    x=float(x_off + 5),
                    y=float(y_off + 20),
                    track=track,
                    instance=inst,
                    category="fly",
                    name=f"centroid_f{fi}_i{ii}",
                    source="manual",
                )
            else:
                centroid = PredictedCentroid(
                    x=float(x_off + 5),
                    y=float(y_off + 20),
                    track=track,
                    instance=inst,
                    category="fly",
                    name=f"centroid_f{fi}_i{ii}",
                    source="model_v1",
                    score=0.95,
                )
            frame_centroids.append(centroid)

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
            source="manual",
        )

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
            source="model_v1",
            score=0.88,
            score_map=pred_score_map,
        )

        lf = LabeledFrame(video=video, frame_idx=fi, instances=instances)
        lf.masks.extend(frame_masks)
        lf.rois.extend(frame_rois)
        lf.bboxes.extend(frame_bboxes)
        lf.centroids.extend(frame_centroids)
        lf.label_images.extend([user_li, pred_li])
        labeled_frames.append(lf)

    return sleap_io.Labels(
        labeled_frames=labeled_frames,
        videos=[video],
        skeletons=[skeleton],
        tracks=tracks,
    )


@pytest.fixture
def labels_all_annotations():
    """Labels with all annotation types (masks, ROIs, bboxes, label_images).

    Includes both User and Predicted variants with scores and score maps.
    All annotations linked to video, tracks, and instances.
    """
    return make_labels_all_annotations()
