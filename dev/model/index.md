# Data model

sleap-io organizes pose tracking data into a hierarchy of containers and annotations. The core flow is: a **[`Skeleton`](poses.md)** defines the body plan (what landmarks exist), an **[`Instance`](poses.md)** records one animal's pose (where each landmark is), a **[`LabeledFrame`](labels.md)** groups instances at a single video frame, and **[`Labels`](labels.md)** ties everything together into a dataset that can be saved, loaded, and manipulated.

## Overview

The data model is split into five areas, each covered on its own page:

**[Labels](labels.md)**: The dataset container. `Labels` holds labeled frames, videos, skeletons, and tracks. `LabeledFrame` groups annotations for a single video frame. `LabelsSet` manages multiple datasets (e.g., train/val/test splits).

**[Video](video.md)**: Lazy array-like access to video data. `Video` wraps multiple backends (MP4, HDF5, image sequences) behind a unified interface.

**[Poses](poses.md)**: The skeleton template and pose instances. A `Skeleton` declares landmark types (`Node`), connections (`Edge`), and symmetries. `Instance` and `PredictedInstance` store per-animal coordinates and confidence scores. `Track` links the same animal across frames.

**[3D](3d.md)**: Multi-camera support. `Camera` stores calibration parameters, `RecordingSession` links cameras to videos, and `FrameGroup`/`InstanceGroup` pair 2D views for 3D reconstruction.

**Spatial annotations**: Annotation types beyond keypoints — [Centroids](centroids.md) and [Boxes](boxes.md) for detection, [ROIs](rois.md) for vector polygons, and [Segmentation](segmentation.md) (`SegmentationMask`, `LabelImage`) for pixel-level masks.

## Working with annotations in frames

Spatial annotations (centroids, boxes, ROIs, masks, label images) are nested in
[`LabeledFrame`](labels.md) — you add them directly to a frame's annotation
lists. [`LabeledFrame.append`][sleap_io.LabeledFrame.append] dispatches on the
runtime type of the annotation and pushes it onto the correct per-type list — you
never have to touch `lf.instances`, `lf.bboxes`, `lf.centroids`, `lf.masks`,
`lf.label_images`, or `lf.rois` directly.

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> from shapely.geometry import box
>>> video = sio.Video("test.mp4", open_backend=False)
>>> lf = sio.LabeledFrame(video=video, frame_idx=0)
>>> lf.append(sio.UserBoundingBox(x1=10, y1=20, x2=50, y2=60))  # → lf.bboxes
>>> lf.append(sio.UserCentroid(x=100, y=200))                    # → lf.centroids
>>> lf.append(sio.UserSegmentationMask.from_numpy(np.zeros((8, 8), bool)))  # → lf.masks
>>> lf.append(sio.UserLabelImage.from_numpy(np.zeros((8, 8), int)))         # → lf.label_images
>>> lf.append(sio.UserROI(geometry=box(0, 0, 10, 10)))            # → lf.rois
>>> labels = sio.Labels(labeled_frames=[lf])
>>> print(len(labels.centroids), len(labels.bboxes))
>>> print(len(labels.masks), len(labels.label_images), len(labels.rois))

```

The `labels.centroids`, `labels.bboxes`, `labels.masks`, `labels.label_images`,
and `labels.rois` properties return flattened read-only views across all frames.
Static, video-level ROIs (with no frame association) live separately on
[`Labels.static_rois`][sleap_io.Labels.static_rois] — see
[Static vs. temporal ROIs](rois.md#static-vs-temporal-rois).

## Class diagram

``` mermaid
classDiagram
    direction TB

    class Skeleton:::poses {
        +nodes
        +edges
        +symmetries
    }
    class Node:::poses {
        +str name
    }
    class Edge:::poses {
        +Node source
        +Node destination
    }
    class Symmetry:::poses {
        +Set~Node~ nodes
    }
    class Track:::poses {
        +str name
    }
    class Instance:::poses {
        +PointsArray points
        +Skeleton skeleton
        +Track track
    }
    class PredictedInstance:::poses {
        +float score
    }

    class Labels:::labels {
        +labeled_frames
        +videos
        +skeletons
        +tracks
    }
    class LabeledFrame:::labels {
        +Video video
        +int frame_idx
        +instances
        +centroids
        +bboxes
        +masks
        +label_images
        +rois
    }
    class SuggestionFrame:::labels {
        +Video video
        +int frame_idx
    }
    class LabelsSet:::labels {
        +labels
    }

    class Video:::video {
        +str filename
        +VideoBackend backend
    }

    class Camera:::threed {
        +ndarray matrix
        +ndarray dist
        +str name
    }
    class CameraGroup:::threed {
        +cameras
    }
    class RecordingSession:::threed {
        +CameraGroup camera_group
        +frame_groups
    }
    class FrameGroup:::threed {
        +int frame_idx
        +instance_groups
    }
    class InstanceGroup:::threed {
        +instance_by_camera
        +Instance3D instance_3d
        +Identity identity
    }
    class Identity:::threed {
        +str name
        +str color
    }
    class Instance3D:::threed {
        +ndarray points
        +Skeleton skeleton
    }
    class PredictedInstance3D:::threed {
        +ndarray point_scores
    }

    class LabelImageWriter:::labels {
        +str filename
        +add()
    }

    class ROI:::regions {
        <<abstract>>
        +geometry
        +str name
    }
    class SegmentationMask:::regions {
        <<abstract>>
        +rle_counts
        +int height
        +int width
    }
    class BoundingBox:::regions {
        <<abstract>>
        +float x1
        +float y1
        +float x2
        +float y2
    }
    class UserBoundingBox:::regions
    class PredictedBoundingBox:::regions {
        +float score
    }

    class UserROI:::regions
    class PredictedROI:::regions {
        +float score
    }
    class UserSegmentationMask:::regions
    class PredictedSegmentationMask:::regions {
        +float score
        +ndarray score_map
    }
    class UserLabelImage:::regions
    class PredictedLabelImage:::regions {
        +float score
        +ndarray score_map
    }

    class LabelImage:::regions {
        <<abstract>>
        +ndarray data
        +dict objects
        +int n_objects
        +to_masks()
    }

    class Centroid:::regions {
        <<abstract>>
        +float x
        +float y
    }
    class UserCentroid:::regions
    class PredictedCentroid:::regions {
        +float score
    }

    Skeleton "1" *-- "1..*" Node
    Skeleton "1" *-- "0..*" Edge
    Skeleton "1" *-- "0..*" Symmetry
    Instance --> Skeleton : uses
    Instance --> Track
    Instance <|-- PredictedInstance

    Labels "1" *-- "0..*" LabeledFrame
    Labels --> Video
    Labels --> Skeleton
    Labels --> Track
    LabeledFrame "1" *-- "0..*" Instance
    LabeledFrame --> Video
    LabelsSet "1" *-- "1..*" Labels

    CameraGroup "1" *-- "0..*" Camera
    RecordingSession --> CameraGroup
    RecordingSession "1" *-- "0..*" FrameGroup
    FrameGroup "1" *-- "0..*" InstanceGroup
    InstanceGroup --> Instance
    InstanceGroup --> Camera
    InstanceGroup --> Instance3D
    InstanceGroup --> Identity
    Instance3D --> Skeleton : uses
    Instance3D <|-- PredictedInstance3D
    Labels --> Identity

    Centroid <|-- UserCentroid
    Centroid <|-- PredictedCentroid
    BoundingBox <|-- UserBoundingBox
    BoundingBox <|-- PredictedBoundingBox
    ROI <|-- UserROI
    ROI <|-- PredictedROI
    SegmentationMask <|-- UserSegmentationMask
    SegmentationMask <|-- PredictedSegmentationMask
    LabelImage <|-- UserLabelImage
    LabelImage <|-- PredictedLabelImage
    LabelImage --> SegmentationMask : to_masks()
    LabelImage --> BoundingBox : to_bboxes()
    LabelImageWriter --> LabelImage : streams
    LabeledFrame --> Centroid
    LabeledFrame --> BoundingBox
    LabeledFrame --> ROI
    LabeledFrame --> SegmentationMask
    LabeledFrame --> LabelImage

    classDef poses fill:#0097a7,stroke:#00796b,color:#fff
    classDef labels fill:#43a047,stroke:#2e7d32,color:#fff
    classDef video fill:#ef6c00,stroke:#e65100,color:#fff
    classDef threed fill:#7b1fa2,stroke:#6a1b9a,color:#fff
    classDef regions fill:#d32f2f,stroke:#c62828,color:#fff
```

## Quick reference

| Class | Page | Description |
|-------|------|-------------|
| [`Skeleton`](poses.md) | [Poses](poses.md) | Template defining landmark types and their connections |
| [`Node`](poses.md) | [Poses](poses.md) | A single landmark type within a skeleton |
| [`Edge`](poses.md) | [Poses](poses.md) | Directed connection between two nodes |
| [`Symmetry`](poses.md) | [Poses](poses.md) | Left/right pairing between two nodes |
| [`Instance`](poses.md) | [Poses](poses.md) | One animal's pose in a single frame |
| [`PredictedInstance`](poses.md) | [Poses](poses.md) | Model-predicted pose with confidence scores |
| [`Track`](poses.md) | [Poses](poses.md) | Identity linking instances of the same animal across frames |
| [`Labels`](labels.md) | [Labels](labels.md) | Top-level dataset container |
| [`LabeledFrame`](labels.md) | [Labels](labels.md) | All instances at a specific frame of a video |
| [`SuggestionFrame`](labels.md) | [Labels](labels.md) | Frame suggested for labeling |
| [`LabelsSet`](labels.md) | [Labels](labels.md) | Named collection of `Labels` (e.g., train/val/test) |
| [`Video`](video.md) | [Video](video.md) | Video file with lazy backend loading |
| [`Camera`](3d.md) | [3D](3d.md) | Calibrated camera with intrinsic/extrinsic parameters |
| [`CameraGroup`](3d.md) | [3D](3d.md) | Set of cameras used together |
| [`RecordingSession`](3d.md) | [3D](3d.md) | Multi-camera recording linking cameras to videos |
| [`FrameGroup`](3d.md) | [3D](3d.md) | Matched labeled frames across views at one time point |
| [`InstanceGroup`](3d.md) | [3D](3d.md) | Same animal matched across cameras, with optional 3D points |
| [`Identity`](3d.md#identity) | [3D](3d.md) | Cross-session persistent animal identity (distinct from per-video `Track`) |
| [`Instance3D`](3d.md#instance3d) | [3D](3d.md) | Structured triangulated 3D keypoint storage |
| [`PredictedInstance3D`](3d.md#instance3d) | [3D](3d.md) | Model-predicted 3D keypoints with per-point scores |
| [`Centroid`](centroids.md) | [Centroids](centroids.md) | Abstract base centroid point annotation |
| [`UserCentroid`](centroids.md) | [Centroids](centroids.md) | Human-annotated centroid |
| [`PredictedCentroid`](centroids.md) | [Centroids](centroids.md) | Model-predicted centroid with score |
| [`ROI`](rois.md) | [ROIs](rois.md) | Vector geometry annotation (polygon, etc.) |
| [`SegmentationMask`](segmentation.md) | [Segmentation](segmentation.md) | Run-length encoded pixel mask |
| [`BoundingBox`](boxes.md) | [Boxes](boxes.md) | Axis-aligned or rotated bounding box |
| [`UserBoundingBox`](boxes.md) | [Boxes](boxes.md) | Human-annotated bounding box |
| [`PredictedBoundingBox`](boxes.md) | [Boxes](boxes.md) | Model-predicted bounding box with score |
| [`UserROI`](rois.md) | [ROIs](rois.md) | Human-annotated region of interest |
| [`PredictedROI`](rois.md) | [ROIs](rois.md) | Model-predicted region of interest with score |
| [`UserSegmentationMask`](segmentation.md) | [Segmentation](segmentation.md) | Human-annotated segmentation mask |
| [`PredictedSegmentationMask`](segmentation.md) | [Segmentation](segmentation.md) | Model-predicted segmentation mask with score |
| [`UserLabelImage`](segmentation.md) | [Segmentation](segmentation.md) | Human-annotated label image |
| [`PredictedLabelImage`](segmentation.md) | [Segmentation](segmentation.md) | Model-predicted label image with score |
| [`LabelImage`](segmentation.md) | [Segmentation](segmentation.md) | Dense integer label image for instance segmentation |
| [`LabelImageWriter`](segmentation.md#streaming-writes) | [Segmentation](segmentation.md) | Streaming writer for chunked label image SLP files |

!!! tip "Hands-on examples"

    For practical code recipes — loading data, modifying skeletons, exporting formats, and more — see the **[Examples](../examples.md)** guide.
