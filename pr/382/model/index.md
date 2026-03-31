# Data model

sleap-io organizes pose tracking data into a hierarchy of containers and annotations. The core flow is: a **[`Skeleton`](poses.md)** defines the body plan (what landmarks exist), an **[`Instance`](poses.md)** records one animal's pose (where each landmark is), a **[`LabeledFrame`](labels.md)** groups instances at a single video frame, and **[`Labels`](labels.md)** ties everything together into a dataset that can be saved, loaded, and manipulated.

## Overview

The data model is split into five areas, each covered on its own page:

**[Labels](labels.md)**: The dataset container. `Labels` holds labeled frames, videos, skeletons, and tracks. `LabeledFrame` groups annotations for a single video frame. `LabelsSet` manages multiple datasets (e.g., train/val/test splits).

**[Video](video.md)**: Lazy array-like access to video data. `Video` wraps multiple backends (MP4, HDF5, image sequences) behind a unified interface.

**[Poses](poses.md)**: The skeleton template and pose instances. A `Skeleton` declares landmark types (`Node`), connections (`Edge`), and symmetries. `Instance` and `PredictedInstance` store per-animal coordinates and confidence scores. `Track` links the same animal across frames.

**[3D](3d.md)**: Multi-camera support. `Camera` stores calibration parameters, `RecordingSession` links cameras to videos, and `FrameGroup`/`InstanceGroup` pair 2D views for 3D reconstruction.

**[Regions](regions.md)**: Spatial annotations beyond keypoints. `BoundingBox` for detection, `ROI` for vector polygons, `SegmentationMask` for pixel-level binary masks, and `LabelImage` for dense instance segmentation.

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
        +ndarray points
    }

    class ROI:::regions {
        +geometry
        +str name
    }
    class SegmentationMask:::regions {
        +rle_counts
        +int height
        +int width
    }
    class BoundingBox:::regions {
        +float x_center
        +float y_center
        +float width
        +float height
    }
    class UserBoundingBox:::regions
    class PredictedBoundingBox:::regions {
        +float score
    }

    class LabelImage:::regions {
        +ndarray data
        +dict objects
        +int n_objects
        +to_masks()
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

    BoundingBox <|-- UserBoundingBox
    BoundingBox <|-- PredictedBoundingBox
    LabelImage --> SegmentationMask : to_masks()

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
| [`ROI`](regions.md) | [Regions](regions.md) | Vector geometry annotation (polygon, etc.) |
| [`SegmentationMask`](regions.md) | [Regions](regions.md) | Run-length encoded pixel mask |
| [`BoundingBox`](regions.md) | [Regions](regions.md) | Axis-aligned or rotated bounding box |
| [`UserBoundingBox`](regions.md) | [Regions](regions.md) | Human-annotated bounding box |
| [`PredictedBoundingBox`](regions.md) | [Regions](regions.md) | Model-predicted bounding box with score |
| [`LabelImage`](regions.md) | [Regions](regions.md) | Dense integer label image for instance segmentation |

!!! tip "Hands-on examples"

    For practical code recipes — loading data, modifying skeletons, exporting formats, and more — see the **[Examples](../examples.md)** guide.
