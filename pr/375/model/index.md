# Data model

Pose tracking data captures the spatial positions of anatomical landmarks -- such as joints, body parts, or other points of interest -- on animals or objects as they move through video frames. At its simplest, a single pose is a set of 2D coordinates (one per landmark), but real-world datasets quickly become richer: multiple animals appear in the same frame, each with its own identity; skeletons define which landmarks exist and how they connect; and predictions carry confidence scores from machine learning models.

sleap-io organizes all of this into a clean hierarchy. A **`Skeleton`** defines the *body plan*: the landmark types (`Node`s), the connections between them (`Edge`s), and any left/right symmetries. An **`Instance`** captures one animal's pose in a single frame by mapping each node to a 2D coordinate. A **`LabeledFrame`** groups every instance that appears at a particular frame index of a particular video. Finally, **`Labels`** is the top-level dataset container that ties together labeled frames, videos, skeletons, and tracks into one coherent object that can be saved, loaded, and manipulated.

Beyond keypoints, sleap-io supports several complementary annotation types. **`BoundingBox`** stores axis-aligned or oriented rectangles for object detection workflows. **`ROI`** holds arbitrary vector geometry (polygons, points, and other Shapely shapes) for defining arenas, exclusion zones, or spatial regions of interest. **`SegmentationMask`** stores per-pixel raster annotations in a compact run-length encoded format. All three can be associated with specific videos, frames, tracks, and instances.

For multi-camera experiments, the data model extends into three dimensions. A **`Camera`** stores intrinsic and extrinsic calibration parameters, a **`CameraGroup`** collects cameras that were used together, and a **`RecordingSession`** links each camera to its video. Within a session, **`FrameGroup`** and **`InstanceGroup`** pair up 2D views of the same frame and the same animal across cameras, enabling 3D triangulation and multi-view analysis.

## Class diagram

The following diagram shows every class in the data model and the key relationships between them. Classes are color-coded by documentation page.

``` mermaid
classDiagram
    direction TB

    class Skeleton {
        +List~Node~ nodes
        +List~Edge~ edges
        +List~Symmetry~ symmetries
        +str name
    }
    class Node {
        +str name
    }
    class Edge {
        +Node source
        +Node destination
    }
    class Symmetry {
        +Set~Node~ nodes
    }
    class Track {
        +str name
    }
    class Instance {
        +PointsArray points
        +Skeleton skeleton
        +Track track
    }
    class PredictedInstance {
        +float score
        +float tracking_score
    }

    class Labels {
        +List~LabeledFrame~ labeled_frames
        +List~Video~ videos
        +List~Skeleton~ skeletons
        +List~Track~ tracks
        +List~SuggestionFrame~ suggestions
        +List~RecordingSession~ sessions
    }
    class LabeledFrame {
        +Video video
        +int frame_idx
        +List~Instance~ instances
    }
    class SuggestionFrame {
        +Video video
        +int frame_idx
    }
    class LabelsSet {
        +Dict~str, Labels~ labels
    }

    class Video {
        +str filename
        +VideoBackend backend
    }

    class Camera {
        +ndarray matrix
        +ndarray dist
        +ndarray rvec
        +ndarray tvec
        +str name
    }
    class CameraGroup {
        +List~Camera~ cameras
    }
    class RecordingSession {
        +CameraGroup camera_group
        +Dict~int, FrameGroup~ frame_groups
    }
    class FrameGroup {
        +int frame_idx
        +List~InstanceGroup~ instance_groups
    }
    class InstanceGroup {
        +Dict~Camera, Instance~ instance_by_camera
        +ndarray points
    }

    class ROI {
        +BaseGeometry geometry
        +str name
        +Video video
        +int frame_idx
    }
    class SegmentationMask {
        +ndarray rle_counts
        +int height
        +int width
        +Video video
    }
    class BoundingBox {
        +float x_center
        +float y_center
        +float width
        +float height
        +float angle
    }
    class UserBoundingBox
    class PredictedBoundingBox {
        +float score
    }

    %% Poses & Skeletons relationships
    Skeleton "1" *-- "1..*" Node : contains
    Skeleton "1" *-- "0..*" Edge : contains
    Skeleton "1" *-- "0..*" Symmetry : contains
    Edge "1" --> "2" Node : connects
    Symmetry "1" --> "2" Node : pairs
    Instance "0..*" --> "1" Skeleton : uses
    Instance "0..*" --> "0..1" Track : belongs to
    Instance <|-- PredictedInstance : inherits

    %% Labels & Frames relationships
    Labels "1" *-- "0..*" LabeledFrame : contains
    Labels "1" *-- "0..*" Video : contains
    Labels "1" *-- "0..*" Skeleton : contains
    Labels "1" *-- "0..*" Track : contains
    Labels "1" *-- "0..*" SuggestionFrame : contains
    Labels "1" *-- "0..*" RecordingSession : contains
    LabeledFrame "1" *-- "0..*" Instance : contains
    LabeledFrame "0..*" --> "1" Video : references
    SuggestionFrame "0..*" --> "1" Video : references
    LabelsSet "1" *-- "1..*" Labels : contains

    %% 3D & Multi-View relationships
    CameraGroup "1" *-- "0..*" Camera : contains
    RecordingSession "1" --> "1" CameraGroup : uses
    RecordingSession "1" *-- "0..*" FrameGroup : contains
    FrameGroup "1" *-- "0..*" InstanceGroup : contains
    InstanceGroup "1" --> "0..*" Instance : links
    InstanceGroup "1" --> "0..*" Camera : indexed by

    %% Regions & Segmentation relationships
    BoundingBox <|-- UserBoundingBox : inherits
    BoundingBox <|-- PredictedBoundingBox : inherits

    %% Color coding by documentation page
    classDef poses fill:#0097a7,stroke:#00796b,color:#fff
    classDef labels fill:#43a047,stroke:#2e7d32,color:#fff
    classDef video fill:#ef6c00,stroke:#e65100,color:#fff
    classDef threed fill:#7b1fa2,stroke:#6a1b9a,color:#fff
    classDef regions fill:#d32f2f,stroke:#c62828,color:#fff

    class Skeleton,Node,Edge,Symmetry,Track,Instance,PredictedInstance poses
    class Labels,LabeledFrame,SuggestionFrame,LabelsSet labels
    class Video video
    class Camera,CameraGroup,RecordingSession,FrameGroup,InstanceGroup threed
    class ROI,SegmentationMask,BoundingBox,UserBoundingBox,PredictedBoundingBox regions
```

**Legend:** [Poses & Skeletons]{style="color:#0097a7;font-weight:bold"} | [Labels & Frames]{style="color:#43a047;font-weight:bold"} | [Video]{style="color:#ef6c00;font-weight:bold"} | [3D & Multi-View]{style="color:#7b1fa2;font-weight:bold"} | [Regions & Segmentation]{style="color:#d32f2f;font-weight:bold"}

## Quick reference

| Class | Page | Description |
|-------|------|-------------|
| [`Skeleton`](poses.md) | [Poses & Skeletons](poses.md) | Template defining landmark types and their connections |
| [`Node`](poses.md) | [Poses & Skeletons](poses.md) | A single landmark type within a skeleton (e.g., "left eye") |
| [`Edge`](poses.md) | [Poses & Skeletons](poses.md) | Directed connection between two nodes |
| [`Symmetry`](poses.md) | [Poses & Skeletons](poses.md) | Left/right pairing between two nodes |
| [`Track`](poses.md) | [Poses & Skeletons](poses.md) | Identity that links instances of the same animal across frames |
| [`Instance`](poses.md) | [Poses & Skeletons](poses.md) | One animal's pose in a single frame (user-annotated) |
| [`PredictedInstance`](poses.md) | [Poses & Skeletons](poses.md) | Model-predicted pose with a confidence score |
| [`Labels`](labels.md) | [Labels & Frames](labels.md) | Top-level dataset container for all pose data |
| [`LabeledFrame`](labels.md) | [Labels & Frames](labels.md) | All instances at a specific frame of a video |
| [`SuggestionFrame`](labels.md) | [Labels & Frames](labels.md) | Frame suggested for labeling during active learning |
| [`LabelsSet`](labels.md) | [Labels & Frames](labels.md) | Named collection of `Labels` (e.g., train/val/test splits) |
| [`Video`](video.md) | [Video](video.md) | A video file with lazy backend loading |
| [`Camera`](3d.md) | [3D & Multi-View](3d.md) | Calibrated camera with intrinsic and extrinsic parameters |
| [`CameraGroup`](3d.md) | [3D & Multi-View](3d.md) | Set of cameras used together in a multi-view rig |
| [`RecordingSession`](3d.md) | [3D & Multi-View](3d.md) | Multi-camera recording linking cameras to videos |
| [`FrameGroup`](3d.md) | [3D & Multi-View](3d.md) | Matched labeled frames across views at one time point |
| [`InstanceGroup`](3d.md) | [3D & Multi-View](3d.md) | Same animal matched across camera views, with optional 3D points |
| [`ROI`](regions.md) | [Regions & Segmentation](regions.md) | Vector geometry annotation (polygon, point, etc.) |
| [`SegmentationMask`](regions.md) | [Regions & Segmentation](regions.md) | Run-length encoded per-pixel mask annotation |
| [`BoundingBox`](regions.md) | [Regions & Segmentation](regions.md) | Axis-aligned or oriented bounding box |
| [`UserBoundingBox`](regions.md) | [Regions & Segmentation](regions.md) | Human-annotated bounding box |
| [`PredictedBoundingBox`](regions.md) | [Regions & Segmentation](regions.md) | Model-predicted bounding box with confidence score |

## Where to go next

**[Poses & Skeletons](poses.md)** -- Start here if you are working with keypoint data. Covers skeleton construction, node/edge management, instances, tracks, and the relationship between user annotations and model predictions.

**[Labels & Frames](labels.md)** -- Read this to understand how pose data is organized into datasets. Covers the `Labels` container, frame-level operations, searching and filtering, suggestions, and the `LabelsSet` split manager.

**[Video](video.md)** -- Explains how sleap-io handles video I/O, including lazy backend loading, supported formats, and frame access patterns.

**[3D & Multi-View](3d.md)** -- For multi-camera setups. Covers camera calibration, recording sessions, and how 2D poses are grouped across views for 3D reconstruction.

**[Regions & Segmentation](regions.md)** -- Covers non-keypoint annotation types: bounding boxes for detection, ROIs for spatial regions, and segmentation masks for pixel-level labels.

!!! tip "Hands-on examples"

    For practical code recipes that put these classes to work -- loading data, modifying skeletons, exporting to different formats, and more -- see the **[Examples](../examples.md)** guide.
