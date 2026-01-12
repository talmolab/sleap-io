# Data model

`sleap-io` implements the core data structures used in SLEAP for storing data related to multi-instance pose tracking, including for annotation, training and inference.

## Class Relationships

The following diagram shows the relationships between the main classes in the `sleap-io` data model:

``` mermaid
classDiagram
    class Labels {
        +List~LabeledFrame~ labeled_frames
        +List~Video~ videos
        +List~Skeleton~ skeletons
        +List~Track~ tracks
    }
    
    class LabeledFrame {
        +Video video
        +int frame_idx
        +List~Instance~ instances
    }
    
    class Instance {
        +Skeleton skeleton
        +Track track
        +PointsArray points
    }
    
    class PredictedInstance {
        +float score
        +int tracking_score
    }
    
    class Skeleton {
        +List~Node~ nodes
        +List~Edge~ edges
        +List~Symmetry~ symmetries
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
    
    class Video {
        +str filename
        +VideoBackend backend
    }
    
    class LabelsSet {
        +List~Labels~ labels
    }
    
    Labels "1" *-- "0..*" LabeledFrame : contains
    Labels "1" *-- "0..*" Video : contains
    Labels "1" *-- "0..*" Skeleton : contains
    Labels "1" *-- "0..*" Track : contains
    
    LabeledFrame "0..*" --> "1" Video : references
    LabeledFrame "1" *-- "0..*" Instance : contains
    
    Instance "0..*" --> "1" Skeleton : uses
    Instance "0..*" --> "0..1" Track : belongs to
    Instance <|-- PredictedInstance : inherits
    
    Skeleton "1" *-- "1..*" Node : contains
    Skeleton "1" *-- "0..*" Edge : contains
    Skeleton "1" *-- "0..*" Symmetry : contains
    
    Edge "0..*" --> "2" Node : connects
    Symmetry "0..*" --> "2" Node : pairs
    
    LabelsSet "1" *-- "1..*" Labels : contains
```



::: sleap_io.Labels

::: sleap_io.LabeledFrame

::: sleap_io.Instance

::: sleap_io.PredictedInstance

::: sleap_io.Skeleton

::: sleap_io.Node

::: sleap_io.Edge

::: sleap_io.Symmetry

::: sleap_io.Track

::: sleap_io.Video

::: sleap_io.SuggestionFrame

::: sleap_io.Camera

::: sleap_io.CameraGroup

::: sleap_io.FrameGroup

::: sleap_io.InstanceGroup

::: sleap_io.RecordingSession

::: sleap_io.LabelsSet
