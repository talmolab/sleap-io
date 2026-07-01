# Poses

This page covers how sleap-io represents pose data: from defining a body plan
with [`Skeleton`][sleap_io.Skeleton] to storing actual landmark positions with [`Instance`][sleap_io.Instance]. Together,
these two classes form the core of the pose tracking data model: the skeleton
says *what* to label, and instances record *where* each landmark is.

---

## Overview

The pose data model is built around four key types:

- **[`Skeleton`][sleap_io.Skeleton]** is the **template**: defines what landmarks exist, how they connect, and which are symmetric.
- **[`Instance`][sleap_io.Instance]** is the **data**: stores actual (x, y) coordinates for one animal in one frame.
- **[`PredictedInstance`][sleap_io.PredictedInstance]** is like `Instance` but includes per-point and instance-level confidence scores from a model.
- **[`Track`][sleap_io.Track]** is the **video-local identity**: links the same animal across frames of one recording.
- **[`Identity`](3d.md#identity)** is the **global identity**: a persistent, cross-session animal label assigned via `Instance.identity` (with `identity_score`), distinct from the ephemeral `Track`.

A `Skeleton` is shared across all instances in a dataset. Each `Instance` references a `Skeleton` to know which landmarks it contains, optionally a `Track` (and a global `Identity`) to indicate which animal it belongs to, optionally re-ID [`embeddings`](embedding.md) describing its appearance, and optionally [`categories`](categories.md) tagging it with discrete attributes (e.g. `sex`, `strain`).

---

## Skeleton

A `Skeleton` is a **template** that defines what landmarks (body parts) exist
and how they connect. Think of it as a form to fill in: the skeleton says
"head, thorax, abdomen" while instances fill in the actual (x, y) coordinates.

Skeletons are composed of three building blocks:

| Component                            | Purpose                                         |
| ------------------------------------ | ----------------------------------------------- |
| [`Node`][sleap_io.Node]              | A single landmark type (e.g. "head")            |
| [`Edge`][sleap_io.Edge]              | A directed connection between two nodes          |
| [`Symmetry`][sleap_io.Symmetry]      | A left/right pairing (e.g. "left eye" / "right eye") |

### Creating a skeleton

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(
...     nodes=["head", "thorax", "abdomen"],
...     edges=[("head", "thorax"), ("thorax", "abdomen")],
... )
>>> print(skeleton)
>>> print(len(skeleton))
>>> print(skeleton.node_names)
>>> print(skeleton.edge_inds)

```

Nodes and edges can be specified as strings or indices: they are converted to
[`Node`][sleap_io.Node] and [`Edge`][sleap_io.Edge] objects automatically.

### Accessing nodes

Nodes can be retrieved by **name** or **integer index**, and you can look up a
node's index in the skeleton:

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(
...     nodes=["head", "thorax", "abdomen"],
...     edges=[("head", "thorax"), ("thorax", "abdomen")],
... )
>>> print(skeleton["head"])
>>> print(skeleton[0])
>>> print(skeleton.index("thorax"))
>>> print("head" in skeleton)

```

### Symmetries

Symmetries record which nodes are left/right mirrors of each other. This is used
during data augmentation (horizontal flipping) to swap the correct landmark
indices.

```pycon
>>> import sleap_io as sio
>>> skel = sio.Skeleton(["A", "B_left", "B_right"])
>>> skel.add_symmetry("B_left", "B_right")
>>> print(skel.symmetry_names)

```

When a skeleton is imported without symmetry metadata (common for formats that
don't store it) but its node names encode laterality, you can infer the pairs
from names instead of adding them one by one. `infer_symmetries_by_name` is
**non-mutating** -- it returns suggested `(left_index, right_index)` pairs so you
can review them before applying, since a wrong guess would silently corrupt flip
augmentation:

```pycon
>>> import sleap_io as sio
>>> skel = sio.Skeleton(["nose", "eye_L", "eye_R", "ear_L", "ear_R"])
>>> skel.infer_symmetries_by_name()
[(1, 2), (3, 4)]
>>> skel.add_symmetries(skel.infer_symmetries_by_name())  # apply if they look right
>>> print(skel.symmetry_names)

```

Names are matched by splitting on separators (`_`, `-`, `.`, space), camelCase
boundaries, and letter/digit boundaries, so `Ear_L`/`Ear_R`, `left_eye`/`right_eye`,
`LeftPaw`/`RightPaw`, and `L1`/`R1` all pair up. Truly non-semantic pairings such
as `L1`/`L2` cannot be inferred and must be declared with `add_symmetry`.

### Node, Edge, and Symmetry

These are lightweight value types that you rarely need to construct directly --
the `Skeleton` constructor and convenience methods handle them for you.

| Class      | Fields                                    |
| ---------- | ----------------------------------------- |
| `Node`     | `name: str`                               |
| `Edge`     | `source: Node`, `destination: Node`       |
| `Symmetry` | `nodes: set[Node]` (exactly 2 nodes)      |

!!! tip "Building skeletons incrementally"
    You can also build up a skeleton step by step:
    ```python
    skel = sio.Skeleton()
    skel.add_nodes(["head", "thorax", "abdomen"])
    skel.add_edge("head", "thorax")
    skel.add_edge("thorax", "abdomen")
    ```

---

## Instance

An `Instance` is one animal's pose in one frame: the "filled-in form." It
stores (x, y) coordinates for each landmark defined by a `Skeleton`.

### From a numpy array

The most common way to create an instance is from a `(n_nodes, 2)` array of
coordinates:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10.2, 20.4], [5.8, 15.1], [0.3, 10.6]]),
...     skeleton=skeleton,
... )
>>> print(inst)
>>> print(inst.numpy())

```

You can access individual landmarks by node name and inspect their fields:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10.2, 20.4], [5.8, 15.1], [0.3, 10.6]]),
...     skeleton=skeleton,
... )
>>> print(inst["head"]["xy"])
>>> print(inst["head"]["visible"])
>>> print(inst.n_visible)
>>> print(inst.is_empty)

```

### From a dictionary

If you prefer to specify coordinates by node name:

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance(
...     {"head": [10, 20], "thorax": [5, 15], "abdomen": [0, 10]},
...     skeleton=skeleton,
... )
>>> print(inst.numpy())

```

### Empty instances

Create an instance with no visible points (all coordinates are unset):

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> empty_inst = sio.Instance.empty(skeleton=skeleton)
>>> print(empty_inst.is_empty)

```

!!! note "See also"
    Instances are organized into frames and datasets through
    [`LabeledFrame`](labels.md#labeled-frames) and [`Labels`](labels.md#labels).
    See the [Labels & Frames](labels.md) page for the full picture.

### Converting to other modalities

A pose `Instance` can be projected onto any of the other spatial detection
modalities without losing its metadata, through the unified
[conversion matrix](segmentation.md#converting-between-annotation-types). Every
verb returns the `User*`/`Predicted*` variant matching the instance (a
`PredictedInstance` carries its `score`) and propagates `track`,
`tracking_score`, and an `instance=self` backref.

[`Instance.centroid_xy`][sleap_io.Instance.centroid_xy] returns the raw `(x, y)`
of the visible landmarks, while [`Instance.to_centroid()`][sleap_io.Instance.to_centroid]
produces a full [`Centroid`](centroids.md) object:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(
...     ["head", "thorax", "abdomen"],
...     edges=[("head", "thorax"), ("thorax", "abdomen")],
... )
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40], [50, 60]]),
...     skeleton=skeleton,
... )
>>> print(inst.centroid_xy)        # (mean_x, mean_y) of the visible points
>>> c = inst.to_centroid()
>>> print(type(c).__name__, c.xy)

```

[`Instance.to_bbox()`][sleap_io.Instance.to_bbox],
[`Instance.to_roi()`][sleap_io.Instance.to_roi], and
[`Instance.to_mask()`][sleap_io.Instance.to_mask] fit a
[`BoundingBox`](boxes.md), [`ROI`](rois.md), or `SegmentationMask` to the pose:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(
...     ["head", "thorax", "abdomen"],
...     edges=[("head", "thorax"), ("thorax", "abdomen")],
... )
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40], [50, 60]]),
...     skeleton=skeleton,
... )
>>> print(inst.to_bbox().xyxy)                      # tight box of visible points
>>> print(inst.to_bbox(mode="centered", size=30).xyxy)
>>> roi = inst.to_roi(method="shapes", node_radius=5, edge_radius=2)
>>> print(roi.is_empty)
>>> mask = inst.to_mask(80, 80, method="shapes", node_radius=5, edge_radius=2)
>>> print(mask.area > 0)

```

`to_bbox` supports `mode="tight"` (axis-aligned or `rotated=True`) and
`mode="centered"` (a fixed `size` box around a computed centroid). `to_roi`
"burns in" the pose with `method="shapes"` (union of discs around nodes and
capsules around edges; at least one of `node_radius`/`edge_radius` must be
`> 0`) or `method="convex_hull"`. `to_mask(height, width, **roi_kwargs)` is
exactly `to_roi(**roi_kwargs).to_mask(height, width)`. All verbs accept
`error_on_empty=False` and return an empty target for an instance with no
visible points.

Centroids can be turned back into single-node instances with
[`Centroid.to_pose`][sleap_io.Centroid.to_pose] (formerly `to_instance`, now a
deprecated alias), so the two representations are fully interchangeable. See
[Regions → Centroids](centroids.md) for the full data model.

---

## Predicted instances

`PredictedInstance` extends `Instance` with confidence scores: both a
per-point score for each landmark and an overall instance-level score.

When creating from a numpy array, the third column is interpreted as the
per-point confidence score:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> pred = sio.PredictedInstance.from_numpy(
...     np.array([[10.2, 20.4, 0.9], [5.8, 15.1, 0.8], [0.3, 10.6, 0.7]]),
...     skeleton=skeleton,
...     score=0.85,
... )
>>> print(pred.score)
>>> print(pred.numpy(scores=True))

```

!!! tip
    Call `pred.numpy()` (without `scores=True`) to get the same `(n_nodes, 2)`
    array as a regular `Instance`. Use `pred.numpy(scores=True)` when you need
    the `(n_nodes, 3)` array with confidence scores appended.

---

## Track

A `Track` represents the identity of a single animal or object across multiple
frames. Assigning the same `Track` to instances in different frames links them
as belonging to the same individual.

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> track = sio.Track("animal_1")
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
...     track=track,
... )
>>> print(inst.track.name)

```

!!! note
    `Track` objects are compared by **identity** (not by name). Two different
    `Track("mouse")` objects are considered distinct: this allows multiple
    tracks with the same display name if needed.

!!! tip "Track vs. Identity"
    `Track` is a **per-video** temporal trajectory — it links instances in consecutive frames of the same recording and disappears when the tracker loses its target. [`Identity`](3d.md#identity) is a **cross-session** persistent label for the same animal across recordings, sessions, and multi-view setups. In multi-view workflows, multiple per-camera `Track`s typically map to a single `Identity` through [`InstanceGroup.identity`](3d.md#instance-group).

---

## Points array

Under the hood, an instance stores its landmark data in a `PointsArray`, a
structured numpy array with named fields for coordinates, visibility, and
metadata.

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> print(inst.points.dtype.names)
>>> print(inst.points["xy"])
>>> print(inst.points["visible"])

```

`PredictedInstance` uses `PredictedPointsArray`, which adds a `score` field:

| Field      | Type         | Description                        |
| ---------- | ------------ | ---------------------------------- |
| `xy`       | `float64[2]` | (x, y) coordinates                 |
| `score`    | `float64`    | Per-point confidence (predicted only) |
| `visible`  | `bool`       | Whether the point is labeled/visible |
| `complete` | `bool`       | Whether the point is fully visible  |
| `name`     | `object`     | Node name string                    |

!!! tip "Performance"
    Accessing `inst.points["xy"]` directly returns a view into the underlying
    array without copying. Use this when working with large datasets where
    `inst.numpy()` (which copies) would be too slow.

---

## Class diagram

The following diagram shows how the pose-related classes relate to each other:

``` mermaid
classDiagram
    class Skeleton {
        +List~Node~ nodes
        +List~Edge~ edges
        +List~Symmetry~ symmetries
        +str name
        +node_names: list[str]
        +edge_inds: list[tuple]
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

    class Instance {
        +PointsArray points
        +Skeleton skeleton
        +Track track
        +Identity identity
        +float identity_score
        +dict~str,Embedding~ embeddings
        +dict~str,object~ categories
        +numpy() ndarray
        +n_visible: int
        +is_empty: bool
    }

    class PredictedInstance {
        +float score
        +numpy(scores) ndarray
    }

    class Track {
        +str name
    }

    Skeleton "1" *-- "1..*" Node : contains
    Skeleton "1" *-- "0..*" Edge : contains
    Skeleton "1" *-- "0..*" Symmetry : contains
    Edge "1" --> "2" Node : connects
    Symmetry "1" --> "2" Node : pairs

    Instance "1" --> "1" Skeleton : uses
    Instance "0..1" --> "0..1" Track : belongs to
    PredictedInstance --|> Instance : inherits
```

---

## API reference

::: sleap_io.Skeleton

::: sleap_io.Node

::: sleap_io.Edge

::: sleap_io.Symmetry

::: sleap_io.Instance

::: sleap_io.PredictedInstance

::: sleap_io.Track
