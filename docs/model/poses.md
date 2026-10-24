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

A `Skeleton` is shared across all instances in a dataset. Each `Instance` references a `Skeleton` to know which landmarks it contains, optionally a `Track` (and a global `Identity`) to indicate which animal it belongs to, and optionally a re-ID [`identity_embedding`](embedding.md) describing its appearance.

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

Create an instance where every point is labeled but hidden, with no location (NaN
coordinates):

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> empty_inst = sio.Instance.empty(skeleton=skeleton)
>>> print(empty_inst.is_empty)

```

See [Point flags and states](#point-flags-and-states) for what `visible`,
`labeled` and NaN coordinates mean, and [Emptiness](#emptiness) for when an
instance counts as empty.

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
| `xy`       | `float64[2]` | (x, y) coordinates; NaN means the point has no location |
| `score`    | `float64`    | Per-point confidence (predicted only) |
| `visible`  | `bool`       | Whether the point is visible in the image (`False` covers occluded and out of view) |
| `verified` | `bool`       | Set by the SLEAP GUI when a human placed, moved, or toggled visibility of the point (stored, not interpreted). Named `complete` before sleap-io 0.10 |
| `labeled`  | `bool`       | Whether the source annotated this node (`False` means unlabeled). Defaults to `True` |
| `name`     | `object`     | Node name string                    |

!!! tip "Performance"
    Accessing `inst.points["xy"]` directly returns a view into the underlying
    array without copying. Use this when working with large datasets where
    `inst.numpy()` (which copies) would be too slow.

---

## Point flags and states

Each point carries two independent flags plus its coordinates:

- `visible`: the point is visible in the image. `False` covers both occluded and
  out-of-view points.
- `labeled`: the source annotated this node. `False` means *unlabeled*: the
  annotator (or model) said nothing about it, so training should give it zero
  weight.
- `xy`: NaN means "no location". Finite coordinates on a hidden point are an
  estimate of where the occluded landmark is.

Together they give four [`PointState`][sleap_io.PointState]s, numbered to match
COCO's keypoint visibility `v` (map `ABSENT` to `0` to get COCO's value):

| `PointState`    | `visible` | `labeled` | `xy`   | Meaning                                    |
| --------------- | --------- | --------- | ------ | ------------------------------------------ |
| `VISIBLE` (2)   | `True`    | `True`    | finite | Visible at this location                   |
| `OCCLUDED` (1)  | `False`   | `True`    | finite | Hidden, with an estimated location          |
| `ABSENT` (3)    | `False`   | `True`    | NaN    | Hidden with no location (occluded somewhere unknown, out of view, or absent) |
| `UNLABELED` (0) | `False`   | `False`   | NaN    | Not annotated; ignore in training          |

A few rules keep these consistent:

1. **Visible implies labeled.** Helpers read the flag as `labeled | visible`, so a
   visible point always counts as labeled.
2. **NaN means no location.** Everything that creates points
   ([`Instance.empty`][sleap_io.Instance.empty], dictionaries with missing nodes,
   skeleton changes, merging) fills them as labeled, hidden and NaN. If you set
   `points["visible"] = False` directly, clear `points["xy"]` too unless the
   coordinates are a real estimate, or use
   [`set_hidden`][sleap_io.Instance.set_hidden].
3. **`labeled` defaults to `True`.** You only need to set it to mark nodes as
   unlabeled.
4. **`verified` is provenance only.** It records that a human acted on the node in
   the SLEAP GUI; sleap-io never interprets it.

[`Instance.from_numpy`][sleap_io.Instance.from_numpy] accepts `visible`,
`labeled` and `verified` arrays (or scalars), and the instance has helpers to
change and inspect states:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen", "tail"])
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40], [50, 60], [np.nan, np.nan]]),
...     skeleton=skeleton,
...     visible=[True, True, False, False],  # abdomen: occluded estimate
...     labeled=[True, True, True, False],  # tail: unlabeled
... )
>>> print([sio.PointState(s).name for s in inst.point_states])
>>> print(inst.n_visible, inst.n_labeled)
>>> inst.set_hidden("thorax")  # labeled, hidden, NaN
>>> inst.set_hidden("head", keep_location=True)  # keep xy as an occluded estimate
>>> inst.set_unlabeled("abdomen")
>>> inst["tail"] = (70, 80)  # assigning a point marks it visible and labeled
>>> print([sio.PointState(s).name for s in inst.point_states])

```

[`Instance.point_states`][sleap_io.Instance.point_states] returns an `int8`
array, and [`Labels.numpy(return_states=True)`](labels.md#array-conversion)
returns the same states for a whole dataset.
[`PredictedInstance.from_numpy`][sleap_io.PredictedInstance.from_numpy] takes the
same flags; there, `labeled=False` marks nodes a partial-skeleton model has no
output for.

### Emptiness

[`Instance.is_empty`][sleap_io.Instance.is_empty] is `True` when no labeled point
has finite coordinates. An instance whose points are all occluded with estimated
locations is **not** empty (even though `n_visible == 0`), so it survives
`Labels.remove_empty_instances()` and `Labels.clean()`:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax"])
>>> occluded = sio.Instance.from_numpy(
...     np.array([[10, 20], [30, 40]]), skeleton=skeleton, visible=False
... )
>>> print(occluded.n_visible, occluded.is_empty)
>>> print(sio.Instance.empty(skeleton=skeleton).is_empty)

```

### Changing the skeleton

When an instance moves to a skeleton with nodes it has no data for
([`Instance.replace_skeleton`][sleap_io.Instance.replace_skeleton],
[`Instance.update_skeleton`][sleap_io.Instance.update_skeleton], or
[`Labels.replace_skeleton`][sleap_io.Labels.replace_skeleton]), the `unmapped`
argument decides how those nodes are filled:

- `unmapped="unlabeled"` (default): the new nodes are unlabeled, so training
  ignores them.
- `unmapped="absent"`: the new nodes are labeled, hidden and NaN, i.e. annotated
  as not present (the behavior before sleap-io 0.10).

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax"])
>>> inst = sio.Instance.from_numpy(np.array([[10, 20], [30, 40]]), skeleton=skeleton)
>>> inst.replace_skeleton(sio.Skeleton(["head", "thorax", "tail"]))
>>> print([sio.PointState(s).name for s in inst.point_states])

```

!!! warning "Saving unlabeled points to SLP"
    The current SLP layout has no `labeled` column, so
    [`save_slp`][sleap_io.save_slp] refuses to write unlabeled points by default
    (`unlabeled_as="error"`). Pass `unlabeled_as="absent"` to save them as hidden
    points instead (with a warning). See [Labels → Saving](labels.md#saving).

---

## Metadata

`Instance` and `PredictedInstance` (as well as
[`LabeledFrame`](labels.md#labeled-frames) and [bounding boxes](boxes.md#metadata-fields))
have a `metadata` dictionary for per-record values such as COCO's `iscrowd` or a
source annotation id:

```pycon
>>> import numpy as np
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax"])
>>> inst = sio.Instance.from_numpy(np.array([[10, 20], [30, 40]]), skeleton=skeleton)
>>> inst.metadata["iscrowd"] = 0
>>> inst.metadata["area"] = np.float32(12.5)  # numpy scalars become Python types
>>> print(inst.metadata)
>>> try:
...     inst.metadata["keypoint_ids"] = [1, 2]
... except TypeError as err:
...     print(err)

```

- Values must be `bool`, `int`, `float` or `str`; `None` means missing. Nested
  values raise `TypeError`, so JSON-encode them into a `str`.
- Keys are non-empty strings without `/`.
- Values are validated when they are set (on assignment, `update`, or when
  replacing the whole dictionary).
- Each key must hold one kind of value across a table (e.g. all instances): `int`
  and `float` mix as `float`, but integers beyond 2^53 can't be mixed with floats.
- Metadata travels with copies and merges. When frames are merged, their metadata
  is combined key by key, with the target frame winning ties.

!!! note
    The current SLP layout doesn't store metadata yet: `save_slp` drops it with a
    warning.

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
        +Embedding identity_embedding
        +dict metadata
        +numpy() ndarray
        +n_visible: int
        +n_labeled: int
        +point_states: ndarray
        +is_empty: bool
        +set_hidden(nodes, keep_location)
        +set_unlabeled(nodes)
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

::: sleap_io.PointState

::: sleap_io.Track
