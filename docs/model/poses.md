# Poses

This page covers how sleap-io represents pose data:from defining a body plan
with `Skeleton` to storing actual landmark positions with `Instance`. Together,
these two classes form the core of the pose tracking data model: the skeleton
says *what* to label, and instances record *where* each landmark is.

---

## Overview

The pose data model is built around four key types:

- **`Skeleton`** is the **template**: defines what landmarks exist, how they connect, and which are symmetric.
- **`Instance`** is the **data**: stores actual (x, y) coordinates for one animal in one frame.
- **`PredictedInstance`** is like `Instance` but includes per-point and instance-level confidence scores from a model.
- **`Track`** is the **identity**: links the same animal across frames.

A `Skeleton` is shared across all instances in a dataset. Each `Instance` references a `Skeleton` to know which landmarks it contains, and optionally a `Track` to indicate which animal it belongs to.

---

## Skeleton

A `Skeleton` is a **template** that defines what landmarks (body parts) exist
and how they connect. Think of it as a form to fill in:the skeleton says
"head, thorax, abdomen" while instances fill in the actual (x, y) coordinates.

Skeletons are composed of three building blocks:

| Component    | Purpose                                         |
| ------------ | ----------------------------------------------- |
| `Node`       | A single landmark type (e.g. "head")            |
| `Edge`       | A directed connection between two nodes          |
| `Symmetry`   | A left/right pairing (e.g. "left eye" / "right eye") |

### Creating a skeleton

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(
...     nodes=["head", "thorax", "abdomen"],
...     edges=[("head", "thorax"), ("thorax", "abdomen")],
... )
>>> skeleton
Skeleton(nodes=["head", "thorax", "abdomen"], edges=[(0, 1), (1, 2)])
>>> len(skeleton)
3
>>> skeleton.node_names
['head', 'thorax', 'abdomen']
>>> skeleton.edge_inds
[(0, 1), (1, 2)]

```

Nodes and edges can be specified as strings or indices:they are converted to
`Node` and `Edge` objects automatically.

### Accessing nodes

Nodes can be retrieved by **name** or **integer index**, and you can look up a
node's index in the skeleton:

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(
...     nodes=["head", "thorax", "abdomen"],
...     edges=[("head", "thorax"), ("thorax", "abdomen")],
... )
>>> skeleton["head"]
Node(name='head')
>>> skeleton[0]
Node(name='head')
>>> skeleton.index("thorax")
1
>>> "head" in skeleton
True

```

### Symmetries

Symmetries record which nodes are left/right mirrors of each other. This is used
during data augmentation (horizontal flipping) to swap the correct landmark
indices.

```pycon
>>> import sleap_io as sio
>>> skel = sio.Skeleton(["A", "B_left", "B_right"])
>>> skel.add_symmetry("B_left", "B_right")
>>> skel.symmetry_names
[('B_left', 'B_right')]

```

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

An `Instance` is one animal's pose in one frame:the "filled-in form." It
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
>>> inst
Instance(points=[[10.2, 20.4], [5.8, 15.1], [0.3, 10.6]], track=None)
>>> inst.numpy()
array([[10.2, 20.4],
       [ 5.8, 15.1],
       [ 0.3, 10.6]])

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
>>> inst["head"]["xy"]
array([10.2, 20.4])
>>> inst["head"]["visible"]
True
>>> inst.n_visible
3
>>> inst.is_empty
False

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
>>> inst.numpy()
array([[10., 20.],
       [ 5., 15.],
       [ 0., 10.]])

```

### Empty instances

Create an instance with no visible points (all coordinates are unset):

```pycon
>>> import sleap_io as sio
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> empty_inst = sio.Instance.empty(skeleton=skeleton)
>>> empty_inst.is_empty
True

```

!!! note "See also"
    Instances are organized into frames and datasets through
    [`LabeledFrame`](labels.md#labeled-frames) and [`Labels`](labels.md#labels).
    See the [Labels & Frames](labels.md) page for the full picture.

---

## Predicted instances

`PredictedInstance` extends `Instance` with confidence scores:both a
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
>>> pred.score
0.85
>>> pred.numpy(scores=True)
array([[10.2, 20.4,  0.9],
       [ 5.8, 15.1,  0.8],
       [ 0.3, 10.6,  0.7]])

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
>>> inst.track.name
'animal_1'

```

!!! note
    `Track` objects are compared by **identity** (not by name). Two different
    `Track("mouse")` objects are considered distinct:this allows multiple
    tracks with the same display name if needed.

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
>>> inst.points.dtype.names
('xy', 'visible', 'complete', 'name')
>>> inst.points["xy"]
array([[10., 20.],
       [ 5., 15.],
       [ 0., 10.]])
>>> inst.points["visible"]
array([ True,  True,  True])

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
