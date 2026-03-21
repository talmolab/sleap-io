# Labels & frames

`Labels` is the top-level container in sleap-io -- the entry point for all pose tracking data. It holds videos, skeletons, labeled frames, and tracks. A `LabeledFrame` groups all annotations for a single video frame, while `SuggestionFrame` marks frames suggested for labeling.

---

## Labels

The `Labels` class is the primary data structure you will work with. It stores everything needed to represent a pose tracking project: videos, skeletons, labeled frames with instances, tracks, and metadata.

### Creating from scratch

Build a `Labels` object programmatically from skeleton definitions, videos, and instances:

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> labels = sio.Labels(
...     videos=[video], skeletons=[skeleton], labeled_frames=[lf]
... )
>>> labels
```

You do not need to manually specify `videos` and `skeletons` -- they are automatically collected from the labeled frames:

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> labels = sio.Labels(labeled_frames=[lf])
>>> labels.videos
>>> labels.skeletons
```

### Loading from file

Load labels from any supported format. The format is detected automatically from the file extension:

```python
import sleap_io as sio

labels = sio.load_file("predictions.slp")
# or
labels = sio.load_file("predictions.nwb")
```

### Inspecting

Check the contents of a labels dataset:

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> labels = sio.Labels(labeled_frames=[lf])
>>> len(labels)
>>> labels.videos
>>> labels.skeletons
>>> labels.skeleton  # shortcut when there's exactly one
```

### Querying

Retrieve labeled frames by index or by searching for a specific video and frame:

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> labels = sio.Labels(labeled_frames=[lf])
>>> labels[0]  # first labeled frame
>>> labels.find(video=video, frame_idx=0)
```

You can also index with a `(video, frame_idx)` tuple:

```python
lf = labels[video, 0]  # same as labels.find(video, 0)[0]
```

### Array conversion

Convert all tracked instances to a NumPy array for numerical analysis:

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> track = sio.Track("animal")
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
...     track=track,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> labels = sio.Labels(labeled_frames=[lf])
>>> trx = labels.numpy()
>>> trx.shape  # (n_frames, n_tracks, n_nodes, 2)
```

!!! note "See also"

    The [Codecs guide](../codecs.md) covers `to_dict()`, `to_dataframe()`, and other conversion methods in detail.

### Manipulation

`Labels` provides methods for modifying the skeleton and instance data across the entire dataset.

**Removing predictions** strips all `PredictedInstance` objects and cleans up empty frames:

```python
labels.remove_predictions()
```

**Renaming nodes** updates node names in the skeleton and all associated instances:

```python
labels.rename_nodes({"old_name": "new_name"})
# or rename all at once with a list
labels.rename_nodes(["head", "body", "tail"])
```

**Removing nodes** deletes nodes (and their associated edges and symmetries) from the skeleton and all instances:

```python
labels.remove_nodes(["node_to_remove"])
```

**Replacing the skeleton** swaps the skeleton used by all instances, mapping points from old nodes to new nodes by name (or via an explicit `node_map`):

```python
new_skeleton = sio.Skeleton(["HEAD", "BODY", "TAIL"])
labels.replace_skeleton(new_skeleton)
```

### Saving

Save labels to any supported format. The format is inferred from the file extension:

```python
labels.save("output.slp")
labels.save("output.nwb")
```

To embed images directly into an SLP file (creating a self-contained package):

```python
labels.save("output.pkg.slp", embed=True)
```

---

## Labeled frames

A `LabeledFrame` contains all annotations for a single frame of a video. Each frame holds a list of `Instance` and/or `PredictedInstance` objects.

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> lf.video
>>> lf.frame_idx
>>> lf.instances
>>> lf.user_instances
>>> lf.predicted_instances
>>> lf.has_user_instances
```

You can iterate over instances in a frame directly:

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> len(lf)  # number of instances
>>> lf[0]  # first instance
```

A frame can also be converted to a NumPy array:

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> lf.numpy().shape  # (n_instances, n_nodes, 2)
```

!!! note "See also"

    See [Poses & skeletons](poses.md) for details on `Instance`, `PredictedInstance`, and the `Skeleton` class.

---

## Suggestion frames

A `SuggestionFrame` is a lightweight pointer to a frame that has been suggested for annotation. Unlike `LabeledFrame`, it carries no instance data -- just a video reference and frame index.

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> sf = sio.SuggestionFrame(video=video, frame_idx=5)
>>> sf.video
>>> sf.frame_idx
```

Suggestions are stored on the `Labels` object:

```python
labels.suggestions.append(sio.SuggestionFrame(video=video, frame_idx=10))
```

---

## Labels set

A `LabelsSet` manages multiple `Labels` datasets as a named collection, useful for organizing train/validation/test splits.

```pycon exec="1" source="console"
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> labels = sio.Labels(labeled_frames=[lf])
>>> ls = sio.LabelsSet({"train": labels})
>>> ls.keys()
>>> ls["train"]
```

`LabelsSet` supports tuple-style unpacking and dictionary-style access:

```python
# Unpack like a tuple
train, val = labels_set

# Iterate like a dictionary
for name, split_labels in labels_set.items():
    print(f"{name}: {len(split_labels)} frames")
```

Save all splits at once:

```python
labels_set.save("splits/", embed=True)
# Creates: splits/train.pkg.slp, splits/val.pkg.slp, ...
```

---

## Class relationships

The following diagram shows how `Labels` connects to the other core data structures:

``` mermaid
classDiagram
    class Labels {
        +List~LabeledFrame~ labeled_frames
        +List~Video~ videos
        +List~Skeleton~ skeletons
        +List~Track~ tracks
        +List~SuggestionFrame~ suggestions
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

    class Video {
        +str filename
    }

    class Skeleton {
        +List~Node~ nodes
    }

    class Track {
        +str name
    }

    class SuggestionFrame {
        +Video video
        +int frame_idx
    }

    class LabelsSet {
        +Dict labels
    }

    Labels "1" *-- "0..*" LabeledFrame : labeled_frames
    Labels "1" *-- "0..*" Video : videos
    Labels "1" *-- "0..*" Skeleton : skeletons
    Labels "1" *-- "0..*" Track : tracks
    Labels "1" *-- "0..*" SuggestionFrame : suggestions
    LabeledFrame "0..*" --> "1" Video : references
    LabeledFrame "1" *-- "0..*" Instance : contains
    Instance "0..*" --> "1" Skeleton : uses
    Instance "0..*" --> "0..1" Track : belongs to
    SuggestionFrame "0..*" --> "1" Video : references
    LabelsSet "1" *-- "1..*" Labels : contains
```

!!! note "See also"

    - [Poses & skeletons](poses.md) for `Instance`, `PredictedInstance`, `Skeleton`, `Node`, and `Track`
    - [Examples](../examples.md) for end-to-end workflows
    - [Merging annotations](../merging.md) for combining datasets with `Labels.merge()`

---

## API reference

::: sleap_io.Labels

::: sleap_io.LabeledFrame

::: sleap_io.SuggestionFrame

::: sleap_io.LabelsSet
