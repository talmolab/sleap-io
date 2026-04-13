# Labels

[`Labels`](#sleap_io.Labels) is the top-level container in sleap-io -- the entry point for all pose tracking data. It holds videos, skeletons, labeled frames, and tracks. A [`LabeledFrame`](#sleap_io.LabeledFrame) groups all annotations for a single video frame, while [`SuggestionFrame`](#sleap_io.SuggestionFrame) marks frames suggested for labeling.

## Container hierarchy

The core data structures form a layered hierarchy:

- **[`Labels`](#sleap_io.Labels)** is the top-level container: it owns everything in a project.
- **[`LabeledFrame`](#sleap_io.LabeledFrame)** groups all annotations for one frame of one video — pose [`instances`](poses.md), [`centroids`](regions.md), [`bboxes`](regions.md), [`masks`](regions.md), [`label_images`](regions.md), and [`rois`](regions.md).
- **[`SuggestionFrame`](#sleap_io.SuggestionFrame)** is a lightweight pointer to frames suggested for annotation (no instance data).
- **[`LabelsSet`](#sleap_io.LabelsSet)** manages named collections of `Labels` (e.g., train/val/test splits).

The hierarchy flows as **Labels -> LabeledFrame -> [`Instance`](poses.md)** (or any spatial annotation type). `Labels` also holds shared objects ([videos](video.md), [skeletons](poses.md), [tracks](poses.md), [identities](3d.md#identity), and `static_rois` for video-level ROIs) that are referenced by frames and instances, so each object is stored once and reused throughout the project. Flat read-only views like `labels.centroids`, `labels.bboxes`, `labels.masks`, `labels.label_images`, and `labels.rois` flatten the per-frame annotations across every `LabeledFrame` for convenient iteration.

---

## Labels

The `Labels` class is the primary data structure you will work with. It stores everything needed to represent a pose tracking project: videos, skeletons, labeled frames with instances, tracks, and metadata.

### Creating from scratch

Build a `Labels` object programmatically from skeleton definitions, videos, and instances:

```pycon
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
>>> print(labels)
```

You do not need to manually specify `videos` and `skeletons` — they are automatically collected from the labeled frames:

```pycon
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
>>> print(labels.videos)
>>> print(labels.skeletons)
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

```pycon
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
>>> print(len(labels))
>>> print(labels.videos)
>>> print(labels.skeletons)
>>> print(labels.skeleton)  # shortcut when there's exactly one
```

### Querying

Retrieve labeled frames by index or by searching for a specific video and frame:

```pycon
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
>>> print(labels[0])  # first labeled frame
>>> print(labels.find(video=video, frame_idx=0))
```

You can also index with a `(video, frame_idx)` tuple:

```python
lf = labels[video, 0]  # same as labels.find(video, 0)[0]
```

### Fast lookups

`Labels` maintains lazy indices for O(1) frame and track lookups. Use
`get_frame()` instead of `find()` when you need a single frame:

```pycon
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
>>> print(labels.get_frame(video, 0))
>>> print(labels.get_frame(video, 999))

```

Track-level queries return all annotations for a given track across frames,
sorted by frame index:

```pycon
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
>>> annotations = labels.get_track_annotations(video, track)
>>> print(len(annotations))

```

After batch mutations (e.g., changing `frame_idx` or track assignments
directly), call `reindex()` to rebuild the indices:

```python
labels.reindex()
```

### Array conversion

Convert all tracked instances to a NumPy array for numerical analysis:

```pycon
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
>>> print(trx.shape)  # (n_frames, n_tracks, n_nodes, 2); n_frames == len(video) in v0.7.0
```

In v0.7.0, the frame dimension equals `len(video)` instead of `last_labeled_frame + 1` (PR #368). Frames past the last labeled frame are NaN-padded. Code that previously sized downstream arrays from this shape may need updating.

!!! note "See also"

    The [Codecs guide](../codecs.md) covers `to_dict()`, `to_dataframe()`, and other conversion methods in detail.

### Manipulation

`Labels` provides methods for modifying the skeleton and instance data across the entire dataset.

**Removing predictions** strips all [`PredictedInstance`](poses.md) objects and cleans up empty frames:

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

**Adding spatial annotations** to a frame by appending to its annotation lists. `LabeledFrame.append()` dispatches by type, routing each annotation into the correct list (`instances`, `centroids`, `bboxes`, `masks`, `label_images`, or `rois`):

```python
lf = labels.get_frame(video, frame_idx=0)  # O(1) frame lookup
lf.append(sio.UserCentroid(x=100, y=200))         # → lf.centroids
lf.append(sio.UserBoundingBox(x1=10, y1=20, x2=50, y2=60))  # → lf.bboxes
lf.append(mask)                                    # → lf.masks
lf.append(label_image)                             # → lf.label_images
lf.append(roi)                                     # → lf.rois
```

**Cleaning up unused state** with [`Labels.clean()`](#sleap_io.Labels.clean): removes empty frames, unused tracks, unused skeletons, and orphaned annotations whose track is no longer present. As of v0.7.0, frames that contain any annotation type (centroids, bboxes, masks, label_images, rois) are preserved even if they have no pose instances — only frames with zero annotations of any kind are dropped.

```python
labels.clean()  # remove empty frames + unused tracks/skeletons + orphan annotations
```

**Static ROIs** (video-level regions, e.g. arena boundaries) live on `labels.static_rois` instead of inside a `LabeledFrame`. The list is mutable, so you can append to it directly:

```python
arena = sio.UserROI(geometry=polygon, video=video, category="arena")
labels.static_rois.append(arena)
```

**Closing lazy file handles** with `labels.close()`: when a labels file was loaded with `lazy=True` and contains chunked label image data, h5py file handles are kept open for on-demand decompression. Call `close()` to release them when finished.

!!! note "See also"

    See [Regions](regions.md) for details on creating spatial annotation objects
    and [Working with annotations in frames](regions.md#working-with-annotations-in-frames)
    for more examples.

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

A [`LabeledFrame`](#sleap_io.LabeledFrame) contains all annotations for a single frame of a [`Video`](video.md). Each frame holds a list of [`Instance`](poses.md) and/or [`PredictedInstance`](poses.md) objects, along with optional spatial annotations: [`centroids`](regions.md), [`bboxes`](regions.md) (bounding boxes), [`masks`](regions.md) (segmentation masks), [`label_images`](regions.md), and [`rois`](regions.md) (regions of interest).

```pycon
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> print(lf.video)
>>> print(lf.frame_idx)
>>> print(lf.instances)
>>> print(lf.user_instances)
>>> print(lf.predicted_instances)
>>> print(lf.has_user_instances)
```

You can iterate over instances in a frame directly:

```pycon
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> print(len(lf))  # number of instances
>>> print(lf[0])  # first instance
```

A frame can also be converted to a NumPy array:

```pycon
>>> import sleap_io as sio
>>> import numpy as np
>>> skeleton = sio.Skeleton(["head", "thorax", "abdomen"])
>>> video = sio.Video("test.mp4", open_backend=False)
>>> inst = sio.Instance.from_numpy(
...     np.array([[10, 20], [5, 15], [0, 10]]),
...     skeleton=skeleton,
... )
>>> lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
>>> print(lf.numpy().shape)  # (n_instances, n_nodes, 2)
```

!!! note "See also"

    See [Poses & skeletons](poses.md) for details on `Instance`, `PredictedInstance`, and the `Skeleton` class.

---

## Suggestion frames

A [`SuggestionFrame`](#sleap_io.SuggestionFrame) is a lightweight pointer to a frame that has been suggested for annotation. Unlike [`LabeledFrame`](#sleap_io.LabeledFrame), it carries no instance data -- just a video reference and frame index.

```pycon
>>> import sleap_io as sio
>>> video = sio.Video("test.mp4", open_backend=False)
>>> sf = sio.SuggestionFrame(video=video, frame_idx=5)
>>> print(sf.video)
>>> print(sf.frame_idx)
```

Suggestions are stored on the `Labels` object:

```python
labels.suggestions.append(sio.SuggestionFrame(video=video, frame_idx=10))
```

---

## Labels set

A [`LabelsSet`](#sleap_io.LabelsSet) manages multiple [`Labels`](#sleap_io.Labels) datasets as a named collection, useful for organizing train/validation/test splits.

```pycon
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
>>> print(ls.keys())
>>> print(ls["train"])
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
        +List~Centroid~ centroids
        +List~BoundingBox~ bboxes
        +List~SegmentationMask~ masks
        +List~LabelImage~ label_images
        +List~ROI~ rois
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
    LabeledFrame "1" *-- "0..*" Centroid : centroids
    LabeledFrame "1" *-- "0..*" BoundingBox : bboxes
    LabeledFrame "1" *-- "0..*" SegmentationMask : masks
    LabeledFrame "1" *-- "0..*" LabelImage : label_images
    LabeledFrame "1" *-- "0..*" ROI : rois
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
