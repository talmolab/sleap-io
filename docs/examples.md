# Examples

## Load and save in different formats

```py
import sleap_io as sio

# Load from SLEAP file.
labels = sio.load_file("predictions.slp")

# Save to NWB file.
labels.save("predictions.nwb")
```

**See also:** [`Labels.save`](model.md#sleap_io.Labels.save) and [Formats](formats.md)


## Convert labels to raw arrays

```py
import sleap_io as sio

labels = sio.load_slp("tests/data/slp/centered_pair_predictions.slp")

# Convert predictions to point coordinates in a single array.
trx = labels.numpy()
n_frames, n_tracks, n_nodes, xy = trx.shape
assert xy == 2

# Convert to array with confidence scores appended.
trx_with_scores = labels.numpy(return_confidence=True)
n_frames, n_tracks, n_nodes, xy_score = trx.shape 
assert xy_score == 3
```

**See also:** [`Labels.numpy`](model.md#sleap_io.Labels.numpy)


## Read video data

```py
import sleap_io as sio

video = sio.load_video("test.mp4")
n_frames, height, width, channels = video.shape

frame = video[0]
height, width, channels = frame.shape
```

**See also:** [`sio.load_video`](formats.md#sleap_io.load_video) and [`Video`](model.md#sleap_io.Video)



## Create labels from raw data

```py
import sleap_io as sio
import numpy as np

# Create skeleton.
skeleton = sio.Skeleton(
    nodes=["head", "thorax", "abdomen"],
    edges=[("head", "thorax"), ("thorax", "abdomen")]
)

# Create video.
video = sio.load_video("test.mp4")

# Create instance.
instance = sio.Instance.from_numpy(
    points=np.array([
        [10.2, 20.4],
        [5.8, 15.1],
        [0.3, 10.6],
    ]),
    skeleton=skeleton
)

# Create labeled frame.
lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])

# Create labels.
labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[lf])

# Save.
labels.save("labels.slp")
```

**See also:** [Model](model.md), [`Labels`](model.md#sleap_io.Labels),
[`LabeledFrame`](model.md#sleap_io.LabeledFrame),
[`Instance`](model.md#sleap_io.Instance),
[`PredictedInstance`](model.md#sleap_io.PredictedInstance),
[`Skeleton`](model.md#sleap_io.Skeleton), [`Video`](model.md#sleap_io.Video), [`Track`](model.md#sleap_io.Track), [`SuggestionFrame`](model.md#sleap_io.SuggestionFrame)


## Fix video paths

```py
import sleap_io as sio

# Load labels without trying to open the video files.
labels = sio.load_file("labels.v001.slp", open_videos=False)

# Fix paths using prefix replacement.
labels.replace_filenames(prefix_map={
    "D:/data/sleap_projects": "/home/user/sleap_projects",
    "C:/Users/sleaper/Desktop/test": "/home/user/sleap_projects",
})

# Save labels with updated paths.
labels.save("labels.v002.slp")
```

**See also:** [`Labels.replace_filenames`](model.md#sleap_io.Labels.replace_filenames)


## Save labels with embedded images

```py
import sleap_io as sio

# Load source labels.
labels = sio.load_file("labels.v001.slp")

# Save with embedded images for frames with user labeled data and suggested frames.
labels.save("labels.v001.pkg.slp", embed="user+suggestions")
```

**See also:** [`Labels.save`](model.md#sleap_io.Labels.save)


## Make training/validation/test splits

```py
import sleap_io as sio

# Load source labels.
labels = sio.load_file("labels.v001.slp")

# Make splits and export with embedded images.
labels.make_training_splits(n_train=0.8, n_val=0.1, n_test=0.1, save_dir="split1", seed=42)

# Splits will be saved as self-contained SLP package files with images and labels.
labels_train = sio.load_file("split1/train.pkg.slp")
labels_val = sio.load_file("split1/val.pkg.slp")
labels_test = sio.load_file("split1/test.pkg.slp")
```

**See also:** [`Labels.make_training_splits`](model.md#sleap_io.Labels.make_training_splits)


## Reencode video

Some video formats are not readily seekable at frame-level accuracy. By reencoding them
with the default settings in our video writer, they will be reliably seekable with
minimal loss of quality and can be achieved in a single line:

```py
import sleap_io as sio

sio.save_video(sio.load_video("input.mp4"), "output.mp4")
```

**See also:** [`save_video`](formats.md#sleap_io.save_video)


## Trim labels and video

It can be sometimes be useful to pull out a short clip of frames, either for sharing or
for generating data on only a subset of the video. We can do this with the following
recipe:

```py
import sleap_io as sio

# Load existing data.
labels = sio.load_file("labels.slp")

# Create a new labels file with data from frames 1000-2000 in video 0.
# Note: a new video will be saved with filename "clip.mp4" and frame indices adjusted in
# the labels.
clip = labels.trim("clip.slp", list(range(1_000, 2_000)), video=0)
```

**See also:** [`Labels.trim`](model.md#sleap_io.Labels.trim)


## Replace skeleton

[`Skeleton`](model.md#sleap_io.Skeleton) objects hold metadata about the keypoints,
their ordering, names and connections. When converting between different annotation
formats, it can be useful to change skeletons while retaining as much information as
possible. We can do this as follows:

```py
import sleap_io as sio

# Load existing labels with skeleton with nodes: "head", "trunk", "tti"
labels = sio.load_file("labels.slp")

# Create a new skeleton with different nodes.
new_skeleton = sio.Skeleton(["HEAD", "CENTROID", "TAIL_BASE" "TAIL_TIP"])

# Replace the skeleton with correspondences where possible.
labels.replace_skeleton(
    new_skeleton,
    node_map={
        "head": "HEAD",
        "trunk": "CENTROID",
        "tti": "TAIL_BASE"
    }
)

# Save with the new skeleton format.
labels.save("labels_with_new_skeleton.slp")
```

**See also:** [`Labels.replace_skeleton`](model.md#sleap_io.Labels.replace_skeleton)