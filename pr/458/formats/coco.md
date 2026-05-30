# COCO Format (.json)

[COCO](https://cocodataset.org/) (Common Objects in Context) format is widely used in computer vision and pose estimation. sleap-io provides full read and write support, making it compatible with tools like [mmpose](https://github.com/open-mmlab/mmpose), [CVAT](https://www.cvat.ai/), and other COCO-compatible frameworks.

!!! note "Unannotated images become empty frames"
    `load_coco` creates a `LabeledFrame` for every entry in the `images` array,
    including images with zero annotations — these become **empty**
    `LabeledFrame`s, so the frame count matches the input and unannotated images
    round-trip losslessly. One caveat: an image whose file path cannot be resolved
    is skipped, so a 0-annotation image with a missing file is dropped rather than
    preserved.

::: sleap_io.io.main.load_coco

::: sleap_io.io.main.save_coco
