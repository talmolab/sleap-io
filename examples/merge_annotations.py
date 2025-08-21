#!/usr/bin/env python
"""
Example script demonstrating how to merge annotation files using sleap-io.

This script shows common merging scenarios including:
- Basic merging of two annotation files
- Human-in-the-loop workflow (merging predictions with manual labels)
- Handling conflicts and errors
- Custom matching configurations
"""

import argparse
from pathlib import Path

import numpy as np
from sleap_io import Labels, load_file, save_file
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.matching import InstanceMatcher, InstanceMatchMethod
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.video import Video


def create_sample_labels():
    """Create sample Labels objects for demonstration."""
    # Create a simple skeleton
    skeleton = Skeleton(["head", "thorax", "abdomen", "wingtip_left", "wingtip_right"])
    skeleton.add_edge("head", "thorax")
    skeleton.add_edge("thorax", "abdomen")
    skeleton.add_edge("thorax", "wingtip_left")
    skeleton.add_edge("thorax", "wingtip_right")
    skeleton.add_symmetry("wingtip_left", "wingtip_right")

    # Create a dummy video
    video = Video(filename="sample_video.mp4", open_backend=False)

    # Create manual annotations
    manual_labels = Labels()
    manual_labels.videos.append(video)
    manual_labels.skeletons.append(skeleton)

    # Add some manual annotations
    for frame_idx in [0, 5, 10]:
        frame = LabeledFrame(video=video, frame_idx=frame_idx)
        # Add a user-labeled instance
        points = np.random.randn(5, 2) * 20 + [100, 100]
        instance = Instance.from_numpy(points, skeleton=skeleton)
        frame.instances.append(instance)
        manual_labels.append(frame)

    # Create predictions
    predicted_labels = Labels()
    predicted_labels.videos.append(video)
    predicted_labels.skeletons.append(skeleton)

    # Add predictions for all frames
    for frame_idx in range(15):
        frame = LabeledFrame(video=video, frame_idx=frame_idx)
        # Add predicted instances
        for i in range(2):  # Two predicted instances per frame
            points = np.random.randn(5, 2) * 20 + [100 + i * 50, 100]
            scores = np.random.uniform(0.5, 1.0, size=5)
            instance = PredictedInstance.from_numpy(
                points, skeleton=skeleton, point_scores=scores, score=scores.mean()
            )
            frame.instances.append(instance)
        predicted_labels.append(frame)

    return manual_labels, predicted_labels


def example_basic_merge():
    """Demonstrate basic merging of two annotation files."""
    print("\n=== Basic Merge Example ===")

    # Create sample data
    manual, predictions = create_sample_labels()
    print(f"Manual labels: {len(manual.labeled_frames)} frames, "
          f"{sum(len(f.instances) for f in manual.labeled_frames)} instances")
    print(f"Predictions: {len(predictions.labeled_frames)} frames, "
          f"{sum(len(f.instances) for f in predictions.labeled_frames)} instances")

    # Perform basic merge
    result = manual.merge(predictions)

    print("\nMerge Result:")
    print(result.summary())

    print(f"\nAfter merge: {len(manual.labeled_frames)} frames, "
          f"{sum(len(f.instances) for f in manual.labeled_frames)} instances")


def example_hitl_workflow():
    """Demonstrate human-in-the-loop workflow."""
    print("\n=== Human-in-the-Loop Workflow ===")

    # Create sample data
    manual, predictions = create_sample_labels()

    # Merge with smart strategy (preserves manual labels)
    result = manual.merge(
        predictions,
        frame_strategy="smart",
        validate=True,
        error_mode="warn"
    )

    print("HITL Merge Result:")
    print(f"- Frames merged: {result.frames_merged}")
    print(f"- Instances added: {result.instances_added}")
    print(f"- Instances updated: {result.instances_updated}")
    print(f"- Conflicts resolved: {len(result.conflicts)}")

    # Analyze conflicts
    if result.conflicts:
        print("\nConflicts:")
        for conflict in result.conflicts[:5]:  # Show first 5 conflicts
            frame_idx = conflict[0].frame_idx if hasattr(conflict[0], 'frame_idx') else 'unknown'
            print(f"  - Frame {frame_idx}: {conflict[2]}")


def example_custom_matching():
    """Demonstrate custom instance matching configurations."""
    print("\n=== Custom Matching Example ===")

    # Create sample data
    manual, predictions = create_sample_labels()

    # Test different matching methods
    matching_configs = [
        ("Spatial (5px)", InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=5.0)),
        ("Spatial (20px)", InstanceMatcher(method=InstanceMatchMethod.SPATIAL, threshold=20.0)),
        ("IoU (0.3)", InstanceMatcher(method=InstanceMatchMethod.IOU, threshold=0.3)),
    ]

    for name, matcher in matching_configs:
        # Create fresh copies for each test
        manual_copy = create_sample_labels()[0]
        predictions_copy = create_sample_labels()[1]

        result = manual_copy.merge(
            predictions_copy,
            instance_matcher=matcher,
            frame_strategy="smart"
        )

        print(f"\n{name} matching:")
        print(f"  - Instances added: {result.instances_added}")
        print(f"  - Instances skipped: {result.instances_skipped}")
        print(f"  - Conflicts: {len(result.conflicts)}")


def example_merge_strategies():
    """Demonstrate different merge strategies."""
    print("\n=== Merge Strategies Example ===")

    strategies = ["smart", "keep_original", "keep_new", "keep_both"]

    for strategy in strategies:
        # Create fresh data for each strategy
        manual, predictions = create_sample_labels()
        initial_count = sum(len(f.instances) for f in manual.labeled_frames)

        result = manual.merge(predictions, frame_strategy=strategy)

        final_count = sum(len(f.instances) for f in manual.labeled_frames)

        print(f"\nStrategy '{strategy}':")
        print(f"  - Initial instances: {initial_count}")
        print(f"  - Final instances: {final_count}")
        print(f"  - Instances added: {result.instances_added}")
        print(f"  - Instances skipped: {result.instances_skipped}")


def example_merge_from_files(base_path: str, new_path: str, output_path: str = None):
    """Merge actual annotation files."""
    print("\n=== Merging Files ===")
    print(f"Base: {base_path}")
    print(f"New: {new_path}")

    # Load files
    base_labels = load_file(base_path)
    new_labels = load_file(new_path)

    print(f"\nBase labels: {len(base_labels.labeled_frames)} frames")
    print(f"New labels: {len(new_labels.labeled_frames)} frames")

    # Perform merge
    result = base_labels.merge(
        new_labels,
        frame_strategy="smart",
        validate=True,
        error_mode="warn"
    )

    print("\nMerge Result:")
    print(result.summary())

    # Save if output path provided
    if output_path:
        save_file(base_labels, output_path)
        print(f"\nMerged labels saved to: {output_path}")

    return base_labels


def main():
    parser = argparse.ArgumentParser(
        description="Example script for merging SLEAP annotation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demonstration with synthetic data
  python merge_annotations.py --demo

  # Merge two annotation files
  python merge_annotations.py base.slp new.slp -o merged.slp

  # Merge with specific strategy
  python merge_annotations.py base.slp new.slp -s keep_both -o merged.slp
        """
    )

    parser.add_argument("base", nargs="?", help="Base annotation file")
    parser.add_argument("new", nargs="?", help="New annotation file to merge")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-s", "--strategy",
                        choices=["smart", "keep_original", "keep_new", "keep_both"],
                        default="smart",
                        help="Merge strategy (default: smart)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demonstration with synthetic data")

    args = parser.parse_args()

    if args.demo:
        # Run all examples with synthetic data
        example_basic_merge()
        example_hitl_workflow()
        example_custom_matching()
        example_merge_strategies()
    elif args.base and args.new:
        # Merge actual files
        example_merge_from_files(args.base, args.new, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()