"""Demonstration of CVAT dataset merging with image deduplication."""

import json
from pathlib import Path
import tempfile

import sleap_io as sio
from sleap_io.io import coco
from sleap_io.model.matching import IMAGE_DEDUP_VIDEO_MATCHER, SHAPE_VIDEO_MATCHER


def create_mock_images(json_file, output_dir):
    """Create empty image files based on JSON metadata."""
    with open(json_file) as f:
        data = json.load(f)
    
    # Create dummy image files
    for img_info in data["images"]:
        img_path = Path(output_dir) / img_info["file_name"]
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.touch()
    
    return len(data["images"])


def main():
    """Demonstrate merging CVAT datasets with different strategies."""
    
    # Check if CVAT files exist
    cvat1 = Path("tmp/skeletons-batch2.s1.p0.json")
    cvat2 = Path("tmp/skeletons-batch3.s1.p2.json")
    
    if not cvat1.exists() or not cvat2.exists():
        print("CVAT files not found in tmp/")
        return
    
    print("=" * 60)
    print("CVAT Dataset Merging Demonstration")
    print("=" * 60)
    
    # Create temporary directory with mock images
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nCreating mock images in temporary directory...")
        n1 = create_mock_images(cvat1, tmpdir)
        n2 = create_mock_images(cvat2, tmpdir)
        print(f"Created {n1 + n2} mock image files")
        
        # Load datasets
        print("\nLoading CVAT datasets...")
        labels1 = coco.read_labels(cvat1, dataset_root=tmpdir)
        labels2 = coco.read_labels(cvat2, dataset_root=tmpdir)
        
        print(f"Dataset 1: {len(labels1.labeled_frames)} frames, {len(labels1.videos)} videos")
        print(f"Dataset 2: {len(labels2.labeled_frames)} frames, {len(labels2.videos)} videos")
        
        # Get video objects
        video1 = labels1.videos[0] if labels1.videos else None
        video2 = labels2.videos[0] if labels2.videos else None
        
        if video1 and video2:
            print(f"\nVideo 1: {len(video1.filename)} images")
            print(f"Video 2: {len(video2.filename)} images")
            
            # Check for duplicates
            v1_names = set(Path(f).name for f in video1.filename)
            v2_names = set(Path(f).name for f in video2.filename)
            duplicates = v1_names & v2_names
            print(f"Duplicate images: {len(duplicates)}")
            
            if duplicates:
                print("\nSample duplicates:", list(sorted(duplicates))[:5])
            
            # Test 1: IMAGE_DEDUP merging
            print("\n" + "=" * 60)
            print("Method 1: IMAGE_DEDUP - Remove duplicate images")
            print("=" * 60)
            
            labels1_dedup = coco.read_labels(cvat1, dataset_root=tmpdir)
            result = labels1_dedup.merge(labels2, video_matcher=IMAGE_DEDUP_VIDEO_MATCHER)
            
            print(f"Merge result: {result.successful}")
            print(f"Videos after merge: {len(labels1_dedup.videos)}")
            
            total_images = sum(len(v.filename) for v in labels1_dedup.videos)
            print(f"Total unique images: {total_images}")
            print(f"Frames merged: {result.frames_merged}")
            print(f"Instances added: {result.instances_added}")
            
            # Verify deduplication worked
            expected = len(v1_names) + len(v2_names) - len(duplicates)
            print(f"\nVerification:")
            print(f"  Expected unique images: {expected}")
            print(f"  Actual unique images: {total_images}")
            print(f"  Deduplication successful: {total_images == expected}")
            
            # Test 2: SHAPE merging
            print("\n" + "=" * 60)
            print("Method 2: SHAPE - Merge videos with same dimensions")
            print("=" * 60)
            
            # Check shapes
            shape1 = video1.backend_metadata.get('shape', 'Not available')
            shape2 = video2.backend_metadata.get('shape', 'Not available')
            print(f"Video 1 shape: {shape1}")
            print(f"Video 2 shape: {shape2}")
            print(f"Shapes match: {video1.matches_shape(video2)}")
            
            labels1_shape = coco.read_labels(cvat1, dataset_root=tmpdir)
            result = labels1_shape.merge(labels2, video_matcher=SHAPE_VIDEO_MATCHER)
            
            print(f"\nMerge result: {result.successful}")
            print(f"Videos after merge: {len(labels1_shape.videos)}")
            
            if labels1_shape.videos:
                merged_video = labels1_shape.videos[0]
                print(f"Merged video has {len(merged_video.filename)} images")
                
                # Check for automatic deduplication in merge_with
                unique_names = set(Path(f).name for f in merged_video.filename)
                print(f"Unique image names: {len(unique_names)}")
                print(f"Automatic deduplication: {len(unique_names) == len(merged_video.filename)}")
            
            print(f"Total frames: {len(labels1_shape.labeled_frames)}")
            print(f"Total instances: {sum(len(f.instances) for f in labels1_shape.labeled_frames)}")
            
            # Summary
            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            print("\nBoth methods successfully handle the duplicate images:")
            print("- IMAGE_DEDUP: Creates separate videos, removes duplicates from newer one")
            print("- SHAPE: Merges into single video, automatically deduplicates by filename")
            print("\nChoose based on your workflow:")
            print("- IMAGE_DEDUP: When you want to keep datasets somewhat separate")
            print("- SHAPE: When all images are from the same camera/setup")


if __name__ == "__main__":
    main()