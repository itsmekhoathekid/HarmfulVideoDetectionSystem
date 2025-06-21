import os 
import shutil
import random
from pathlib import Path

# source_path = os.path.join(source_dir, filename)
#         target_path = os.path.join(target_dir, filename)

#         # Move the file
#         shutil.move(source_path, target_path)

def moving_files(source_dir, dest_dir):
    if dest_dir is None:
        # create this destination directory if it does not exist
        pass

    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(".mp4"):
                source_path = os.path.join(dirpath, filename)
                target_path = os.path.join(dest_dir, filename)

                # Move the file
                shutil.move(source_path, target_path)
                print(f"Moved {filename} to {dest_dir}")

# train_{label}_index.mp4
# data -> train -> label -> mp4

def rename_files(source_dir):
    """
        /data/train/normal/random_index.mp4
        -> /data/train/normal/index.mp4
        -> train_normal_index.mp4

        source_dir : data directory
    """
    for split in os.listdir(source_dir):
        for label in os.listdir(os.path.join(source_dir, split)):
            i = 0 
            for filename in os.listdir(os.path.join(source_dir, split, label)):
                if filename.endswith(".mp4"):
                    old_path = os.path.join(source_dir, split, label, filename)
                    new_path = os.path.join(source_dir, split, label, f"{split}_{label}_{i}.mp4")
                    os.rename(old_path, new_path)
                    i += 1
                    print(f"Renamed {filename} to {split}_{label}_{i}.mp4")


def split_videos(
    src_folder,
    dst_root,
    train_pct=0.7,
    val_pct=0.15,
    test_pct=0.15,
    video_exts=(".mp4", ".avi", ".mov", ".mkv"),
    seed=42
):
    # Set random seed for reproducibility
    random.seed(seed)

    # Get all video files
    video_files = [
        str(f) for f in Path(src_folder).rglob("*")
        if f.suffix.lower() in video_exts
    ]

    print(f"Found {len(video_files)} videos.")

    # Shuffle videos
    random.shuffle(video_files)

    # Calculate split sizes
    n_total = len(video_files)
    n_train = int(train_pct * n_total)
    n_val = int(val_pct * n_total)
    n_test = n_total - n_train - n_val

    train_files = video_files[:n_train]
    val_files = video_files[n_train:n_train+n_val]
    test_files = video_files[n_train+n_val:]

    print(f"Splitting into: {n_train} train, {n_val} val, {n_test} test")

    # Helper to move files
    def move_files(files, split_name):
        # Thêm thư mục con "superstitious" trong mỗi split
        dst_dir = Path(dst_root) / split_name / "superstitious"
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            file_path = Path(file)
            dst_file = dst_dir / file_path.name
            shutil.move(str(file_path), str(dst_file))

    move_files(train_files, "train")
    move_files(val_files, "val")
    move_files(test_files, "test")

    print("✅ Done moving files!")

"""
{
    "idx" : "url"
}
"""


def create_json(src_folder, dst_folder):

    

    for split in os.listdir(src_folder):
        res = {}
        split_path = os.path.join(src_folder, split)
        if not os.path.isdir(split_path):
            continue
        
        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            if not os.path.isdir(label_path):
                continue
            
            for filename in os.listdir(label_path):
                if filename.endswith(".mp4"):
                    idx = filename.split(".")[0]
                    url = os.path.join(split_path, label, filename)
                    res[idx] = {}
                    res[idx]["url"] = url
                    res[idx]["label"] = label
        
        # Save to json file
        json_path = os.path.join(dst_folder, f"{split}.json")
        with open(json_path, "w") as f:
            import json
            json.dump(res, f, indent=4)
    

    print(f"Created json files in {dst_folder} for splits: {os.listdir(src_folder)}")
                    