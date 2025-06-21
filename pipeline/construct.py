from utils import moving_files, rename_files, split_videos, create_json

# source_dir = "/home/anhkhoa/spark_video_streaming/data"
# dest_dir = "/home/anhkhoa/spark_video_streaming/data_folder/video"

# rename_files(source_dir)
# moving_files(source_dir, dest_dir)

# src_folder = "/home/anhkhoa/spark_video_streaming/data/superstitute"
# dst_root = "/home/anhkhoa/spark_video_streaming/data"

# split_videos(
#     src_folder=src_folder,
#     dst_root=dst_root,
#     train_pct=0.7,
#     val_pct=0.15,
#     test_pct=0.15
# )


src_folder = "/home/anhkhoa/spark_video_streaming/data"
dst_root = "/home/anhkhoa/spark_video_streaming/json"

create_json(
    src_folder, dst_root
)