from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
import torch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from .feature_extractor import VideoProcessor
from .utils import tensor_to_base64, base64_to_tensor

video_processor = VideoProcessor({
    "num_frames": 30,
    "resize": (224, 224),
    "n_mfcc": 40,
    "max_length": 40
})

@pandas_udf(StringType())
def extract_video_features_base64(video_strings: pd.Series) -> pd.Series:
    results = []
    count = 0
    for video_string in video_strings:
        try:
            tensor = video_processor.process_video_base64(video_string)
        except Exception as e:
            print(f"❌ Video error: {e}")
            tensor = torch.zeros((30, 3, 224, 224))
        results.append(tensor_to_base64(tensor))
        count += 1
        print(f"[Video processing] {count}/{len(video_strings)}")
    
    return pd.Series(results)

@pandas_udf(StringType())
def extract_audio_features_base64(video_strings: pd.Series) -> pd.Series:
    results = []
    count = 0
    for video_string in video_strings:
        try:
            tensor = video_processor.process_audio_base64(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        results.append(tensor_to_base64(tensor))
        count += 1
        print(f"[Audio processing] {count}/{len(video_strings)}")
    return pd.Series(results)


@pandas_udf(StringType())
def extract_video_features(video_paths: pd.Series) -> pd.Series:
    results = []
    count = 0
    for path in video_paths:
        try:
            tensor = video_processor.process_video(path)
        except Exception as e:
            print(f"❌ Video error: {e}")
            tensor = torch.zeros((30, 3, 224, 224))
        results.append(tensor_to_base64(tensor))
        count += 1
        print(f"[Video processing] {count}/{len(video_paths)} videos at path {path}")
    
    return pd.Series(results)

@pandas_udf(StringType())
def extract_audio_features(video_strings: pd.Series) -> pd.Series:
    results = []
    count = 0 
    for video_string in video_strings:
        try:
            tensor = video_processor.process_audio(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        results.append(tensor_to_base64(tensor))
        count += 1
        print(f"[Video audio] {count}/{len(video_strings)} audio ")
    return pd.Series(results)



@pandas_udf(StringType())
def extract_audio_features_base64_parallel(video_strings: pd.Series) -> pd.Series:
    def process_one(video_string):
        try:
            tensor = video_processor.process_audio_base64(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        return tensor_to_base64(tensor)

    with ThreadPoolExecutor(max_workers=4) as executor:  # bạn có thể tăng lên tùy CPU
        results = list(executor.map(process_one, video_strings.tolist()))

    return pd.Series(results)

@pandas_udf(StringType())
def extract_video_features_base64_parallel(video_strings: pd.Series) -> pd.Series:
    def process_one(video_string):
        try:
            tensor = video_processor.process_video_base64(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        return tensor_to_base64(tensor)

    with ThreadPoolExecutor(max_workers=4) as executor:  # bạn có thể tăng lên tùy CPU
        results = list(executor.map(process_one, video_strings.tolist()))

    return pd.Series(results)


