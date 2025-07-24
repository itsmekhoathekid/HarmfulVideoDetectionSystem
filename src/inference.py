import logging
import os
from datetime import datetime
from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType 

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType
import torch
from pipeline import VideoProcessor
import pandas as pd
import uuid
import os
import base64
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import gzip
from pipeline import extract_video_features_base64, extract_audio_features_base64, extract_video_features, extract_audio_features, extract_video_tensor, extract_audio_tensor
from pipeline import create_spark_connection, create_cassandra_connection, create_keyspace, create_table, connect_to_kafka, load_config, logg, create_selection_df_from_kafka, tensor_to_base64, base64_to_tensor
import argparse
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, FloatType
from models import MultimodalModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultimodalModel(num_classes=6, visual_hidden_size=256,
                        audio_hidden_size=128, text_hidden_size=128, embed_dim=256)
model.load_state_dict(torch.load("/home/anhkhoa/spark_video_streaming/checkpoint/model_checkpoint.pth"))
model.to(device)
model.eval()

# Tạo thư mục kết quả nếu chưa tồn tại
OUTPUT_DIR = "/home/anhkhoa/spark_video_streaming/checkpoint/result_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def map_class_idx(pred_class: int):
    target_names = ["horrible", "normal", "offensive", "pornographic", "superstitious", "violent"]
    if 0 <= pred_class < len(target_names):
        return target_names[pred_class]

def process_batch(batch_df, batch_id, config):
    print(f"\n==== 🟢 BATCH ID {batch_id} | ROWS: {batch_df.count()} ====\n")
    
    if batch_df.count() == 0:
        return
    
    if config['kafka']['inference']:
        batch_df = batch_df \
            .withColumn("video_feat", extract_video_features_base64(col("video_encoded"))) \
            .withColumn("audio_feat", extract_audio_features_base64(col("video_encoded")))
    else:
        batch_df = batch_df \
            .withColumn("video_feat", extract_video_features(col("url"))) \
            .withColumn("audio_feat", extract_audio_features(col("url")))
    
    # Extract features
    video_feat = [base64_to_tensor(v) for v in batch_df.select("video_feat").rdd.flatMap(lambda x: x).collect()]
    audio_feat = [base64_to_tensor(a) for a in batch_df.select("audio_feat").rdd.flatMap(lambda x: x).collect()]
    text_embedding = batch_df.select("text_embedding").rdd.flatMap(lambda x: x).collect()

    video_feat = torch.stack(video_feat).to(device)
    audio_feat = torch.stack(audio_feat).to(device)
    text_embedding = torch.tensor(text_embedding, dtype=torch.float).to(device)
    
    # Model prediction
    output = model(video_feat, audio_feat, text_embedding)
    probs = torch.softmax(output, dim=1)
    pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
    pred_classes_names = [map_class_idx(pred_class) for pred_class in pred_classes]
    
    print(f"Predicted classes: {pred_classes_names}")
    

    # Code for writing to local folder
    # Collect original data
    original_data = batch_df.select("idx", "label").collect()
    
    # Create result dataframe
    result_data = []
    for i, row in enumerate(original_data):
        result_data.append({
            "id": row["idx"],
            "label": row["label"],
            "predicted_label": pred_classes_names[i],
            "prediction_confidence": float(torch.max(probs[i]).detach().cpu().numpy()),
            "timestamp": datetime.now().isoformat()
        })
    
    # Convert to pandas DataFrame
    result_df = pd.DataFrame(result_data)
    # Save to parquet file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_batch_{batch_id}_{timestamp}.parquet"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    result_df.to_parquet(filepath, index=False)
    print(f"✅ Saved batch {batch_id} to {filepath}")
    
    # Also save to a consolidated file for easier reading
    consolidated_file = os.path.join(OUTPUT_DIR, "latest_predictions.parquet")
    if os.path.exists(consolidated_file):
        existing_df = pd.read_parquet(consolidated_file)
        combined_df = pd.concat([existing_df, result_df], ignore_index=True)
        # Keep only last 1000 records to avoid file getting too large
        combined_df = combined_df.tail(1000)
        combined_df.to_parquet(consolidated_file, index=False)
    else:
        result_df.to_parquet(consolidated_file, index=False)
    
    print(f"✅ Updated consolidated file: {consolidated_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)

    logg(config)

    spark_conn = create_spark_connection(config)
    if spark_conn is not None:
        spark_df = connect_to_kafka(spark_conn, config)
        selection_df = create_selection_df_from_kafka(spark_df, config)

        if selection_df is not None:
            session = create_cassandra_connection(config)
            if session is not None:
                create_keyspace(session, config)
                create_table(session, config)
            
            query = (selection_df.writeStream
                     .foreachBatch(lambda batch_df, batch_id: process_batch(batch_df, batch_id, config))
                     .outputMode("append")
                     .option("checkpointLocation", os.path.join(OUTPUT_DIR, "checkpoints"))
                     #.trigger(processingTime="5 seconds")
                     .start())

            logging.info("✅ Streaming started with foreachBatch.")
            query.awaitTermination()