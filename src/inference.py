import logging

from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
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

def map_class_idx(pred_class: int):
    target_names = ["horrible", "normal", "offensive", "pornographic", "superstitious", "violent"]
    if 0 <= pred_class < len(target_names):
        return target_names[pred_class]

def process_batch(batch_df, batch_id, config):

    print(f"\n==== 🟢 BATCH ID {batch_id} | ROWS: {batch_df.count()} ====\n")
    if config['kafka']['inference']:
        batch_df = batch_df \
            .withColumn("video_feat", extract_video_features_base64(col("video_encoded"))) \
            .withColumn("audio_feat", extract_audio_features_base64(col("video_encoded")))
    else:
        batch_df = batch_df \
            .withColumn("video_feat", extract_video_features(col("url"))) \
            .withColumn("audio_feat", extract_audio_features(col("url")))
    
    # video_feat = batch_df.select("video_feat").rdd.flatMap(lambda x: x).collect()
    # audio_feat = batch_df.select("audio_feat").rdd.flatMap(lambda x: x).collect()
    # text_embedding = batch_df.select("text_embedding").rdd.flatMap(lambda x: x).collect()
    video_feat = [base64_to_tensor(v) for v in batch_df.select("video_feat").rdd.flatMap(lambda x: x).collect()]
    audio_feat = [base64_to_tensor(a) for a in batch_df.select("audio_feat").rdd.flatMap(lambda x: x).collect()]
    text_embedding = batch_df.select("text_embedding").rdd.flatMap(lambda x: x).collect()

    video_feat = torch.stack(video_feat).to(device)
    audio_feat = torch.stack(audio_feat).to(device)
    text_embedding = torch.tensor(text_embedding, dtype=torch.float).to(device)
    output = model(video_feat, audio_feat, text_embedding)
    probs = torch.softmax(output, dim=1)
    pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
    pred_classes = [map_class_idx(pred_class) for pred_class in pred_classes]
    print(f"Predicted classes: {pred_classes}")




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
                     #.trigger(processingTime="5 seconds")
                     .start())

            logging.info("✅ Streaming started with foreachBatch.")
            query.awaitTermination()