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
from pipeline import extract_video_features_base64, extract_audio_features_base64, extract_video_features, extract_audio_features
from pipeline import create_spark_connection, create_cassandra_connection, create_keyspace, create_table, connect_to_kafka, load_config, logg, create_selection_df_from_kafka
import argparse
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, FloatType



        

def process_batch(batch_df, batch_id, config):
    print(f"[Batch ID: {batch_id}] Received batch with {batch_df.count()} rows")
    from pyspark.sql.functions import col

    if not batch_df.rdd.isEmpty():
        print(f"[Batch ID: {batch_id}] Row count: {batch_df.count()}")
        batch_df = batch_df \
            .withColumn("video_feat", extract_video_features(col("url"))) \
            .withColumn("audio_feat", extract_audio_features(col("url")))
        
        try:
            batch_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .option("keyspace", config['cassandra']['keyspace']) \
                .option("table", config['cassandra']['table']) \
                .mode("append") \
                .save()
            print(f"[Batch ID: {batch_id}] ✅ Written to Cassandra.\n")
        except Exception as e:
            print(f"[Batch ID: {batch_id}] ❌ Failed to write to Cassandra: {e}")
    else:
        print(f"[Batch ID: {batch_id}] ⚠️ No data to write to Cassandra.\n")




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

            logging.info("⚡ Streaming started with feature extraction...")

            streaming_query = (selection_df.writeStream
                   .foreachBatch(lambda batch_df, batch_id: process_batch(batch_df, batch_id, config))
                   .option("checkpointLocation", config['cassandra']['checkpoint'])
                #    .trigger(processingTime="5 seconds")
                   .start())


            streaming_query.awaitTermination()
        else:
            print("⚠️ Kafka selection_df is None. Check Kafka connection or schema.")


