import logging

from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType 

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType
import torch
from feature_extractor import VideoProcessor
import pandas as pd
import uuid

video_processor = VideoProcessor({
    "num_frames": 10,
    "resize": (112, 112),
    "n_mfcc": 20,
    "max_length": 20
})

@pandas_udf(ArrayType(FloatType()))
def extract_video_features(video_paths: pd.Series) -> pd.Series:
    
    results = []
    for path in video_paths:
        try:
            tensor = video_processor.process_video(path)
            flat = tensor.flatten().tolist()
        except Exception as e:
            print(e)
            flat = [0.0] * (3 * 30 * 224 * 224)
        results.append(flat)
    print(f"✅ Processing video: {video_paths}")
    return pd.Series(results)

@pandas_udf(ArrayType(FloatType()))
def extract_audio_features(video_paths: pd.Series) -> pd.Series:

    results = []
    for path in video_paths:
        try:
            tensor = video_processor.process_audio(path)
            flat = tensor.flatten().tolist()
        except Exception as e:
            print(e)
            flat = [0.0] * (40 * 40)
        results.append(flat)
    print(f"✅ Processing video: {video_paths}")
    return pd.Series(results)

# @pandas_udf(ArrayType(FloatType()))
# def extract_text_embedding(video_paths: pd.Series) -> pd.Series:
#     init_model_once({})
    
#     results = []
#     for path in video_paths:
#         try:
#             emb = video_processor.process_text(path)  # path to audio
#             flat = emb.cpu().numpy().tolist()
#         except Exception as e:
#             print(e)
#             flat = [0.0] * 768
#         results.append(flat)
#     return pd.Series(results)


def create_keyspace(session):
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS spark_streams
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'};
    """)

    print("Keyspace created successfully!")


def create_table(session):
    session.execute("DROP TABLE IF EXISTS spark_streams.created_users")
    session.execute("""
        CREATE TABLE IF NOT EXISTS spark_streams.created_users (
            id UUID PRIMARY KEY,
            url TEXT,
            label TEXT,
            video_feat LIST<FLOAT>,
            audio_feat LIST<FLOAT>,
        );
    """)
    print("Table created successfully!")


def insert_data(session, **kwargs):
    print("inserting data...")

    idx = kwargs.get('idx')
    url = kwargs.get('url')
    label = kwargs.get('label')

    try:
        session.execute("""
            INSERT INTO spark_streams.created_users(id, url, label)
                VALUES (%s, %s, %s)
        """, (idx, url, label))
        logging.info(f"Data inserted for id : {idx}")

    except Exception as e:
        logging.error(f'could not insert data due to {e}')

def create_spark_connection():
    s_conn = None
    try:    
        s_conn = SparkSession.builder \
            .appName("SparkVideoStreaming") \
            .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.4.1,"
                               "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
            .config("spark.cassandra.connection.host", "localhost") \
            .getOrCreate()

        s_conn.sparkContext.setLogLevel("ERROR")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/pipeline/feature_extractor.py")
        logging.info("Spark connection created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Spark connection: {e}")
    
    return s_conn


def create_cassandra_connection():
    try:
        # connecting to the cassandra cluster
        cluster = Cluster(['localhost'], protocol_version=5)

        cas_session = cluster.connect()

        return cas_session
    except Exception as e:
        logging.error(f"Could not create cassandra connection due to {e}")
        return None


def connect_to_kafka(spark_conn):
    spark_df = None
    try:
        spark_df = spark_conn.readStream \
            .format('kafka') \
            .option('kafka.bootstrap.servers', 'localhost:9092') \
            .option('subscribe', 'users_created') \
            .option('startingOffsets', 'earliest') \
            .option("failOnDataLoss", "false") \
            .load()
        logging.info("kafka dataframe created successfully")
    except Exception as e:
        logging.warning(f"kafka dataframe could not be created because: {e}")

    return spark_df

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType



            
def create_selection_df_from_kafka(spark_df):
    if spark_df is not None:
        schema = StructType([
            StructField("idx", StringType()),
            StructField("url", StringType()),
            StructField("label", StringType())
        ])

        selection_df = spark_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")

        

        selection_df = selection_df.withColumn("id", col("idx")) \
                                   .drop("idx")

        # 💡 Thêm các đặc trưng ngay tại đây
        selection_df = selection_df \
            .withColumn("video_feat", extract_video_features(col("url"))) \
            .withColumn("audio_feat", extract_audio_features(col("url"))) \

        logging.info("✅ Selection dataframe with features created successfully")
        return selection_df
    else:
        logging.warning("❌ No valid Kafka dataframe available")
        return None


def extract_features_udf(batch_iter):
    # model_loader = PretrainedModelLoader()
    video_processor = VideoProcessor({
        "num_frames": 30,
        "resize": (224, 224),
        "n_mfcc": 40,
        "max_length": 40
    })

    for pdf in batch_iter:
        video_feats, audio_feats, text_feats = [], [], []
        for path in pdf["url"]:
            try:
                video_tensor = video_processor.process_video(path)
                video_feats.append(video_tensor.flatten().tolist())
            except:
                video_feats.append([0.0] * (3 * 30 * 224 * 224))

            try:
                audio_tensor = video_processor.process_audio(path)
                audio_feats.append(audio_tensor.flatten().tolist())
            except:
                audio_feats.append([0.0] * (40 * 40))

            # try:
            #     text_tensor = video_processor.process_text(path)
            #     text_feats.append(text_tensor.cpu().numpy().tolist())
            # except:
            #     text_feats.append([0.0] * 768)

        pdf["video_feat"] = video_feats
        pdf["audio_feat"] = audio_feats
        # pdf["text_feat"] = text_feats
        yield pdf

def process_batch(batch_df, batch_id):
    from pyspark.sql.functions import col

    if not batch_df.rdd.isEmpty():
        # Trích xuất đặc trưng
        enriched_df = batch_df \
            .withColumn("video_feat", extract_video_features(col("url"))) \
            .withColumn("audio_feat", extract_audio_features(col("url"))) \
            # .withColumn("text_feat", extract_text_embedding(col("url")))

        # Ghi vào Cassandra
        enriched_df.write \
            .format("org.apache.spark.sql.cassandra") \
            .option("keyspace", "spark_streams") \
            .option("table", "created_users") \
            .mode("append") \
            .save()
        print(f"[Batch ID: {batch_id}] Written to Cassandra.\n")


if __name__ == "__main__":
    spark_conn = create_spark_connection()
    if spark_conn is not None:
        spark_df = connect_to_kafka(spark_conn)
        selection_df = create_selection_df_from_kafka(spark_df)

        if selection_df is not None:
            session = create_cassandra_connection()
            if session is not None:
                create_keyspace(session)
                create_table(session)

            logging.info("⚡ Streaming started with feature extraction...")

            streaming_query = (selection_df.writeStream
                               .foreachBatch(process_batch)  # ✅ Thay vì writeStream trực tiếp
                               .option("checkpointLocation", "/tmp/checkpoint-v2")
                               .start())

            streaming_query.awaitTermination()
        else:
            print("⚠️ Kafka selection_df is None. Check Kafka connection or schema.")


