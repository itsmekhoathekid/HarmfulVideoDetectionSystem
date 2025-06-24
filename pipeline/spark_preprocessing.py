import logging

from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType 

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType
import torch
from feature_extractor import PretrainedModelLoader, VideoProcessor
import pandas as pd

model_loader = None
video_processor = None

def init_model_once(configs):
    global model_loader, video_processor
    if model_loader is None:
        model_loader = PretrainedModelLoader()
        video_processor = VideoProcessor(model_loader, configs)

@pandas_udf(ArrayType(FloatType()))
def extract_video_features(video_paths: pd.Series) -> pd.Series:
    init_model_once({"num_frames": 30, "resize": (224, 224)})
    
    results = []
    for path in video_paths:
        try:
            tensor = video_processor.process_video(path)
            flat = tensor.flatten().tolist()
        except Exception as e:
            print(e)
            flat = [0.0] * (3 * 30 * 224 * 224)
        results.append(flat)
    return pd.Series(results)

@pandas_udf(ArrayType(FloatType()))
def extract_audio_features(video_paths: pd.Series) -> pd.Series:
    init_model_once({"n_mfcc": 40, "max_length": 40})

    results = []
    for path in video_paths:
        try:
            tensor = video_processor.process_audio(path)
            flat = tensor.flatten().tolist()
        except Exception as e:
            print(e)
            flat = [0.0] * (40 * 40)
        results.append(flat)
    return pd.Series(results)

@pandas_udf(ArrayType(FloatType()))
def extract_text_embedding(video_paths: pd.Series) -> pd.Series:
    init_model_once({})
    
    results = []
    for path in video_paths:
        try:
            emb = video_processor.process_text(path)  # path to audio
            flat = emb.cpu().numpy().tolist()
        except Exception as e:
            print(e)
            flat = [0.0] * 768
        results.append(flat)
    return pd.Series(results)


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
            text_feat LIST<FLOAT>
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
        cluster = Cluster(['localhost'])

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
            .load()
        logging.info("kafka dataframe created successfully")
    except Exception as e:
        logging.warning(f"kafka dataframe could not be created because: {e}")

    return spark_df

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import uuid

def create_selection_df_from_kafka(spark_df):
    if spark_df is not None:
        # Define the schema for incoming Kafka JSON
        schema = StructType([
            StructField("idx", StringType()),
            StructField("url", StringType()),
            StructField("label", StringType())
        ])

        # Parse Kafka JSON payload
        selection_df = spark_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")

        # UDF to convert idx to UUID safely
        @udf(StringType())
        def to_uuid(idx):
            try:
                if idx is None or idx.strip() == "":
                    raise ValueError("empty")
                return str(uuid.UUID(idx))  # if already valid UUID
            except:
                return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(idx)))  # generate UUID from idx

        # Add 'id' column and drop 'idx'
        selection_df = selection_df.withColumn("id", to_uuid(col("idx"))) \
                                   .drop("idx")
        
        # Optional: Print schema and preview (for debug only)
        selection_df.printSchema()
        # selection_df.writeStream.outputMode("append").format("console").start().awaitTermination()

        logging.info("✅ Selection dataframe created successfully with UUID-safe 'id'")
        return selection_df
    else:
        logging.warning("❌ No valid Kafka dataframe available to create selection dataframe")
        return None
    

def process_batch(batch_df, batch_id):
    from pyspark.sql.functions import col

    if not batch_df.rdd.isEmpty():
        # Trích xuất đặc trưng
        enriched_df = batch_df \
            .withColumn("video_feat", extract_video_features(col("url"))) \
            .withColumn("audio_feat", extract_audio_features(col("url"))) \
            .withColumn("text_feat", extract_text_embedding(col("url")))

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
        session = create_cassandra_connection()

        if session is not None:
            create_keyspace(session)
            create_table(session)

            logging.info("⚡ Streaming started with feature extraction...")

            streaming_query = (selection_df.writeStream
                               .foreachBatch(process_batch)
                               .option("checkpointLocation", "/tmp/checkpoint")
                               .start())

            streaming_query.awaitTermination()

