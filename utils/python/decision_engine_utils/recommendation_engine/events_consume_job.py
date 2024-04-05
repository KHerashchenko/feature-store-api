# The job for consuming events from Kafka topic. Runs on schedule, inserts stream into observations FG. 
# On the first run, autodetects event schema and creates "observations" FG, "training" FV and empty training dataset.

# naming conventions here are fucked tbh
import logging
import hopsworks
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType, DoubleType
from pyspark.sql.functions import from_json
from hsfs.feature import Feature

# Setting up argparse
parser = argparse.ArgumentParser()
parser.add_argument("-name", type=str, help="Name of DE project", default='none')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# Login to Hopsworks project
project = hopsworks.login()
fs = project.get_feature_store()

import yaml
dataset_api = project.get_dataset_api()
downloaded_file_path = dataset_api.download(f"Resources/decision-engine/{args.name}/configuration.yml")
with open(downloaded_file_path, "r") as f:
    config = yaml.safe_load(f)
prefix = 'de_' + config['name'] + '_'

from hsfs import engine

kafka_config = engine.get_instance()._get_kafka_config(feature_store_id=fs.id)
topic_name = '_'.join([project.name, config['name'], "events"])

logging.info("Creating Spark df")
df_read = spark \
    .readStream \
    .format("kafka") \
    .options(**kafka_config) \
    .option("startingOffsets", "latest") \
    .option("subscribe", topic_name) \
    .load()

# Define schema to parse by pyspark
struct_fields = StructType([
    StructField("event_id", LongType(), nullable=True),
    StructField("session_id", StringType(), nullable=True),
    StructField("event_timestamp", TimestampType(), nullable=True),
    StructField("item_id", StringType(), nullable=True),
    StructField("event_type", StringType(), nullable=True),
    StructField("event_value", DoubleType(), nullable=True),
    StructField("event_weight", DoubleType(), nullable=True),
    StructField("longitude", DoubleType(), nullable=True),
    StructField("latitude", DoubleType(), nullable=True),
    StructField("language", StringType(), nullable=True),
    StructField("useragent", StringType(), nullable=True)
])

parse_schema = StructType(struct_fields)

# Deserialise data from and create streaming query
df_deser = df_read.selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", struct_fields).alias("value")) \
    .select("value.event_id", "value.session_id", "value.event_timestamp", "value.item_id",
            "value.event_type", "value.event_value", "value.event_weight",
            "value.longitude", "value.latitude", "value.language", "value.useragent") \
    .selectExpr("CAST(event_id AS long)", "CAST(session_id AS string)", "CAST(event_timestamp AS timestamp)",
                "CAST(item_id AS string)", "CAST(event_type AS string)", "CAST(event_value AS double)",
                "CAST(event_weight AS double)", "CAST(longitude AS long)", "CAST(latitude AS long)",
                "CAST(language AS string)", "CAST(useragent AS string)")

events_fg = fs.get_feature_group(prefix + "events")
fg_stream_query = events_fg.insert_stream(df_deser)
