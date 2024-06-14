# naming conventions here are fucked tbh
import logging
import time
import hopsworks
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType, DoubleType
from pyspark.sql.functions import from_json  

# Setting up argparse
parser = argparse.ArgumentParser()
parser.add_argument("-name", type=str, help="Name of DE project", default='none')
parser.add_argument("-start_time", default=None)  # to be ignored
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# Login to Hopsworks project
project = hopsworks.login()
fs = project.get_feature_store()

import yaml
dataset_api = project.get_dataset_api()
de_api = project.get_decision_engine_api()
decision_engine = de_api.get_by_name(args.name)
downloaded_file_path = dataset_api.download(decision_engine._config_file_path, overwrite=True)
with open(downloaded_file_path, "r") as f:
    config = yaml.safe_load(f)
prefix = 'de_' + config['name'] + '_'

from hsfs import engine

kafka_config = engine.get_instance()._get_kafka_config(feature_store_id=fs.id)
topic_name = '_'.join([project.name, config['name'], "events"])
print(topic_name)
logging.info("Creating Spark df")
df_read = spark \
    .readStream \
    .format("kafka") \
    .options(**kafka_config) \
    .option("startingOffsets", "earliest") \
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
                "CAST(event_weight AS double)", "CAST(longitude AS double)", "CAST(latitude AS double)",
                "CAST(language AS string)", "CAST(useragent AS string)")

events_fg = fs.get_feature_group(prefix + "events")
fg_stream_query = events_fg.insert_stream(df_deser)
print(fg_stream_query.status)
time.sleep(60) 

while fg_stream_query.status['isDataAvailable']:
    print(fg_stream_query.status)
    time.sleep(60)

print("Data processing completed.")
fg_stream_query.stop()
spark.stop()
