import argparse
import logging
import os
import sys
import hopsworks
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf

try:
    import tensorflow_recommenders as tfrs
except ModuleNotFoundError:
    print("module 'tensorflow-recommenders' is not installed")
    import pip
    pip.main(['install', 'tensorflow-recommenders'])
    import tensorflow_recommenders as tfrs

from opensearchpy import OpenSearch
from xgboost import XGBRegressor
import joblib

from hsml.transformer import Transformer
from hsml.client.exceptions import ModelServingException
from hopsworks.engine import decision_engine_model

logging.basicConfig(level=logging.INFO)
BATCH_SIZE = 2048


def df_to_ds(df, pad_length=10):
    features = {
        col: tf.convert_to_tensor(df[col])
        for col in df.columns
        if col != "context_item_ids"
    }
    if "context_item_ids" in df.columns:
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            df["context_item_ids"].to_list(),
            maxlen=pad_length,
            dtype="str",
            padding="post",
            truncating="post",
            value="",
        )
        padded_sequences = tf.convert_to_tensor(padded_sequences, dtype=tf.string)
        features["context_item_ids"] = padded_sequences

    return tf.data.Dataset.from_tensor_slices(features)


def login_to_project(args):
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    ms = project.get_model_serving()
    opensearch_api = project.get_opensearch_api()
    dataset_api = project.get_dataset_api()
    downloaded_file_path = dataset_api.download(
        f"Resources/decision-engine/{args.name}/configuration.yml", overwrite=True
    )
    with open(downloaded_file_path, "r") as f:
        config = yaml.safe_load(f)
    prefix = "de_" + config["name"] + "_"
    return project, fs, mr, ms, prefix, config, opensearch_api


def load_data(fs, prefix, config):
    events_fg = fs.get_feature_group(name=prefix + "events")
    events_df = events_fg.read(dataframe_type="pandas")
    events_df["event_timestamp"] = (
        events_df["event_timestamp"].astype(np.int64) // 10**9
    )

    items_fg = fs.get_feature_group(
        name=prefix + config["product_list"]["feature_view_name"]
    )
    items_df = items_fg.read(dataframe_type="pandas")
    for feat, val in config["product_list"]["schema"].items():
        if val["type"] == "float":
            items_df[feat] = items_df[feat].astype("float32")
        if "transformation" in val.keys() and val["transformation"] == "timestamp":
            items_df[feat] = items_df[feat].astype(np.int64) // 10**9
    items_df[config["product_list"]["primary_key"]] = items_df[
        config["product_list"]["primary_key"]
    ].astype(str)

    return events_fg, events_df, items_fg, items_df


def compute_fg_size(events_fg):
    try:
        fg_stat = events_fg.statistics
    except Exception as e:
        logging.info(f"Couldn't retrieve events FG statistics. Quitting. Error: {e}")
        # no data available yet - no events received
        sys.exit()

    fg_size = next(
        column["count"]
        for column in fg_stat.to_dict()["featureDescriptiveStatistics"]
        if column["featureName"] == "event_id"
    )
    logging.info(f"Events feature group size is {fg_size} rows")
    return fg_size


def compute_td_size(fs, fg_name):
    events_fv = fs.get_feature_view(name=fg_name)

    td_version = max(events_fv.get_training_datasets(), key=lambda m: m.version).version
    logging.info(f"Retrieving training dataset of version {td_version}")

    ds_size = 0  # training dataset size
    if (
        not "split_statistics"
        in events_fv.get_training_dataset_statistics(td_version).to_dict()
    ):
        logging.info(f"Training dataset is empty. Returning 0.")
        return ds_size

    train_test_stats = [
        m.feature_descriptive_statistics
        for m in events_fv.get_training_dataset_statistics(td_version).split_statistics
    ]
    for feat_list in train_test_stats:
        for feature in feat_list:
            if feature.to_dict()["featureName"] == "event_id":
                ds_size += feature.to_dict()["count"]
    logging.info(f"Training dataset size is {ds_size} rows")
    return ds_size


def preprocess_retrieval_train_df(events_df, items_df, config):
    # Count occurrences of each session_id
    session_counts = events_df.groupby("session_id").size().reset_index(name="count")

    # Filter sessions with more than 5 interactions
    multi_interaction_sessions = session_counts[session_counts["count"] > 5]

    # Join with the original events dataframe
    filtered_df = pd.merge(
        events_df,
        multi_interaction_sessions[["session_id"]],
        on="session_id",
        how="inner",
    )

    # Aggregate unique item_ids per session
    aggregated_df = (
        filtered_df.groupby("session_id")
        .agg(context_item_id=("item_id", lambda x: list(set(x))[:10]))
        .reset_index()
    )

    aggregated_df["context_item_ids"] = aggregated_df["context_item_id"].copy()
    aggregated_df_exploded = aggregated_df.explode("context_item_id")

    # Join with items dataframe
    primary_key = config["product_list"]["primary_key"]
    retrieval_train_df = pd.merge(
        aggregated_df_exploded,
        items_df,
        left_on="context_item_id",
        right_on=primary_key,
        how="inner",
    )

    # Split data into train and validation sets
    train_df, val_df = train_test_split(
        retrieval_train_df, test_size=0.2, random_state=123
    )

    # Convert DataFrames to TensorFlow datasets
    print("train_df: ", train_df.dtypes)
    train_ds = df_to_ds(train_df).batch(BATCH_SIZE).cache().shuffle(BATCH_SIZE * 10)
    print("train_ds: ", train_ds)
    val_ds = df_to_ds(val_df).batch(BATCH_SIZE).cache()

    return train_ds, val_ds


class TwoTowerModel(tf.keras.Model):
    def __init__(self, query_model, item_model, item_ds):
        super().__init__()
        self.query_model = query_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_ds.batch(BATCH_SIZE).map(self.item_model)
            )
        )

    def train_step(self, batch) -> tf.Tensor:
        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:
            # Loss computation.
            item_embeddings = self.item_model(batch)
            user_embeddings = self.query_model.compute_emb(batch)['query_emb']
            loss = self.task(
                user_embeddings,
                item_embeddings,
                compute_metrics=False,
            )

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {
            "loss": loss,
            "regularization_loss": regularization_loss,
            "total_loss": total_loss,
        }

        return metrics

    def test_step(self, batch) -> tf.Tensor:
        # Loss computation.
        item_embeddings = self.item_model(batch)
        user_embeddings = self.query_model(batch)['query_emb']

        loss = self.task(
            user_embeddings,
            item_embeddings,
            compute_metrics=False,
        )

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


def train_retrieval_model(catalog_df, train_ds, val_ds, configs_dict):
    catalog_config = configs_dict["product_list"]
    retrieval_config = configs_dict["model_configuration"]["retrieval_model"]

    pk_index_list = catalog_df[catalog_config["primary_key"]].unique().tolist()
    categories_lists = {}
    text_features = {}
    for feat, val in catalog_config["schema"].items():
        if "transformation" not in val.keys():
            continue
        if val["transformation"] == "category":
            categories_lists[feat] = catalog_df[feat].astype(str).unique().tolist()
        elif val["transformation"] == "text":
            text_features[feat] = catalog_df[feat].tolist()

    new_query_model = decision_engine_model.QueryModel(
        vocabulary=pk_index_list, item_space_dim=retrieval_config["item_space_dim"]
    )
    new_candidate_model = decision_engine_model.ItemCatalogEmbedding(
        configs_dict, pk_index_list, categories_lists
    )

    model = TwoTowerModel(new_query_model, new_candidate_model, train_ds)
    # Define an optimizer using AdamW with a learning rate of 0.01
    optimizer = tf.keras.optimizers.Adam(weight_decay=0.001, learning_rate=0.01)
    # Compile the model using the specified optimizer
    model.compile(optimizer=optimizer)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
    )
    
    instances_spec = {
        "context_item_ids": tf.TensorSpec(
            shape=(None, None), dtype=tf.string, name="context_item_ids"
        ),
    }

    signatures = query_model.compute_emb.get_concrete_function(
        instances_spec
    )
    tf.saved_model.save(model.query_model, "query_model", signatures=signatures)
    tf.saved_model.save(model.item_model, "candidate_model")
    return new_query_model, new_candidate_model


def train_ranking_model(train_df):
    new_model = XGBRegressor(enable_categorical=True)
    feature_cols = [col for col in train_df.columns if col != 'score']
    X_train = train_df[feature_cols]
    y_train = train_df['score']
    
    for column in X_train.columns:
        if X_train[column].dtype == 'object':
            X_train[column] = X_train[column].astype('category')
    
    new_model.fit(X_train, y_train)
    
    model_dir = "ranking_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    joblib.dump(new_model, model_dir + "/ranking_model.pkl")
    return new_model


def preprocess_ranking_train_df(events_df, items_df, config):
    if "policy_config" in config["model_configuration"]["ranking_model"].keys():
        items_scores = events_df.pivot_table(
            index=["session_id", "item_id"], 
            columns="event_type", 
            values="event_value",
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        event_types = ["click", "add_to_cart", "purchase"] # TODO extract from config file
        for event_type in event_types:
            if event_type not in items_scores.columns:
                items_scores[event_type] = 0 if event_type != 'purchase' else 0.1 # TODO hardcode because expression has log()
                
        policy_expr = config["model_configuration"]["ranking_model"]["policy_config"]
        items_scores['score'] = items_scores.eval(policy_expr)
    else:
        pass
#         items_scores = events_df.groupby(['session_id', 'item_id'])['event_weight'].transform('sum') TODO fix this case handling

    events_df_unique = events_df[['item_id', 'session_id', 'longitude', 'latitude', 'language', 'useragent']].drop_duplicates()
    
    ranking_train_df = pd.merge(events_df_unique, items_scores, on=['session_id', 'item_id'], how='inner')
    ranking_train_df = pd.merge(ranking_train_df, items_df, left_on='item_id', right_on=config["product_list"]["primary_key"], how='inner')
    ranking_train_df.drop(['session_id', 'item_id'] + event_types, inplace=True, axis=1)
    return ranking_train_df


def save_model_to_registry(mr, prefix, model_name, description):
    retrained_model = mr.tensorflow.create_model(
        name=prefix + model_name,
        description=description,
        #         model_schema=current_model.model_schema, TODO: must be ModelSchema object, receiving dict
    )
    retrained_model.save(model_name)
    logging.info(
        f"New {prefix + model_name} model version is {retrained_model.version}"
    )
    return retrained_model.version


def update_deployment(deployment, project, new_version, model_name):
    deployment.model_version = new_version
    deployment.artifact_version = "CREATE"
    transformer_path = os.path.join(
        "/Projects",
        project.name,
        "Resources",
        "decision-engine",
        f"{model_name}_transformer.py",
    )
    deployment.transformer = Transformer(
        script_file=transformer_path, resources={"num_instances": 1}
    )

    logging.info(f"New deployment properties: {deployment.to_dict()}")

    try:
        deployment.save(await_update=120)
    except ModelServingException as e:
        logging.info(f"deployment.save(await_update=120) failed. {e}")


def create_deployment(mr, project, new_version, prefix, model_name, deployment_name, description):
    mr_model = mr.get_model(name=prefix + model_name, version=new_version)

    transformer_script_path = os.path.join(
        "/Projects",
        project.name,
        "Resources",
        "decision-engine",
        f"{model_name}_transformer.py",
    )
    transformer = Transformer(
        script_file=transformer_script_path, resources={"num_instances": 1}
    )
    
    predictor_script_path = os.path.join(
        "/Projects",
        project.name,
        "Resources",
        "decision-engine",
        f"{model_name}_predictor.py",
    )
    deployment = mr_model.deploy(
        name=deployment_name,
        description=description,
        resources={"num_instances": 1},
        transformer=transformer,
        script_file=predictor_script_path,
    )


def update_opensearch_index(items_df, opensearch_api, config, candidate_model, prefix):
    client = OpenSearch(**opensearch_api.get_default_py_config())
    
    index_name = opensearch_api.get_project_index(
        config["product_list"]["feature_view_name"]
    )

    items_ds = tf.data.Dataset.from_tensor_slices(
        {col: items_df[col] for col in items_df}
    )
    item_embeddings = items_ds.batch(2048).map(
        lambda x: (
            x[config["product_list"]["primary_key"]],
            candidate_model(x),
        )
    )

    actions = []
    for batch in item_embeddings:
        item_id_list, embedding_list = batch
        item_id_list = item_id_list.numpy().astype(int)
        embedding_list = embedding_list.numpy()
        
        for item_id, embedding in zip(item_id_list, embedding_list):
        
            actions.append({"update": { "_index": index_name, "_id": int(item_id)}})
            actions.append({'doc': {prefix + "vector": embedding.tolist()}})

    print(f"Example item vectors to be bulked: {actions[:10]}")
    client.bulk(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help="Name of DE project", default="none")
    args = parser.parse_args()

    project, fs, mr, ms, prefix, config, opensearch_api = login_to_project(args)
    events_fg, events_df, items_fg, items_df = load_data(fs, prefix, config)

    if config["model_configuration"]["retrain"]["type"] == "ratio_threshold":
        fg_size = compute_fg_size(events_fg)
        td_size = compute_td_size(fs, prefix + "events")

        # Comparing data
        if not fg_size or (
            td_size
            and fg_size / td_size
            < config["model_configuration"]["retrain"]["parameter"]
        ):
            logging.info(
                "The new data is less than ratio threshold of the current dataset size. Quitting."
            )
            sys.exit()

    # Retrieval model
    train_ds, val_ds = preprocess_retrieval_train_df(events_df, items_df, config)
    query_model, candidate_model = train_retrieval_model(
        items_df, train_ds, val_ds, config
    )
    current_model = mr.get_model(prefix + "query_model")

    new_version = save_model_to_registry(mr, prefix, "query_model", current_model.description)
    deployment_name = (
    (prefix + 'query_model').replace("_", "").lower().replace("model", "deployment")
    )
    deployment = ms.get_deployment(deployment_name)
    update_deployment(
        deployment,
        project,
        new_version,
        "query_model",
    )
    
    current_model = mr.get_model(prefix + "query_model")
    new_version = save_model_to_registry(mr, prefix, "candidate_model", current_model.description)
    update_opensearch_index(items_df, opensearch_api, config, candidate_model, prefix)

    # Ranking model
    train_ds = preprocess_ranking_train_df(events_df, items_df, config)
    ranking_model = train_ranking_model(train_ds)
    retrained_model = mr.python.create_model(
        name=prefix + 'ranking_model',
        description="Ranking model that scores item candidates")
    retrained_model.save('ranking_model')
    logging.info(
        f"New {prefix + 'ranking_model'} model version is {retrained_model.version}"
    )
    new_version = retrained_model.version
    deployment_name = (
    (prefix + 'ranking_model').replace("_", "").lower().replace("model", "deployment")
    )
    try: 
        deployment = ms.get_deployment(deployment_name)
    except:
        create_deployment(
            mr,
            project,
            new_version,
            prefix,
            "ranking_model",
            deployment_name,
            "Deployment that scores candidates based on session context and query embedding"
        )
    else:
        update_deployment(
            deployment,
            project,
            new_version,
            "ranking_model",
        )
