import argparse
import os
import hopsworks
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from hsml.transformer import Transformer
from hsml.client.exceptions import ModelServingException
from hopsworks.engine import decision_engine_model
import sys
    
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

def load_data(de, fs):
    events_fg = fs.get_feature_group(name=de._prefix + "events")
    try:
        events_df = events_fg.read(dataframe_type="pandas")
        events_df["event_timestamp"] = (
            events_df["event_timestamp"].astype(np.int64) // 10**9
        )
    except: #TODO bad practice
        events_df = pd.DataFrame()        

    items_fg = fs.get_feature_group(
        name=de._prefix + de._configs_dict["product_list"]["feature_view_name"]
    )
    items_df = items_fg.read(dataframe_type="pandas")
    for feat, val in de._configs_dict["product_list"]["schema"].items():
        if val["type"] == "float":
            items_df[feat] = items_df[feat].astype("float32")
        if "transformation" in val.keys() and val["transformation"] == "timestamp":
            items_df[feat] = items_df[feat].astype(np.int64) // 10**9
    items_df[de._configs_dict["product_list"]["primary_key"]] = items_df[
        de._configs_dict["product_list"]["primary_key"]
    ].astype(str)

    return events_fg, events_df, items_fg, items_df

def compute_fg_size(events_fg):
    fg_stat = events_fg.statistics
    if not fg_stat:
        # no data available yet - no events received
        print(f"Couldn't retrieve events FG statistics. Quitting.")
        sys.exit()

    fg_size = next(
        column["count"]
        for column in fg_stat.to_dict()["featureDescriptiveStatistics"]
        if column["featureName"] == "event_id"
    )
    print(f"Events feature group size is {fg_size} rows")
    return fg_size


def compute_td_size(de, fs):
    events_fv = fs.get_feature_view(name=de._prefix + "events")

    td_version = max(events_fv.get_training_datasets(), key=lambda m: m.version).version
    print(f"Retrieving training dataset of version {td_version}")

    ds_size = 0  # training dataset size
    if (
        not "split_statistics"
        in events_fv.get_training_dataset_statistics(td_version).to_dict()
    ):
        print(f"Training dataset is empty. Returning 0.")
        return ds_size

    train_test_stats = [
        m.feature_descriptive_statistics
        for m in events_fv.get_training_dataset_statistics(td_version).split_statistics
    ]
    for feat_list in train_test_stats:
        for feature in feat_list:
            if feature.to_dict()["featureName"] == "event_id":
                ds_size += feature.to_dict()["count"]
    print(f"Training dataset size is {ds_size} rows")
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

def preprocess_ranking_train_df(events_df, items_df, config):
    if "policy_config" in config["model_configuration"]["ranking_model"].keys():
        items_scores = events_df.pivot_table(
            index=["session_id", "item_id"], 
            columns="event_type", 
            values="event_value",
            aggfunc='sum',
            fill_value=0
        ).reset_index()

        event_types = list(config["session"]['events'].keys())
        for event_type in event_types:
            if event_type not in items_scores.columns:
                items_scores[event_type] = 0
                
        policy_expr = config["model_configuration"]["ranking_model"]["policy_config"]
        items_scores['score'] = items_scores.eval(policy_expr)
    else:
        items_scores = events_df.groupby(['session_id', 'item_id'])['event_weight'].transform('sum') 

    events_df_unique = events_df[['item_id', 'session_id', 'longitude', 'latitude', 'language', 'useragent']].drop_duplicates()
    
    ranking_train_df = pd.merge(events_df_unique, items_scores, on=['session_id', 'item_id'], how='inner')
    ranking_train_df = pd.merge(ranking_train_df, items_df, left_on='item_id', right_on=config["product_list"]["primary_key"], how='inner')
    ranking_train_df.drop(['session_id', 'item_id'] + event_types, inplace=True, axis=1)
    return ranking_train_df

def save_model_to_registry(mr, model_name, description, tensorflow=True):
    if tensorflow:
        retrained_model = mr.tensorflow.create_model(
            name=de._prefix + model_name,
            description=description,
        )
    else:
        retrained_model = mr.python.create_model(
            name=de._prefix + model_name,
            description=description,
        )
    retrained_model.save(model_name)
    print(
        f"New {de._prefix + model_name} model version is {retrained_model.version}"
    )
    return retrained_model.version

def update_deployment(deployment, project_name, new_version, model_name):
    deployment.model_version = new_version
    deployment.artifact_version = "CREATE"
    transformer_path = os.path.join(
        "/Projects",
        project_name,
        "Resources",
        "decision-engine",
        f"{model_name}_transformer.py",
    )
    deployment.transformer = Transformer(
        script_file=transformer_path, resources={"num_instances": 1}
    )

    print(f"New deployment properties: {deployment.to_dict()}")

    try:
        deployment.save(await_update=120)
    except ModelServingException as e:
        print(f"deployment.save(await_update=120) failed. {e}")

def create_deployment(de, mr, project_name, new_version, model_name, deployment_name, description):
    mr_model = mr.get_model(name=de._prefix + model_name, version=new_version)

    transformer_script_path = os.path.join(
        "/Projects",
        project_name,
        "Resources",
        "decision-engine",
        f"{model_name}_transformer.py",
    )
    transformer = Transformer(
        script_file=transformer_script_path, resources={"num_instances": 1}
    )
    
    predictor_script_path = os.path.join(
        "/Projects",
        project_name,
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
    
def update_opensearch_index(de, items_fg, items_df, candidate_model):

    items_ds = tf.data.Dataset.from_tensor_slices(
        {col: items_df[col] for col in items_df}
    )
    item_embeddings = items_ds.batch(2048).map(
        lambda x: (
            x[de._configs_dict["product_list"]["primary_key"]],
            candidate_model(x),
        )
    )
    all_pk_list = tf.concat([batch[0] for batch in item_embeddings], axis=0).numpy().tolist()
    all_embeddings_list = tf.concat([batch[1] for batch in item_embeddings], axis=0).numpy().tolist()
    
    data_emb = pd.DataFrame({
        de._configs_dict["product_list"]["primary_key"]: all_pk_list, 
        'embeddings': all_embeddings_list,
    })
    items_fg.insert(data_emb)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help="Name of DE project", default="none")
    args = parser.parse_args()

    project = hopsworks.login()
    de_api = project.get_decision_engine_api()
    de = de_api.get_by_name(args.name)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    ms = project.get_model_serving()
    
    events_fg, events_df, items_fg, items_df = load_data(de, fs)

    ### Create Retrieval Two-Tower Model ###
    pk_index_list = items_df[de._configs_dict["product_list"]["primary_key"]].unique().tolist()
    categories_lists = {}
    text_features = {}
    for feat, val in de._configs_dict["product_list"]["schema"].items():
        if "transformation" not in val.keys():
            continue
        if val["transformation"] == "category":
            categories_lists[feat] = items_df[feat].astype(str).unique().tolist()
        elif val["transformation"] == "text":
            text_features[feat] = items_df[feat].tolist()

    new_query_model = decision_engine_model.QueryModel(
        vocabulary=pk_index_list, item_space_dim=de._configs_dict["model_configuration"]["retrieval_model"]["item_space_dim"]
    )
    new_candidate_model = decision_engine_model.ItemCatalogEmbedding(
        de._configs_dict, pk_index_list, categories_lists
    )
        
    ### Train Retrieval Two-Tower Model ###
    if not events_df.empty:
        if de._configs_dict["model_configuration"]["retrain"]["type"] == "ratio_threshold":
            fg_size = compute_fg_size(events_fg)
            td_size = compute_td_size(de, fs)

            # Comparing data sizes
            if not fg_size or (
                td_size
                and fg_size / td_size
                < de._configs_dict["model_configuration"]["retrain"]["parameter"]
            ):
                print("The new data is less than ratio threshold of the current dataset size. Quitting.")
            else:
                train_ds, val_ds = preprocess_retrieval_train_df(events_df, items_df, de._configs_dict)
                model = decision_engine_model.TwoTowerModel(new_query_model, new_candidate_model, train_ds)
                optimizer = tf.keras.optimizers.Adam(weight_decay=0.001, learning_rate=0.01)
                model.compile(optimizer=optimizer)

                update_opensearch_index(de, items_fg, items_df, model.item_model)

    ### Register Query model ###
    instances_spec = {
        "context_item_ids": tf.TensorSpec(
            shape=(None, None), dtype=tf.string, name="context_item_ids"
        ),
    }
    signatures = new_query_model.compute_emb.get_concrete_function(
        instances_spec
    )
    tf.saved_model.save(model.query_model, "query_model", signatures=signatures)
    new_query_model_v = save_model_to_registry(mr, "query_model", "Model that generates embeddings from session interaction sequence")
    
    ### Deploy Query model ###
    deployment_name = (de._prefix + "query_deployment").replace("_", "").lower()
    try:
        deployment = ms.get_deployment(deployment_name)
    except:
        create_deployment(
            de,
            mr,
            project.name,
            new_query_model_v,
            "query_model",
            deployment_name,
            "Deployment that computes query embedding from session activity and finds closest candidates"
            )
    else:
        update_deployment(
            deployment,
            project.name,
            new_query_model_v,
            "query_model",
        )
    
    ### Register Candidate model ###
    tf.saved_model.save(model.item_model, "candidate_model")
    new_candidate_model_v = save_model_to_registry(mr, "candidate_model", "Model that generates embeddings from item features")

    if not events_df.empty:
        new_ranking_model = XGBRegressor(enable_categorical=True)
        
        ### Train Ranking Model ###
        train_df = preprocess_ranking_train_df(events_df, items_df, de._configs_dict)
        feature_cols = [col for col in train_df.columns if col != 'score']
        X_train = train_df[feature_cols]
        y_train = train_df['score']
        
        for column in X_train.columns:
            if X_train[column].dtype == 'object':
                X_train[column] = X_train[column].astype('category')
        
        new_ranking_model.fit(X_train, y_train)
        
        ### Register Ranking model ###
        model_dir = "ranking_model"
        if os.path.isdir(model_dir) == False:
            os.mkdir(model_dir)
        joblib.dump(new_ranking_model, model_dir + "/ranking_model.pkl")
        new_ranking_model_v = save_model_to_registry(mr, "ranking_model", "Ranking model that scores item candidates")
        
        ### Deploy Ranking model ###
        deployment_name = (de._prefix + "ranking_deployment").replace("_", "").lower()
        try: 
            deployment = ms.get_deployment(deployment_name)
        except:
            create_deployment(
                de,
                mr,
                project.name,
                new_ranking_model_v,
                "ranking_model",
                deployment_name,
                "Deployment that scores candidates based on session context and query embedding"
            )
        else:
            update_deployment(
                deployment,
                project.name,
                new_ranking_model_v,
                "ranking_model",
            )
