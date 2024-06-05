import argparse
import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature import Feature

from hsfs import embedding

import tensorflow as tf

def build_feature_store(de, project):
    catalog_config = de._configs_dict["product_list"]
            
    ### Creating Items FG ###
    item_features = [
        Feature(name=feat, type=val["type"])
        for feat, val in catalog_config["schema"].items()
    ]
    item_features.append(Feature(name='embeddings', type="ARRAY<double>"))

    emb = embedding.EmbeddingIndex()
    emb.add_embedding(
        "embeddings",            
        de._configs_dict['model_configuration']['retrieval_model']['item_space_dim'], 
    )

    fs = project.get_feature_store()
    de._items_fg = fs.create_feature_group(
        name=de._prefix + catalog_config["feature_view_name"],
        description="Catalog for the Decision Engine project",
        embedding_index=emb,  
        primary_key=[catalog_config["primary_key"]],
        online_enabled=True,
        version=1,
        features=item_features,
    )
    de._items_fg.save()
    
    ### Creating Items FV ###
    items_fv = fs.create_feature_view(
        name=de._prefix + catalog_config["feature_view_name"],
        query=de._items_fg.select_all(),
        version=1,
    )

    ### Creating Events FG ###
    events_fg = fs.create_feature_group(
        name=de._prefix + "events",
        description="Events stream for the Decision Engine project",
        primary_key=["event_id"],
        online_enabled=True,
        stream=True,
        version=1,
    )

    # Enforced schema for context features in Events FG
    events_features = [
        Feature(name="event_id", type="bigint"),
        Feature(name="session_id", type="string"),
        Feature(name="event_timestamp", type="timestamp"),
        Feature(name="item_id", type="string"),
        Feature(name="event_type", type="string"),
        Feature(
            name="event_value", type="double"
        ),  # e.g. 0 or 1 for click, float for purchase (price)
        Feature(name="event_weight", type="double"),  # event_value multiplier
        Feature(
            name="longitude", type="double"
        ), 
        Feature(name="latitude", type="double"),
        Feature(name="language", type="string"),
        Feature(name="useragent", type="string"),
    ]

    events_fg.save(features=events_features)

    ### Creating Events FV ###
    events_fv = fs.create_feature_view(
        name=de._prefix + "events",
        query=events_fg.select_all(),
        version=1,
    )

    td_version, _ = events_fv.create_train_test_split(
        test_size=0.2,
        description="Models training dataset",
        write_options={"wait_for_job": True},
    )

    ### Creating Decisions FG ###
    decisions_fg = fs.create_feature_group(
        name=de._prefix + "decisions",
        description="Decisions logging for the Decision Engine project",
        primary_key=["decision_id", "session_id"],
        online_enabled=True,
        version=1,
    )

    # Enforced schema for decisions logging in Decisions FG
    decisions_features = [
        Feature(name="decision_id", type="bigint"),
        Feature(name="session_id", type="string"),
        Feature(
            name="session_activity",
            type="ARRAY <string>",
        ),  # item ids that user interacted with (all event types)
        Feature(
            name="predicted_items",
            type="ARRAY <string>",
        ),  # item ids received by getDecision
    ]
    decisions_fg.save(features=decisions_features)

def ingest_data(de, project):
    
    dataset_api = project.get_dataset_api()
    downloaded_file_path = dataset_api.download(
        de._configs_dict["product_list"]["file_path"], overwrite=True
    )
    catalog_df = pd.read_csv(
        downloaded_file_path,
        parse_dates=[
            feat
            for feat, val in de._configs_dict["product_list"]["schema"].items()
            if val["type"] == "timestamp"
        ],
    )
    
    catalog_df[de._configs_dict["product_list"]["primary_key"]] = catalog_df[
        de._configs_dict["product_list"]["primary_key"]
    ].astype(str)
    catalog_df_copy = catalog_df.copy()
    
    for feat, val in de._configs_dict["product_list"]["schema"].items():
        if val["type"] == "float":
            catalog_df[feat] = catalog_df[feat].astype("float32")
        if "transformation" in val.keys() and val["transformation"] == "timestamp":
            catalog_df[feat] = catalog_df[feat].astype(np.int64) // 10**9
        
    items_ds = tf.data.Dataset.from_tensor_slices(
        {col: catalog_df[col] for col in catalog_df}
    )
    try: 
        mr = project.get_model_registry()
        candidate_model = mr.get_model(de._prefix + "candidate_model")
    except:
        print(f"Error occured while retrieving model {de._prefix + 'candidate_model'}.")
        candidate_model = lambda x: tf.random.normal(shape=(2048, 128))
        
    item_embeddings = items_ds.batch(2048).map(
        lambda x: (x[de._configs_dict["product_list"]["primary_key"]], candidate_model(x)) 
    )
    all_embeddings_list = tf.concat([batch[1] for batch in item_embeddings], axis=0).numpy().tolist()[:catalog_df.shape[0]]

    catalog_df_copy['embeddings'] = all_embeddings_list
    de._items_fg.insert(catalog_df_copy[list(de._configs_dict["product_list"]["schema"].keys()) + ['embeddings']])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help="Name of DE project", default="none")
    args = parser.parse_args()

    project = hopsworks.login()
    de_api = project.get_decision_engine_api()
    de = de_api.get_by_name(args.name)
    
    build_feature_store(de, project)
    ingest_data(de, project)