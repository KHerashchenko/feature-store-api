import os
import hopsworks
import yaml
from opensearchpy import OpenSearch
import numpy as np


class Transformer(object):
    
    def __init__(self):            
        self.project = hopsworks.connection().get_project()
        dataset_api = self.project.get_dataset_api()
        de_api = self.project.get_decision_engine_api()
        decision_engine = de_api.get_by_name(os.getenv('MODEL_NAME').removeprefix('de_').removesuffix('_query_model'))
        downloaded_file_path = dataset_api.download(decision_engine._config_file_path, overwrite=True)
        
        with open(downloaded_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.prefix = "de_" + self.config["name"] + "_"
        
        self.ms = self.project.get_model_serving()
        self.ranking_deployment_name = (self.prefix + "ranking_deployment").replace("_", "").lower()
        self.os_api = self.project.get_opensearch_api()
        fs = self.project.get_feature_store()
        self.items_fg = fs.get_feature_group(
            name=self.prefix + self.config['product_list']["feature_view_name"], 
            version=1,
        )
        self.items_fv = fs.get_feature_view(
            name=self.prefix + self.config["product_list"]["feature_view_name"],
            version=1,
        )

    def preprocess(self, inputs): 
        self.inputs = inputs["instances"][0][0]

        context_item_ids = self.inputs['context_item_ids']
        if not context_item_ids: 
            pk_col = self.config['product_list']['primary_key']
            context_item_ids = self.items_fg.select([pk_col]).show(5)[pk_col].tolist()

        model_input = [{'context_item_ids': list(map(str, context_item_ids))}]
        print("model input: ", model_input)
        return model_input
    
    def postprocess(self, outputs):
        print("model output: ", outputs)
        
        self.os_client = OpenSearch(**self.os_api.get_default_py_config())
        self.item_index = self.os_api.get_project_index(
            self.config["product_list"]["feature_view_name"]
        )
        
        query_emb = outputs[0]
        self.session_id = self.inputs.pop("session_id")
        self.decision_id = self.inputs.pop("decision_id")
        self.context_item_ids = self.inputs.pop("context_item_ids")
        
        neighbors = self.items_fg.find_neighbors(
            query_emb, 
            k=100,
        )
        
        # TODO bichass workaround cause find_neighbors does not return PK or feature names
        def find_feature_index(features, feature_name):
            for index, feature in enumerate(features):
                if feature.name == feature_name:
                    return index

        pk_index = find_feature_index(self.items_fg.features, self.config['product_list']['primary_key'])
        item_id_list = [neighbor[1][pk_index] for neighbor in neighbors]
        print('neighbors:', neighbors)
        
        items_df = self.items_fv.get_feature_vectors(
            entry=[
                {self.config["product_list"]["primary_key"]: it_id}
                for it_id in item_id_list
            ],
            return_type="pandas",
        )
        items_df.drop("embeddings", axis=1, inplace=True)
        scores = {neighbor[1][pk_index]: neighbors[0] for neighbor in neighbors}
        print('scores: ', scores)
        items_df['score'] = items_df[self.config["product_list"]["primary_key"]].map(scores)
        items_df['score'].fillna(0, inplace=True)
        print('items_df: ', items_df)
        
        for feat, val in self.config["product_list"]["schema"].items():
            if val["type"] == "float":
                items_df[feat] = items_df[feat].astype("float32")
            if "transformation" in val.keys() and val["transformation"] == "timestamp":
                items_df[feat] = items_df[feat].astype(np.int64) // 10**9
        items_df[self.config["product_list"]["primary_key"]] = items_df[
            self.config["product_list"]["primary_key"]
        ].astype(str)
        self.items_data = items_df.to_dict(orient="list")

        try: 
            deployment = self.ms.get_deployment(self.ranking_deployment_name)
        except:
            print("Couldn't retrieve ranking deployment.")
            print('returning to user: ', self.items_data)
            return self.items_data
        else:
            # Return ordered ranking predictions 
            model_input = { "instances": [[{"items": self.items_data} | self.inputs]]} 
            print('ranking model input: ', model_input)      
            return deployment.predict(model_input)
        