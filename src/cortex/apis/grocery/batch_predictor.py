import os
import torch
import json
import cv2
import wandb
import boto3
import pandas as pd

from src.cortex.apis.utils import download_directory_from_s3
from src.models.handlers.hybrid import HybridHandler
from src.models.handlers.huggingface import HuggingfaceHandler
from src.predict.batch_predict import batch_predict
from src.data.dataset.bert import BertDataset
from src.data.dataset.image import ImageDataset
from src.data.dataset.hybrid import HybridDataset
from src.data.image_utils import get_image_hashdir
from src.data.taxonomy import Taxonomy
from src.preprocess.image_utils import download_image
from src.preprocess.utils import hash_string

from pathlib import Path
from os.path import join, dirname
from torch.utils.data import DataLoader


class PythonPredictor:
    def __init__(self, config, job_spec):
        # constants
        self.text_model_dir = "text-models/"
        self.download_dir = "download/"

        # fetch taxonomy from wandb
        wandb_api = wandb.Api({"project": "main"})
        wandb_api.artifact(config['taxonomy_wandb_identifier']).download(self.download_dir)
        taxonomy = Taxonomy().from_csv(join(self.download_dir, "taxonomy.csv"))

        # setup s3
        self.s3 = boto3.client("s3")
        s3_resource = boto3.resource("s3")
        self.output_bucket = config["output_bucket"]

        # fetch models from s3
        model_bucket = s3_resource.Bucket(config["model_bucket"])
        download_directory_from_s3(model_bucket, config["text_model_bucket_folder"], self.text_model_dir)
        
        # job id will be key in s3 results bucket
        self.key = job_spec["job_id"]

        # device setup
        self.device = (torch.device("cuda") 
                       if torch.cuda.is_available() 
                       else torch.device("cpu"))

        # setup L1, L2 text models
        l1_text_handler = HuggingfaceHandler.load(join(self.text_model_dir, 'grocery'))

        self.l1_text_handler, self.text_tokenizer \
            = l1_text_handler, l1_text_handler.tokenizer

        self.l2_text_handler_dict = {}  # key = class id, value = model

        for node, path in taxonomy.iter(skip_leaves=True):
            # skip l1
            if len(path) == 1:
                continue

            # load checkpoint
            category_dir = join(self.text_model_dir, node.category_id)
            self.l2_text_handler_dict[node.category_id] = HuggingfaceHandler.load(category_dir)

        # args
        self.max_seq_length = 100
        self.batch_size = 128

    def predict(self, payload, batch_id):
        names = [entry["title"] for entry in payload]
        data = pd.DataFrame({"Name": names})
        dataset = BertDataset(data, self.text_tokenizer, self.max_seq_length, preserve_na=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
 
        preds, l2_confidence_scores = batch_predict(self.l1_text_handler, 
                                                    self.l2_text_handler_dict, 
                                                    dataloader, 
                                                    self.device, 
                                                    strategy="complete")
 
        # process predictions 
        def get_labels(l1_class_id, l2_class_id):
            l1_category = self.l1_text_handler.labels[l1_class_id]
            l2_category = self.l2_text_handler_dict[l1_category].labels[l2_class_id]
            return l1_category, l2_category

        preds = [[get_labels(l1_pred, l2_pred) for l1_pred, l2_pred in pred] for pred in preds]
        
        # build and return prediction results payload
        results = []
        for sample_preds, sample_confidence_scores in zip(preds, l2_confidence_scores):
            result = []
            for (l1_pred, l2_pred), confidence in zip(sample_preds, sample_confidence_scores):
                result.append({
                    "L1 Category": l1_pred,
                    "L2 Category": l2_pred,
                    "Confidence": confidence
                })

            results.append(result)

        # add to s3
        json_results = json.dumps(results).encode('utf-8')
        self.s3.put_object(Bucket=self.output_bucket, Key=f"{self.key}/{batch_id}.json", Body=json_results)
        return True

    def on_job_complete(self):
        all_results = []
 
        # aggregate all classifications
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.output_bucket, Prefix=self.key):
            for obj in page["Contents"]:
                body = self.s3.get_object(Bucket=self.output_bucket, Key=obj["Key"])["Body"]
                obj_bytes = body.read().decode("utf-8")
                all_results += json.loads(obj_bytes)
 
        # save single file containing aggregated classifications
        self.s3.put_object(
            Bucket=self.output_bucket,
            Key=os.path.join(self.key, "aggregated_results.json"),
            Body=json.dumps(all_results).encode('utf-8'),
        ) 


if __name__ == '__main__':
    config = {
        "model_bucket": "glisten-models",
        "text_model_bucket_folder": "grocery/text",
        "output_bucket": "glisten-test-batch-results",
        "taxonomy_wandb_identifier": "taxonomy-doordash:latest",
    }

    job_spec = {
        "job_id": "153", 
    }

    payload = [{
        "images": [],
        "title": "chicken",
    }]

    p = PythonPredictor(config, job_spec)
    print(p.predict(payload, "215"))
