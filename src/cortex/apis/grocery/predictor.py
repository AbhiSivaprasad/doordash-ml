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
    def __init__(self, config, python_client=None):
        # constants
        self.model_dir = "model/"
        self.text_model_dir = "text-models/"
        self.download_dir = "download/"
        self.image_dir = "images/"

        # fetch taxonomy from wandb
        wandb_api = wandb.Api({"project": "main"})
        wandb_api.artifact(config['taxonomy_wandb_identifier']).download(self.download_dir)
        taxonomy = Taxonomy().from_csv(join(self.download_dir, "taxonomy.csv"))

        # fetch models from s3
        s3_resource = boto3.resource("s3")
        model_bucket = s3_resource.Bucket(config["model_bucket"])
        download_directory_from_s3(model_bucket, config["model_bucket_folder"], self.model_dir)

        # fetch text models from s3
        model_bucket = s3_resource.Bucket(config["model_bucket"])
        download_directory_from_s3(model_bucket, config["text_model_bucket_folder"], self.text_model_dir)

        # device setup
        self.device = (torch.device("cuda") 
                       if torch.cuda.is_available() 
                       else torch.device("cpu"))
        
        # setup L1, L2 models
        l1_handler = HybridHandler.load(join(self.model_dir, 'grocery'))

        self.l1_handler, self.tokenizer \
            = l1_handler, l1_handler.tokenizer

        self.l2_handler_dict = {}  # key = class id, value = model

        for node, path in taxonomy.iter(skip_leaves=True):
            # skip l1
            if len(path) == 1:
                continue

            # load checkpoint
            category_dir = join(self.model_dir, node.category_id)
            self.l2_handler_dict[node.category_id] = HybridHandler.load(category_dir)

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
        self.image_size = 256
        self.batch_size = 1

    def predict(self, payload, query_params, headers):
        print(payload)

        data = self.process_payload(payload)

        if not pd.isnull(data["Image Name"][0]):
            text_dataset = BertDataset(data, self.tokenizer, self.max_seq_length, preserve_na=True)
            image_dataset = ImageDataset(data, self.image_dir, self.image_size, preserve_na=True, val=True)
            dataset = HybridDataset(image_dataset, text_dataset, val=True, preserve_na=True)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)

            preds, l2_confidence_scores = batch_predict(self.l1_handler, 
                                                        self.l2_handler_dict, 
                                                        dataloader, 
                                                        self.device, 
                                                        strategy="complete")

            # process predictions 
            def get_labels(l1_class_id, l2_class_id):
                l1_category = self.l1_handler.labels[l1_class_id]
                l2_category = self.l2_handler_dict[l1_category].labels[l2_class_id]
                return l1_category, l2_category

        else:
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

        # only one data point
        assert len(preds) == 1
        assert len(l2_confidence_scores) == 1

        preds = preds[0]
        l2_confidence_scores = l2_confidence_scores[0]

        preds = [get_labels(l1_pred, l2_pred) for l1_pred, l2_pred in preds]
        
        # build and return prediction results payload
        return [
            {
                "L1 Category": l1_pred,
                "L2 Category": l2_pred,
                "Confidence": confidence
            } for (l1_pred, l2_pred), confidence in zip(preds, l2_confidence_scores)
        ]

    def process_payload(self, payload):
        item_name = image_url = image_name = None
        if 'item_name' in payload and payload['item_name'] != "":
            item_name = payload['item_name'].lower()

        if 'image_url' in payload and payload['image_url'] != "":
            image_url = payload['image_url']

            # extract extension
            f = image_url.rsplit('/', 1)[-1]
            f_parts = image_url.split(".")

            if len(f_parts) == 1:
                # default extension
                file_extension = "jpeg"
            else:
                file_extension = f_parts[-1]

            # save file with extension
            image_name = f"{hash_string(image_url)}.{file_extension}"
            hash_dir = get_image_hashdir(image_name)

            # make hash dir and download image
            Path(join(self.image_dir, hash_dir)).mkdir(parents=True, exist_ok=True)
            filepath = join(self.image_dir, hash_dir, image_name) 
            download_image(image_url, filepath)

        return pd.DataFrame([[item_name, image_name]], columns=["Name", "Image Name"])


if __name__ == '__main__':
    config = {
        "model_bucket": "glisten-models",
        "model_bucket_folder": "grocery/hybrid",
        "taxonomy_wandb_identifier": "taxonomy-doordash:latest",
        "text_model_bucket_folder": "grocery/text"
    }

    payload = {
        "item_name": "chicken",
        "image_url": "https://e22d0640933e3c7f8c86-34aee0c49088be50e3ac6555f6c963fb.ssl.cf2.rackcdn.com/0052000043190_CL_default_default_thumb.jpeg"
    }

    p = PythonPredictor(config)
    print(p.predict(payload, None, None))
