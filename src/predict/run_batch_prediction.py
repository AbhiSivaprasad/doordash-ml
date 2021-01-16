import json
import wandb
import numpy as np
import pandas as pd

from os import listdir, makedirs
from os.path import join, isdir
from torch.utils.data import DataLoader
from pathlib import Path

from ..utils import DefaultLogger
from .batch_predict import batch_predict
from .predict import predict
from ..eval.evaluate import evaluate_batch_predictions, evaluate_predictions, evaluate_lr_precision
from ..args import BatchPredictArgs
from ..data.dataset.bert import BertDataset
from ..data.taxonomy import Taxonomy
from ..models.utils import load_model_handler
from ..data.utils import get_dataset

from transformers import DistilBertTokenizer


def run_batch_prediction(args: BatchPredictArgs):
    """
    Construct hierarchy of models and run predictions.
    For now, hardcode L1, L2
    """
    logger = DefaultLogger()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)  # create logging dir

    wandb_api = wandb.Api()

    # get raw test data
    test_data = pd.read_csv(args.test_path)
    test_data["Name"] = test_data["Name"].str.lower()

    # read taxonomy
    wandb_api.artifact(args.taxonomy).download(args.save_dir)
    taxonomy = Taxonomy().from_csv(join(args.save_dir, 'taxonomy.csv'))

    # dir for L1 model
    category_dir = join(args.save_dir, 'grocery')
    Path(category_dir).mkdir()

    # download and load model
    l1_handler = load_model_handler(join(args.model_dir, 'grocery'))

    # hack to get l1 models, write an iterator when generalizing
    l2_handler_dict = {}  # key = class id, value = model
    for node, path in taxonomy.iter(skip_leaves=True):
        # skip l1
        if len(path) == 1:
            continue

        # load checkpoint
        category_dir = join(args.model_dir, node.category_id)
        l2_handler_dict[node.category_id] = load_model_handler(category_dir)

    # assumes same tokenizer used on all models
    test_data = get_dataset(test_data, args, l1_handler, val=True)
    test_dataloader = DataLoader(test_data, batch_size=args.predict_batch_size)

    # compute predictions and targets
    preds, l2_confidence_scores = batch_predict(
        l1_handler, l2_handler_dict, test_dataloader, args.device, strategy=args.strategy)

    def get_labels(l1_class_id, l2_class_id):
        l1_category = l1_handler.labels[l1_class_id]
        l2_category = l2_handler_dict[l1_category].labels[l2_class_id]
        return l1_category, l2_category

    preds = [[get_labels(l1_pred, l2_pred) for l1_pred, l2_pred in topk] for topk in preds]

    # process predictions
    df_preds = pd.DataFrame()

    for i, topk in enumerate(preds):
        row = {}
        for j, (l1_pred, l2_pred) in enumerate(topk):
            row[f"L1 #{j + 1}"] = l1_pred
            row[f"L2 #{j + 1}"] = l2_pred
            row[f"Confidence #{j + 1}"] = l2_confidence_scores[i][j]

        df_preds = df_preds.append(row, ignore_index=True)

    # reorder columns
    columns_generator = zip([f"L1 #{i + 1}" for i in range(len(preds[0]))], 
                            [f"L2 #{i + 1}" for i in range(len(preds[0]))],
                            [f"Confidence #{i + 1}" for i in range(len(preds[0]))])
    columns = [column for column_set in columns_generator for column in column_set]
    df_preds = df_preds[columns]

    # aggregate and write results
    results = pd.concat([test_data.data, df_preds], axis=1)

    results.to_csv(args.write_path, index=False)

