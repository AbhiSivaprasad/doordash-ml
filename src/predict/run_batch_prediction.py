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
from ..models.utils import load_model

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
    model = load_model(join(args.model_dir, 'grocery'))
    l1_model, l1_tokenizer, l1_labels = model.model, model.tokenizer, model.labels

    # hack to get l1 models, write an iterator when generalizing
    l2_models_dict = {}  # key = class id, value = model
    l2_labels_dict = {}  # key = class id, value = labels
    for node, path in taxonomy.iter(skip_leaves=True):
        # skip l1
        if len(path) == 1:
            continue

        # load checkpoint
        category_dir = join(args.model_dir, node.category_id)
        model = load_model(category_dir)

        l2_models_dict[node.category_id] = model.model
        l2_labels_dict[node.category_id] = model.labels

    # assumes same tokenizer used on all models
    test_data = BertDataset(test_data, l1_tokenizer, args.max_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    # compute predictions and targets
    preds, l2_confidence_scores = batch_predict(
        l1_model, l1_labels, l2_models_dict, test_dataloader, args.device, strategy=args.strategy)

    def get_labels(l1_class_id, l2_class_id):
        l1_category = l1_labels[l1_class_id]
        l2_category = l2_labels_dict[l1_category][l2_class_id]
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

