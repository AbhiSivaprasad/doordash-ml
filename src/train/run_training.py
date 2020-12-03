import json
import os, csv
import wandb
import torch
import pandas as pd
import torch.nn.functional as F

from os import makedirs
from os.path import join
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from time import time

from ..utils import set_seed, load_checkpoint, DefaultLogger, save_validation_metrics, save_checkpoint, upload_checkpoint
from ..data.data import split_data, encode_target_variable
from ..data.taxonomy import Taxonomy
from ..data.bert import BertDataset
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, RESULTS_FILE_NAME
from ..models.models import get_huggingface_model, get_wandb_model
from .train import train
from ..predict.predict import predict
from ..eval.evaluate import evaluate_predictions
from ..api.wandb import get_latest_artifact_identifier


def run_training(args: TrainArgs):
    # create logging dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = str(int(time()))

    # set seed for reproducibility
    set_seed(args.seed)
    
    # default logger prints
    logger = DefaultLogger()

    # initialize W & B api 
    wandb_api = wandb.Api({
        "project": args.wandb_project,
    })

    # process data sources for better logging ("artifact:latest" --> "artifact:v6")
    data_sources = [get_latest_artifact_identifier(wandb_api, source) 
                    for source in args.data_sources]

    # if all category ids specificed, then get taxonomy and iterate through categories
    category_ids = args.category_ids
    if args.all_categories:
        wandb_api.artifact(args.taxonomy_artifact_identifier).download(args.save_dir)
        taxonomy = Taxonomy.from_csv(join(args.save_dir, "taxonomy.csv"))
        category_ids = [node.category_id for node, _ in taxonomy.iter(skip_leaves=True)]

    # process datasets
    datasets = []
    for category_id in category_ids:
        # read in train dataset for category
        dataset = pd.read_csv(join(args.data_dir, "train", category_id, args.train_data_filename))

        # encode target variable and return labels dict (category id --> class id)
        labels = encode_target_variable(dataset)

        # train dataset splits into (train, val, test) 
        data_splits = split_data(dataset, args)

        # keep track of dataset, labels per category
        datasets.append((category_id, data_splits, labels))

    # For each dataset, create dataloaders and run training
    for category_id, data_splits, labels in datasets:
        logger.debug("Training Model for Category:", category_id)
    
        model_dir = join(args.save_dir, category_id, "model")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # store class id labels with model
        with open(join(model_dir, "labels.json"), 'w') as f:
            json.dump(labels, f)

        # initialize W&B run
        run_id = str(int(time()))
        wandb_config = {
            "id": run_id,
            "batch_timestamp": timestamp,
            "category_id": category_id,
            "train_dataset_size": len(data_splits[0]),
            "num_epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "model_name": args.model_name,
            "patience": args.patience,
            "max_seq_length": args.max_seq_length,
            "cls_dropout": args.cls_dropout,
            "cls_hidden_dim": args.cls_hidden_dim,
            "labels": labels,
            "train_datasets": data_sources
        }
        run = wandb.init(project=args.wandb_project, 
                         name=run_id,
                         job_type="training", 
                         config=wandb_config, 
                         reinit=True)

        # mark data sources as input to run
        for source in data_sources:
            run.use_artifact(source)

        # build model based on # of target classes
        num_classes = len(labels)

        # get model
        if args.model_source == "wandb":
            # mark model artifact as input to run 
            run.use_artifact(args.model_name)
            model, tokenizer = get_wandb_model(wandb_api, args.model_name)
        elif args.model_source == "huggingface":
            model, tokenizer = get_huggingface_model(num_classes, args)
        else:
            raise ValueError("model type not supported")

        # tracks model properties in W&B
        wandb.watch(model, log="all", log_freq=25)  
        model.to(args.device)

        # pass in targets to dataset
        train_data, valid_data, test_data = [
            BertDataset(split, tokenizer, args.max_seq_length) for split in data_splits
        ]

        # pytorch data loaders
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=args.predict_batch_size)
        test_dataloader = DataLoader(test_data, batch_size=args.predict_batch_size)

        # run training
        train(model=model, 
              tokenizer=tokenizer, 
              train_dataloader=train_dataloader, 
              valid_dataloader=valid_dataloader, 
              valid_targets=torch.from_numpy(valid_data.targets.values).to(args.device), 
              args=args, 
              save_dir=model_dir, 
              device=args.device)

        # Evaluate on test set using model with best validation score
        model, tokenizer = load_checkpoint(model_dir)

        # move model
        model.to(args.device)
        
        # predict & evaluate
        preds, probs = predict(model, test_dataloader, args.device, return_probs=True)
        test_acc = evaluate_predictions(preds, test_data.targets.values)
        test_loss = F.nll_loss(torch.log(probs), 
                               torch.from_numpy(test_data.targets.values).to(args.device))

        # Track model and results
        upload_checkpoint(run, category_id, model_dir)
        logger.debug(f"Test Accuracy: {test_acc}, Loss: {test_loss}")

        # delete unwanted parts of summary
        del wandb.summary["train accuracy"]       # accuracy of final batch doesn't say much
        del wandb.summary["train loss"]           # loss of final batch doesn't say much

        wandb.summary.update({
            "test loss": test_loss,
            "test accuracy": test_acc,
        })

        # close W & B logging for run
        run.finish()
