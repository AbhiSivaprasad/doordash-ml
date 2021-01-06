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

from ..utils import set_seed, DefaultLogger, upload_checkpoint
from ..data.utils import get_dataset
from ..data.data import split_data, encode_target_variable, encode_target_variable_with_labels
from ..data.taxonomy import Taxonomy
from ..data.dataset.bert import BertDataset
from ..data.dataset.image import ImageDataset
from ..data.dataset.hybrid import HybridDataset
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, RESULTS_FILE_NAME
from .train import train
from ..predict.predict import predict
from ..eval.evaluate import evaluate_predictions
from ..api.wandb import get_latest_artifact_identifier
from ..models.utils import get_hyperparams, get_model_handler, load_model


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
    train_data_sources = [get_latest_artifact_identifier(wandb_api, source) 
                          for source in args.train_data_sources]

    test_data_sources = [get_latest_artifact_identifier(wandb_api, source) 
                         for source in args.test_data_sources]

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
        dataset = pd.read_csv(join(args.train_dir, category_id, "train.csv"))

        # encode target variable and return labels dict (category id --> class id)
        labels = encode_target_variable(dataset)

        # train dataset splits into (train, val, test) 
        data_splits = split_data(dataset, args)
        
        # if separate test dir has been passed, load test data from dir
        if args.test_dir:
            test_data = pd.read_csv(join(args.test_dir, category_id, 'test.csv'))
            encode_target_variable_with_labels(test_data, labels)
            data_splits.append(test_data)

        # keep track of dataset, labels per category
        datasets.append((category_id, data_splits, labels))

    # if a model dir is specified, this is a finetuning starting from stored models
    # the versions of these models are stored, load them
    text_model_versions = vision_model_versions = None
    if args.vision_model_dir is not None:
        with open(join(args.vision_model_dir, 'versions.txt'), 'r') as f:
            vision_model_versions = json.load(f)

    if args.text_model_dir is not None:
        with open(join(args.text_model_dir, 'versions.txt'), 'r') as f:
            text_model_versions = json.load(f)

    # For each dataset, create dataloaders and run training
    for category_id, data_splits, labels in datasets:
        logger.debug("Training Model for Category:", category_id)
    
        model_dir = join(args.save_dir, category_id, "model")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # store class id labels with model
        with open(join(model_dir, "labels.json"), 'w') as f:
            json.dump(labels, f)
        
        # if a model dir is specified, this is a finetuning starting from stored models
        vision_model_path = (join(args.vision_model_dir, category_id) 
                             if args.vision_model_dir is not None else None)

        text_model_path = (join(args.text_model_dir, category_id) 
                           if args.text_model_dir is not None else None)

        # grab model and model specific hyperparams
        num_classes = len(labels)
        handler = get_model_handler(args, labels, num_classes, vision_model_path, text_model_path) 
        hyperparam_names = get_hyperparams(args.model_type)
        hyperparams = {hyperparam_name: args.__dict__[hyperparam_name] 
                       for hyperparam_name in hyperparam_names}
        # model.model = torch.nn.DataParallel(model.model)

        # initialize W&B run
        run_id = str(int(time()))
        
        # training starts from model stored at path, so is a finetuning
        finetune = vision_model_path is not None or text_model_path is not None,  
        wandb_config = {
            "id": run_id,
            "batch_id": timestamp,
            "category_id": category_id,
            "train_dataset_size": len(data_splits[0]),
            "num_epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "model_type": args.model_type,
            "model_name": args.model_name,
            "patience": args.patience,
            "labels": labels,
            "finetune": finetune,
            "train_datasets": sorted(train_data_sources),  # sort so easily queryable
            "test_datasets": sorted(test_data_sources),  # sort so easily queryable
            "separate_test_set": args.test_dir is not None  # False if test set is created by splitting train set
        }

        # add model specific hyperparameters to config
        wandb_config.update(hyperparams)

        run = wandb.init(project=args.wandb_project, 
                         name=run_id,
                         job_type="training", 
                         config=wandb_config, 
                         reinit=True)

        # mark data sources as input to run
        for source in train_data_sources + test_data_sources:
            run.use_artifact(source)

        # if finetuning from a model, mark it as input
        if text_model_versions:
            run.use_artifact(text_model_versions[category_id])
        if vision_model_versions:
            run.use_artifact(vision_model_versions[category_id])

        # tracks model properties in W&B
        wandb.watch(handler.model, log="all") 
        handler.model.to(args.device)

        # pass in targets to dataset
        train_data, valid_data, test_data = [
            get_dataset(split, args, handler) for split in data_splits
        ]

        # pytorch data loaders
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=args.predict_batch_size)
        test_dataloader = DataLoader(test_data, batch_size=args.predict_batch_size)

        # run training
        train(model=handler, 
              train_dataloader=train_dataloader, 
              valid_dataloader=valid_dataloader, 
              valid_targets=torch.from_numpy(valid_data.targets.values).to(args.device), 
              args=args, 
              save_dir=model_dir, 
              device=args.device)

        # Evaluate on test set using model with best validation score
        handler = load_model(model_dir)
        # model.model = torch.nn.DataParallel(model.model)

        # move model
        handler.model.to(args.device)
        
        # predict & evaluate
        preds, probs = predict(handler.model, test_dataloader, args.device, return_probs=True)
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
