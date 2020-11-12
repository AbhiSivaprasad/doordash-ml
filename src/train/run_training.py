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

from ..utils import set_seed, load_checkpoint, DefaultLogger, save_validation_metrics, save_checkpoint, upload_checkpoint
from ..data.data import split_data, generate_datasets
from ..data.taxonomy import Taxonomy
from ..data.bert import BertDataset
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, RESULTS_FILE_NAME
from ..models.models import get_model
from .train import train
from ..predict.predict import predict
from ..eval.evaluate import evaluate_predictions


def run_training(args: TrainArgs):
    # create logging dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # set seed for reproducibility
    set_seed(args.seed)
    
    # initialize W & B api 
    wandb_api = wandb.Api({
        "project": args.wandb_project,
    })

    # download dataset to specified dir path
    wandb_api.artifact(args.data_artifact_name).download(args.save_dir)
    data = pd.read_csv(join(args.save_dir, args.data_filename))
    
    # default logger prints
    logger = DefaultLogger()

    # taxonomy to map between category names and class ids
    taxonomy = Taxonomy().read(args.taxonomy_path)

    # read full dataset and create data splits before splitting by category
    train_data, valid_data, test_data = split_data(data, args)
    
    # For each category generate train, valid, test
    datasets = generate_datasets(train_data, valid_data, test_data, args.categories)

    # For each dataset, create dataloaders and run training
    for category_name, data_splits in datasets:
        logger.debug("Training Model for Category:", category_name)
    
        model_dir = join(args.save_dir, category_name, "model")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # initialize W&B run
        wandb_config = {
            "train_dataset_size": len(data_splits[0]),
            "num_epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "model_name": args.model_name,
            "patience": args.patience,
            "max_seq_length": args.max_seq_length,
            "cls_dropout": args.cls_dropout,
            "cls_hidden_dim": args.cls_hidden_dim
        }
        run = wandb.init(project=args.wandb_project, job_type="training", config=wandb_config, reinit=True)

        # mark dataset artifact as input to run
        run.use_artifact(args.data_artifact_name)

        # build model based on # of target classes
        num_classes = taxonomy.find_node_by_name(category_name)[0].num_children
        model, tokenizer = get_model(num_classes, args)

        # tracks model properties in W&B
        wandb.watch(model)  
        model.to(args.device)

        # pass in targets to dataset
        train_data, valid_data, test_data = [
            BertDataset(split, tokenizer, args.max_seq_length) for split in data_splits
        ]

        # pytorch data loaders
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
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
        upload_checkpoint(run, category_name, model_dir)
        logger.debug(f"Test Accuracy: {test_acc}, Loss: {test_loss}")
        wandb.summary.update({
            "test loss": test_loss,
            "test accuracy": test_acc,
        })

        # close W & B logging for run
        run.finish()
