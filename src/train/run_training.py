import os, csv
import torch
import pandas as pd
import torch.nn.functional as F

from os import makedirs
from os.path import join
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from .utils import set_seed, load_checkpoint, DefaultLogger, save_validation_metrics, save_checkpoint
from ..data.data import load_data, generate_datasets
from ..data.bert import BertDataset
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, RESULTS_FILE_NAME
from ..models.models import get_model
from .train import train
from .predict import predict
from .evaluate import evaluate_predictions


def run_training(args: TrainArgs):
    # create logging dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # default logger prints
    logger = DefaultLogger()

    # read full dataset and create data splits before splitting by category
    train_data, valid_data, test_data = load_data(args)
    
    # For each category generate train, valid, test
    datasets = generate_datasets(train_data, valid_data, test_data, args.categories)

    # For each dataset, create dataloaders and run training
    all_results = []  # tuples (model name, test accuracy)
    for dataset in datasets:
        # TODO: convert info to object
        info, data_splits = dataset
        logger.debug("Training Model for Category:", info['name'])

        # create subdirectory for saving current model's outputs
        save_dir = join(args.save_dir, info['name'], datetime.now().strftime("%Y%m%d-%H%M%S"))
        makedirs(save_dir)
        args.save(join(save_dir, "args.json"), skip_unpicklable=True)

        # build model based on # of target classes
        model, tokenizer = get_model(info['n_classes'], args)
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
              save_dir=save_dir, 
              device=args.device)

        # Evaluate on test set using model with best validation score
        model, tokenizer = load_checkpoint(save_dir)

        # move model
        model.to(args.device)
        
        # predict & evaluate
        preds, probs = predict(model, test_dataloader, args.device, return_probs=True)
        test_acc = evaluate_predictions(preds, test_data.targets.values)
        test_loss = F.nll_loss(torch.log(probs), 
                               torch.from_numpy(test_data.targets.values).to(args.device))

        # test set in training serves as performance validation
        save_validation_metrics(save_dir, test_acc, test_loss)

        # Track results
        all_results.append((info['name'], test_acc))
        logger.debug(f"Test Accuracy: {test_acc}, Loss: {test_loss}")

    # Write results
    with open(os.path.join(save_dir, RESULTS_FILE_NAME), 'w+') as f:
        writer = csv.writer(f)
        headers, values = zip(*all_results)

        # write results
        writer.writerow(headers)
        writer.writerow(values)
