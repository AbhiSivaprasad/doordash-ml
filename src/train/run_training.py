import os, csv
import pandas as pd

from os import makedirs
from os.path import join
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from .utils import set_seed, load_checkpoint, DefaultLogger, save_validation_metrics
from ..data.data import load_data, generate_datasets
from ..data.bert import BertDataset
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, RESULTS_FILE_NAME
from ..models.models import get_model
from .train import train
from .predict import predict
from .evaluate import evaluate_predictions


def run_training(args: TrainArgs):
    # save args
    makedirs(args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"), skip_unpicklable=True)

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
        save_dir = join(args.save_dir, info['name'])
        makedirs(save_dir)

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
              valid_targets=valid_data.targets, 
              args=args, 
              save_dir=save_dir, 
              device=args.device)

        # Evaluate on test set using model with best validation score
        model, tokenizer = load_checkpoint(save_dir)
        model.to(args.device)
        preds = predict(model, test_dataloader, args.device)
        test_acc = evaluate_predictions(preds, test_data.targets)

        # test set in training serves as performance validation
        save_validation_metrics(save_dir, test_acc)

        # Track results
        all_results.append((info['name'], test_acc))
        logger.debug(f"Test Accuracy: {test_acc}")

    # Write results
    with open(os.path.join(save_dir, RESULTS_FILE_NAME), 'w+') as f:
        writer = csv.writer(f)
        headers, values = zip(*all_results)

        # write results
        writer.writerow(headers)
        writer.writerow(values)
