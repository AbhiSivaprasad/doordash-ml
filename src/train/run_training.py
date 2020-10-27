import os, csv
import pandas as pd

from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from .utils import set_seed, load_checkpoint, DefaultLogger
from ..data.data import split_data, generate_datasets
from ..data.bert import BertDataset
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, RESULTS_FILE_NAME
from ..models.models import DistilBertClassificationModel, get_model
from .train import train
from .predict import predict
from .evaluate import evaluate_predictions


def run_training(args: TrainArgs):
    # save args
    os.makedirs(args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))

    # default logger prints
    logger = DefaultLogger()

    # read full dataset
    data = pd.read_csv(args.data_path)

    # create data splits before splitting by category
    train_data, valid_data, test_data = split_data(
        data, args.train_size, args.valid_size, args.test_size, args.seed)

    # load model specific tools
    if args.model == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased') 

    # track results tuples (model name, test accuracy)
    results = []

    # For each category generate train, valid, test
    datasets = generate_datasets(train_data, valid_data, test_data, args.categories)

    # For each dataset, create dataloaders and run training
    for dataset in datasets:
        # TODO: convert info to object
        info, data_splits = dataset
        args.num_target_classes = info['n_classes']  # save to easily reproduce model
        logger.debug("Training Model for Category:", info['name'])

        # create subdirectory for saving current model's outputs
        save_dir = os.path.join(args.save_dir, info['name'])
        os.makedirs(save_dir)

        # build model based on # of target classes
        model = get_model(args.model)(info['n_classes'])
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
        train(model, train_dataloader, valid_dataloader, valid_data.targets, args, save_dir, args.device)

        # Evaluate on test set using model with best validation score
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME))
        preds = predict(model, test_dataloader, args.device)
        test_acc = evaluate_predictions(preds, test_data.targets)
        results.append((info['name'], test_acc))
        logger.debug(f"Test Accuracy: {test_acc}")

    # Record results
    with open(os.path.join(save_dir, RESULTS_FILE_NAME), 'w+') as f:
        writer = csv.writer(f)
        headers, values = zip(*results)

        # write results
        writer.writerow(headers)
        writer.writerow(values)
