import pandas as pd

from transformers import DistilBertTokenizer
from .utils import set_seed
from .data/data import split_data, generate_datasets


def run_training(args: TrainArgs):
    # TODO: move test here
    # TODO: data splits change numpy state
    # set seed for reproducibility
    set_seed(args.seed)

    # read full dataset
    data = pd.read_csv(args.data_path)
    num_classes = len(set(data.target))

    # create data splits
    train_data, valid_data, test_data = split_data(data)

    # load model specific tools
    if args.model == 'distilbert'
        model = DistilBertClassificationModel(num_classes)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased') 
    else:
        raise ValueError(f"Unsupported Model {args.model}")

    # For each category generate train, valid, test
    datasets = generate_datasets(category, [train_data, valid_data, test_data])

    # For each dataset, create dataloaders and run training
    for dataset in datasets:
        name, data_splits = dataset

        # pass in targets to dataset
        train_data, valid_data, test_data = [
            BertDataset(split, tokenizer, args.max_seq_length) for split in data_splits
        ]

        # pytorch data loaders
        train_dataloader = Dataloader(train_data, shuffle=True, batch_size=args.train_batch_size)
        valid_dataloader = Dataloader(valid_data, shuffle=True, batch_size=args.valid_batch_size)
        test_dataloader = Dataloader(test_data, shuffle=True, batch_size=args.test_batch_size)

        # run training
        train(args, train_dataloader, valid_dataloader, test_dataloader, logger)


