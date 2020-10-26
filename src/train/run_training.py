import csv
import pandas as pd

from transformers import DistilBertTokenizer
from .utils import set_seed
from ../data/data import split_data, generate_datasets


def run_training(args: TrainArgs):
    # index by model name and timestamp
    args.save_dir = os.path.join(args.save_dir, 
                                 args.model_name, 
                                 datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # save args
    os.makedirs(path)
    args.save(os.path.join(args.save_dir, "args.json"))

    # read full dataset
    data = pd.read_csv(args.data_path)
    num_classes = len(set(data.target))

    # create data splits before splitting by category
    train_data, valid_data, test_data = split_data(data, args.seed)

    # load model specific tools
    if args.model == 'distilbert'
        model = DistilBertClassificationModel(num_classes)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased') 
    else:
        raise ValueError(f"Unsupported Model {args.model}")

    # track results tuples (model name, test accuracy)
    results = []

    # For each category generate train, valid, test
    datasets = generate_datasets(category, [train_data, valid_data, test_data])

    # For each dataset, create dataloaders and run training
    for dataset in datasets:
        name, data_splits = dataset
        debug("Training Model for Category:", name)

        # pass in targets to dataset
        train_data, valid_data, test_data = [
            BertDataset(split, tokenizer, args.max_seq_length) for split in data_splits
        ]

        # pytorch data loaders
        train_dataloader = Dataloader(train_data, shuffle=True, batch_size=args.train_batch_size)
        valid_dataloader = Dataloader(valid_data, shuffle=True, batch_size=args.valid_batch_size)
        test_dataloader = Dataloader(test_data, shuffle=True, batch_size=args.test_batch_size)

        # run training
        save_dir = os.path.join(args.save_dir, name)
        train(model, train_dataloader, valid_dataloader, valid_data.targets, args, save_dir)

        # Evaluate on test set using model with best validation score
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME))
        preds = predict(model, test_dataloader)
        test_acc = evaluate_predictions(preds, test_data.targets)
        results[name] = test_acc
        debug(f"Test Accuracy: {test_acc}")

    # Record results
    with open(os.path.join(save_dir, 'results.csv'), 'w+') as f:
        writer = csv.writer(f)
        headers, values = zip(*results)

        # write results
        writer.writerow(headers)
        writer.writerow(values)
