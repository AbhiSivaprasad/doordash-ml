import pandas as pd
from .utils import set_seed


def run_training(args: TrainArgs):
    # set seed for reproducibility
    set_seed(args.seed)

    # read full dataset
    data = pd.read_csv(args.data_path)

    # create data splits
    train_data, valid_data, test_data = split_data(data)
    
    # run training
    train(args, train_data, valid_data, test_data, logger)

    # Split train/test
    rand = np.random.rand(len(data))
    train_data = df[rand < args.train_size]
    valid_data = df[rand > args.train_size and rand < args.train_size + args.valid_size]
    test_data = df[rand > args.train_size + args.valid_size]

    # pass in targets to dataset
    training_set = L1Dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = L1Dataset(test_dataset, tokenizer, MAX_LEN)
