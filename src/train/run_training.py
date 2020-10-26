import pandas as pd
from ./utils import set_seed

def run_training(args: TrainArgs):
    # set seed for reproducibility
    set_seed(args.seed)

    # read full dataset
    data = pd.read_csv(args.data_path)

    # create data splits
    train_data, valid_data, test_data = split_data(data)
    
    train(args, train_data, valid_data, test_data, logger)
