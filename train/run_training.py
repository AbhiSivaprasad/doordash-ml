import torch
import pandas as pd

from logging import Logger
from torch.utils.data import DataLoader

def run_training(args: TrainArgs, 
                 train_data: pd.DataFrame, 
                 test_data: pd.DataFrame, 
                 logger: Logger):
    """
    :param train_data: Pandas DataFrame holding training data. Has a column 
                       'name' with item's name and 'target' with int target class
    :param test_data: Pandas DataFrame holding testing data. Has a column 
                      'name' with item's name and 'target' with int target class
    """
    # get model
    num_classes = len(set(data.target))
    model = DistilBertClassificationModel(num_classes)

    # pytorch data loaders
    train_dataloader = Dataloader(train_data, shuffle=True, batch_size=args.train_batch_size)
    valid_dataloader = Dataloader(valid_data, shuffle=True, batch_size=args.valid_batch_size)
    test_dataloader = Dataloader(test_data, shuffle=True, batch_size=args.test_batch_size)

    # simple loss function, optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # train model
    for epoch in args.epochs:
        train(model, data_loader, optimizer, loss_fn, logger)

        # test after each epoch
        preds = predict(model, valid_dataloader)

        # if model is better then save
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch

            torch.save({
                "args": args,
                "state_dict": model.state_dict(),
            }, os.path.join(args.save_dir, "model.pt"))


