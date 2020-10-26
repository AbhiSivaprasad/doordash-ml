import torch
import pandas as pd

from logging import Logger
from tqdm import tqdm, trange
from typing import Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(args: TrainArgs, 
          train_data: pd.DataFrame, 
          valid_data: pd.DataFrame, 
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
    best_acc = 0
    best_epoch = 0
    for epoch in trange(args.epochs):
        train_epoch(model, data_loader, optimizer, loss_fn, logger)

        # test on validation set after each epoch
        preds = np.array(predict(model, valid_dataloader))
        targets = np.array(valid_data["target"])

        # if model is better then save
        val_acc = (preds == targets).sum() / len(targets) 
        debug(f"Validation Accuracy: {val_acc}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch

            torch.save({
                "args": args,
                "state_dict": model.state_dict(),
            }, os.path.join(args.save_dir, MODEL_FILE_NAME))
        
    # Evaluate on test set using model with best validation score
    model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME))
    preds = np.array(predict(model, test_dataloader))
    targets = np.array(test_data["target"])
    test_acc = (preds == targets).sum() / len(targets) 
    debug(f"Test Accuracy: {test_acc}")


def train_epoch(model: torch.nn.Module, 
                data_loader: DataLoader, 
                optimizer: Optimizer,
                loss_fn: Callable,
                logger: Logger = None):
    # use custom logger for training
    debug = logger.debug if logger is not None else print

    model.train()
    iter_count, total_loss, total_correct, total_steps = 0
    for batch_iter, data in tqdm(enumerate(data_loader), total=len(data_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        
        # predict and compute loss
        outputs = model(ids, mask)
        loss =  loss_fn(outputs, targets)
        _, preds = torch.max(outputs.data, dim=1)
        
        # track loss/acc
        total_correct += (preds==targets).sum().item()
        total_loss += loss.item()
        total_steps += targets.size(0)
        if batch_iter % 100 == 0:
            debug((f"Epoch: {epoch}, "
                   f"Iter: {iter_count}, "
                   f"Loss: {loss.item() / total_steps}, "
                   f"Accuracy: {(total_correct * 100) / total_steps}"))
            
            # reset stats
            total_loss, total_correct, total_steps = 0
            
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # track iterations across batches
        iter_count += targets.size(0)

    return iter_count



