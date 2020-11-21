import argparse
import pickle
from tqdm import tqdm
import configparser
import os
import random
import shutil
import time
import warnings

import wandb
from PIL import Image, ImageOps, ImageFile
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import time

from ..models import ResnetModel
from ..data import  ImageDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

hyperparams = None 

best_acc = 0

def get_latest_model(group):
    f = os.path.join(args.save_dir, args.group) 
    models = sorted(os.listdir(f))
    latest = f + models[len(models) - 1]
    print("Latest model:", latest)
    return latest


def main():
    ngpus_per_node = torch.cuda.device_count()
    hyperparams["lr"] = args.lr
    hyperparams["image_size"] = args.image_size
    hyperparams["group"] = args.group
    hyperparams["epochs"] = args.epochs
    hyperparams["num_classes"] = len(os.listdir(os.path.join(args.train_directory, args.group)))

    if args.pretrained:
        hyperparams["pretrained"] = True
    if args.pad:
        hyperparams["pad"] = True

    if args.eval:
        if args.checkpoint is None:
            checkpoint =  get_latest_model(args.group)
        else:
            checkpoint = os.path.join(args.save_dir, args.group, args.checkpoint + "_checkpoint.pth.tar")
        print("Using checkpoint:", checkpoint)
        run_evaluation(args.group, hyperparams["num_classes"], checkpoint)

    else:

        # Setup model training tracking (wandb)
        wandb.init(project=args.wandb_projects, name=args.id)
        wandb.config.update(hyperparams)

        # Run training
        run_training(args.group, hyperparams["num_classes"], args.resume)


def run_evaluation(group_name, num_classes, checkpoint=None):
    arch = hyperparams["architecture"]
    pretrained = hyperparams["pretrained"]
    lr = hyperparams["learning_rate"]
    momentum = hyperparams["momentum"]
    weight_decay = hyperparams["weight_decay"]
    batch_size = hyperparams["batch_size"]
    workers = hyperparams["workers"]
    epochs = hyperparams["epochs"]
    num_classes = num_classes

    model = Model(group_name, num_classes, pretrained=pretrained, eval=True)
    model.load_model(checkpoint)
    optimizer = model.optimizer

    #class_weights = torch.FloatTensor([100.0, 1.0]).cuda()
    #criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(None)
    criterion = nn.CrossEntropyLoss().cuda(None)
    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.validation_directory, args.group)
    val_loader, val_dataset = create_dataloader(valdir, val=True)
    print("done creating data loader", len(val_dataset))

    y_true = []
    for i in tqdm(range(len(val_dataset))):
        y_true.append(val_dataset[i][1])
    print("=> finished creating image folders")

    avg_acc, avg_loss, output = validate(val_loader, model.model, criterion)

    #y_true = [img[1] for img in val_dataset.imgs]
    y_pred = [o.index(max(o)) for o in output]

    print(confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred))

    correct = 0
    #for i, image in enumerate(val_dataset.imgs):
    for i, image in enumerate(val_dataset):
        #print(image, output[i])
        o = output[i].index(max(output[i]))
        if o == image[1]:
            correct += 1
    #print("accuracy", float(correct)/len(val_dataset.imgs))
    print("accuracy", float(correct)/len(val_dataset))
    return val_dataset.imgs, output

def run_training(group_name, num_classes, resume=False):
    global best_acc

    arch = hyperparams["architecture"]
    pretrained = hyperparams["pretrained"]
    lr = hyperparams["learning_rate"]
    momentum = hyperparams["momentum"]
    weight_decay = hyperparams["weight_decay"]
    batch_size = hyperparams["batch_size"]
    workers = hyperparams["workers"]
    epochs = hyperparams["epochs"]

    model = Model(group_name, num_classes, pretrained=pretrained)

    if resume:
        latest_model = get_latest_model(group_name)
        model.load_model(latest_model, cpu=False)

    wandb.watch(model.model)

    # define loss function (criterion) 
    #class_weights = torch.FloatTensor([100.0, 1.0]).cuda()
    #criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(None)
    criterion = nn.CrossEntropyLoss().cuda(None)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.train_directory, args.group)
    valdir = os.path.join(args.validation_directory, args.group)

    # Store labels
    wandb.config.update({"labels": os.listdir(traindir)})

    train_loader, train_dataset = create_dataloader(traindir, shuffle=True)
    val_loader, val_dataset = create_dataloader(valdir, val=True)

    for epoch in range(0, epochs):
        adjust_learning_rate(model.optimizer, epoch)

        # train for one epoch
        train(train_loader, model.model, criterion, model.optimizer, epoch)

        # evaluate on validation set
        acc,loss,_ = validate(val_loader, model.model, criterion)

        wandb.log({"Test Accuracy": acc, "Test Loss": loss})


        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint(group_name, {
           'epoch': epoch + 1,
           'arch': hyperparams["architecture"],
           'state_dict': model.model.state_dict(),
           'best_acc': best_acc,
           'optimizer' : model.optimizer.state_dict(),
           'labels': os.listdir(traindir) # foldernames for each class
        }, is_best)
    print("Best accuracy", float(best_acc))

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def create_dataloader(directory, shuffle=False, val=False):

    print("create image folder")
    root = directory
    #dataset = datasets.ImageFolder(directory, trans)
    dataset = None #ImageDataset(root, "dresses", 0)

    classes = sorted(os.listdir(root))
    for i in range(len(classes)):
        print("loaded", classes[i])
        if dataset is None:
            dataset = ImageDataset(root, classes[i], i, args.image_size, val)
        else:
            dataset += ImageDataset(root, classes[i], i, args.image_size, val)
	#+ ImageDataset(root, "shoes", 2)
    

    print("done creating image folder")

    if val:
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=hyperparams["batch_size"], shuffle=False,
            num_workers=hyperparams["workers"], pin_memory=True), dataset


    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=hyperparams["batch_size"], shuffle=shuffle,
        num_workers=hyperparams["workers"], pin_memory=True), dataset


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(None, non_blocking=True)
        target = target.cuda(None, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc_value = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        acc.update(acc_value[0], images.size(0))
            
        wandb.log({"Train Acc": float(acc_value), "Train Loss": loss.item()})

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc],
        prefix='Test: ')

    all_output = []

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(None, non_blocking=True)
            target = target.cuda(None, non_blocking=True)

            # compute output
            output = model(images)
            all_output += output.tolist()
            loss = criterion(output, target)

            # measure accuracy and record loss
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc_val = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            acc.update(acc_val[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {acc.avg:.3f}'
              .format(acc=acc))

    return acc.avg, losses.avg, all_output
    #return acc.avg


def save_checkpoint(group, state, is_best):

    filename = hyperparams["group"] + '_checkpoint.pth.tar'

    if not os.path.exists('models/' + hyperparams['group']):
        os.mkdir('models/' + hyperparams['group'])

    best_filename = 'models/' + hyperparams["group"] + "/" + hyperparams["id"] + '_checkpoint.pth.tar'
    torch.save(state, filename)

    if is_best:
        print("New best checkpoint", best_filename)
        shutil.copyfile(filename, best_filename)
        torch.save(state, os.path.join(wandb.run.dir, os.path.basename(best_filename)))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = hyperparams["learning_rate"]
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct = correct.view(-1).float().sum(0, keepdim=True)
        return correct.mul_(100.0 / batch_size)

def run_training(arguments):
    global hyperparams
    global args
    args = arguments
    hyperparams = dict(arguments.__dict__)
    logger.debug("hello")
    logger.debug(hyperparams)
    main()

if __name__ == '__main__':
    main()
