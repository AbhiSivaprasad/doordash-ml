import json
import wandb
import torch
import torch.nn.functional as F
import pandas as pd

from os.path import join
from pathlib import Path
from torch.utils.data import DataLoader

from ..utils import DefaultLogger, load_best_model, load_checkpoint
from ..data.data import encode_target_variable_with_labels
from ..data.bert import BertDataset
from ..predict.predict import predict
from ..eval.evaluate import evaluate_predictions
from ..args import PredictArgs

def run_prediction(args: PredictArgs):
    """Run predictions on a dataset"""
    # dirs to hold model, data
    model_dir = join(args.save_dir, "model")
    data_dir = join(args.save_dir, "data")
    
    # automatically makes args.save_dir
    Path(model_dir).mkdir(parents=True)
    Path(data_dir).mkdir(parents=True)

    # initialize wandb run
    api = wandb.Api({"project": args.wandb_project})
    wandb_config = {
        "model_identifier": args.model_artifact_identifier
    }
    run = wandb.init(project=args.wandb_project, job_type="eval", config=wandb_config)

    # download and read model
    artifact = api.artifact(args.model_artifact_identifier).download(model_dir)
    model, tokenizer = load_checkpoint(model_dir)

    # download and read test data
    test_data_artifact_identifier = f"dataset-{args.category_id}:latest"
    artifact = api.artifact(test_data_artifact_identifier).download(data_dir)
    test_data = pd.read_csv(join(data_dir, "test.csv"))

    # encode a target variable with the given labels
    with open(join(model_dir, "labels.json")) as f:
        labels = json.load(f)

    encode_target_variable_with_labels(test_data, labels)

    # mark used artifacts as input to run
    run.use_artifact(args.model_artifact_identifier)
    run.use_artifact(test_data_artifact_identifier)

    # move model to GPU
    model.to(args.device)

    # assumes same tokenizer used on all models
    test_data = BertDataset(test_data, tokenizer, args.max_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    # get predictions and prediction probabilities
    preds, probs = predict(model, test_dataloader, args.device, return_probs=True)

    # evaluate
    test_acc = evaluate_predictions(preds, test_data.targets.values)
    test_loss = F.nll_loss(torch.log(probs), 
                           torch.from_numpy(test_data.targets.values).to(args.device))
    
    # log results
    run.summary.update({
        "test accuracy": test_acc,
        "test loss": test_loss,
    })
    
    # add preds in file
     
