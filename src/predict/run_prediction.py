import json
import wandb
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from os.path import join
from pathlib import Path
from torch.utils.data import DataLoader

from ..utils import DefaultLogger, load_best_model, load_checkpoint
from ..data.data import encode_target_variable_with_labels
from ..data.bert import BertDataset
from ..predict.predict import predict
from ..eval.evaluate import evaluate_predictions
from ..args import PredictArgs
from ..api.wandb import get_latest_artifact_identifier


def run_prediction(args: PredictArgs):
    """Run predictions on a dataset"""
    # dirs to hold model, data
    model_dir = join(args.save_dir, "model")

    # initialize wandb run
    wandb_api = wandb.Api({"project": args.wandb_project})

    # models are specified with args.model_artifact_identifiers
    # or if just category ids supplied then pull the latest model for each
    model_artifact_identifiers = (args.model_artifact_identifiers 
                                  if args.model_artifact_identifiers 
                                  else [f"model-{category_id}:latest" for category_id in args.category_ids])

    # process artifact identifier for better logging ("artifact:latest" --> "artifact:v6")
    full_eval_dataset_identifiers = [get_latest_artifact_identifier(wandb_api, source) 
                                     for source in args.eval_datasets]

    full_model_artifact_identifiers = [get_latest_artifact_identifier(wandb_api, model_identifier) 
                                       for model_identifier in model_artifact_identifiers]

    wandb_config = {
        "model_identifiers": full_model_artifact_identifiers,
        "eval_datasets": full_eval_dataset_identifiers,
        "category_ids": args.category_ids
    }
    
    # initialize W&B run
    run = wandb.init(project=args.wandb_project, job_type="eval", config=wandb_config)

    # collect summary metrics
    test_accs = []
    test_losses = []
    predictions = []

    for category_id, model_artifact_identifier in zip(args.category_ids, full_model_artifact_identifiers):
        # dir to hold category model
        category_dir = join(model_dir, category_id)
        Path(category_dir).mkdir(parents=True)  # makes parent directories

        # mark data sources as input to run
        run.use_artifact(model_artifact_identifier)
        for eval_dataset_identifier in full_eval_dataset_identifiers:
            run.use_artifact(eval_dataset_identifier)

        # download and read model
        artifact = wandb_api.artifact(model_artifact_identifier).download(category_dir)
        model, tokenizer = load_checkpoint(category_dir)

        # download and read test data
        test_data = pd.read_csv(join(args.data_dir, category_id, "test.csv"), index_col=0)

        # encode a target variable with the given labels
        with open(join(category_dir, "labels.json")) as f:
            labels = json.load(f)

        encode_target_variable_with_labels(test_data, labels)

        # move model to GPU
        model.to(args.device)

        # assumes same tokenizer used on all models
        test_dataset = BertDataset(test_data, tokenizer, args.max_seq_length)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        # get predictions and prediction probabilities
        preds, probs = predict(model, test_dataloader, args.device, return_probs=True)

        # find probability of predicted class as confidence score
        confidence_scores = torch.max(probs, dim=1)[0].cpu().numpy()

        # add predictions to test data and track
        test_data["Pred"] = [labels[pred] for pred in preds]
        test_data["Confidence"] = confidence_scores

        # track predictions
        test_data = test_data.drop(["target"], axis=1)
        predictions.append(test_data)

        # evaluate
        mask = test_dataset.targets != -1

        test_acc = evaluate_predictions(preds[mask], test_dataset.targets[mask].values)
        test_loss = F.nll_loss(torch.log(probs[mask]), 
                               torch.from_numpy(test_dataset.targets[mask].values).to(args.device)).item()
        
       
        test_accs.append(test_acc)
        test_losses.append(test_loss)

    # store predictions
    Path(join(wandb.run.dir, "preds")).mkdir()
    for category_id, preds in zip(args.category_ids, predictions):
        preds.to_csv(join(wandb.run.dir, "preds", f"{category_id}.csv"), index=False)

    # store aggregate results
    results = pd.DataFrame({
        "Test accuracy": test_accs, 
        "Test loss": test_losses
    }, index=args.category_ids)
    
    results.to_csv(join(wandb.run.dir, "results.csv"))

    # also log results in summary
    run.summary.update({
        "f{category_id} test accuracy": test_acc 
        for category_id, test_acc in zip(args.category_ids, test_accs)
    })
    run.summary.update({
        "f{category_id} test loss": test_loss 
        for category_id, test_loss in zip(args.category_ids, test_losses)
    })
 

