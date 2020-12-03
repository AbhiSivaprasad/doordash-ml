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

    # initialize wandb run
    api = wandb.Api({"project": args.wandb_project})

    # process artifact identifier for better logging ("artifact:latest" --> "artifact:v6")
    full_eval_dataset_identifiers = [get_latest_artifact_identifier(wandb_api, source) 
                                     for source in args.eval_dataset_identifiers]

    for i, category_id in enumerate(args.category_ids):
        # dir to hold category model
        category_dir = join(model_dir, category_id)
        Path(category_dir).mkdir(parents=True)  # makes parent directories

        # models are specified with args.model_artifact_identifiers
        # or if just category ids supplied then pull the latest model for each
        model_artifact_identifier = (args.model_artifact_identifiers[i] 
                                     if args.model_artifact_identifiers 
                                     else f"model-{category_id}:latest")

        wandb_config = {
            "model_identifier": full_model_identifier,
            "eval_datasets": full_eval_dataset_identifiers,
            "category_id": category_id
        }
        run = wandb.init(project=args.wandb_project, job_type="eval", config=wandb_config, reinit=True)

        # mark data sources as input to run
        run.use_artifact(model_artifact_identifier)
        for eval_dataset_identifier in full_eval_dataset_identifiers:
            run.use_artifact(eval_dataset_identifier)

        # download and read model
        artifact = api.artifact(model_artifact_identifier).download(category_dir)
        model, tokenizer = load_checkpoint(category_dir)

        # download and read test data
        test_data = pd.read_csv(join(args.data_dir, category_id, "test.csv"))

            # encode a target variable with the given labels
        with open(join(category_dir, "labels.json")) as f:
            labels = json.load(f)

        encode_target_variable_with_labels(test_data, labels)

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

        # close W & B logging for run
        run.finish()
