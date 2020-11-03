import torch
import torch.nn.functional as F
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader

from ..utils import DefaultLogger, load_best_model, load_checkpoint
from ..data.bert import BertDataset
from ..predict.predict import predict
from ..eval.evaluate import evaluate_predictions
from ..args import PredictArgs

def run_prediction(args: PredictArgs):
    """Run predictions on a dataset"""
    # default logger prints
    logger = DefaultLogger()

    # create logging dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # get raw test data
    test_data = pd.read_csv(args.test_path)
    test_data["target"] = test_data[args.target_variable]

    # fetch model
    if args.autoload_best_model:
        # look through subdirectories for models and choose one with best validation results
        model, tokenizer = load_best_model(args.models_dir)
    else:
        # load model directly from path
        model, tokenizer = load_checkpoint(args.models_dir)

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
    
    print(preds[:20])

    # report results
    logger.debug(f"Test Accuracy: {test_acc}, Loss: {test_loss}")
