import numpy as np
import pandas as pd

from os import listdir, makedirs
from os.path import join, isdir
from torch.utils.data import DataLoader
from pathlib import Path

from ..utils import load_best_model, DefaultLogger, load_checkpoint
from .batch_predict import batch_predict
from .predict import predict
from ..eval.evaluate import evaluate_batch_predictions, evaluate_predictions, evaluate_lr_precision
from ..args import BatchPredictArgs
from ..data.bert import BertDataset
from ..data.taxonomy import Taxonomy

from transformers import DistilBertTokenizer


def run_batch_prediction(args: BatchPredictArgs):
    """
    Construct hierarchy of models and run predictions.
    For now, hardcode L1, L2
    """
    logger = DefaultLogger()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)  # create logging dir

    # get raw test data
    test_data = pd.read_csv(args.test_path)

    # read taxonomy
    taxonomy = Taxonomy().read(args.taxonomy_dir)

    # load best models
    l1_model, l1_tokenizer = load_best_model(join(args.models_dir, "L1"))
    l2_models_dict = {}  # key = class id, value = Model

    # hack to get l1 models, write an iterator when generalizing
    for node in taxonomy._root.children:
        path = join(args.models_dir, node.category_name)
        if not isdir(path):
            continue

        model, _ = load_best_model(join(args.models_dir, node.category_name))
        l2_models_dict[node.class_id] = model

    # assumes same tokenizer used on all models
    test_data = BertDataset(test_data, l1_tokenizer, args.max_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    # compute predictions and targets
    l1_targets = test_data.data["L1_target"].values
    l2_targets = test_data.data["L2_target"].values
    targets = np.concatenate((l1_targets[:, np.newaxis], l2_targets[:, np.newaxis]), axis=1)

    preds, l1_confidence_scores, l2_confidence_scores = batch_predict(
        l1_model, l2_models_dict, test_dataloader, args.device, strategy=args.strategy)

    # process predictions
    preds = pd.DataFrame(preds, columns=["L1_preds_class", "L2_preds_class"])
    preds["L1_preds"] = [taxonomy.class_ids_to_category_name([l1_class_id])
                         for l1_class_id in preds[:,0]]
    preds["L2_preds"] = [taxonomy.class_ids_to_category_name([l1_class_id, l2_class_id])
                         for l1_class_id, l2_class_id in preds]

    # aggregate and write results
    results = pd.concat([pd.DataFrame({
        'Overall Confidence': l2_confidence_scores, 
        'L1 Confidence': l1_confidence_scores,
        'L1_target': test_data.data["L1"],
        'L2_target': test_data.data["L2"],
    }), preds], axis=1)
    results.to_csv(join(args.save_dir, "results.csv"), index=False)
