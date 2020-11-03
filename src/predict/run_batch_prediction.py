import numpy as np
import pandas as pd

from os import listdir, makedirs
from os.path import join, isdir
from torch.utils.data import DataLoader
from pathlib import Path

from .utils import load_best_model, DefaultLogger, load_checkpoint
from .batch_predict import batch_predict
from .predict import predict
from ..eval.evaluate import evaluate_batch_predictions, evaluate_predictions, evaluate_lr_precision
from ..args import PredictArgs
from ..data.bert import BertDataset
from ..data.taxonomy import Taxonomy

from transformers import DistilBertTokenizer


def run_batch_prediction(args: PredictArgs):
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
    l1_model, l1_tokenizer = load_best_model(join(args.models_path, "L1"))
    l2_models_dict = {}  # key = class id, value = Model

    # hack to get l1 models, write an iterator when generalizing
    for node in taxonomy._root.children:
        path = join(args.models_path, node.category_name)
        if not isdir(path):
            continue

        model, _ = load_best_model(join(args.models_path, node.category_name))
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
    
    # evaluate precision, recall with confidence scores
    confidence_ordering = l2_confidence_scores.argsort()[::-1]  
    scores = (preds == targets).all(axis=1)
    precisions = evaluate_lr_precision(scores[confidence_ordering])

    # compute prediction accuracy
    overall_acc, l1_overall_acc, l1_class_accs = \
        evaluate_batch_predictions(preds, targets, len(taxonomy._root.children))

    # log results
    logger.debug("Overall Accuracy:", overall_acc)
    logger.debug("L1 Accuracy:", l1_overall_acc)

    logger.debug("Accuracies by Category:")
    for class_id, l1_class_acc in l1_class_accs.items():
        class_name = taxonomy.class_ids_to_category_name([class_id])
        logger.debug(f"L1 Category {class_name}:", l1_class_acc)

    # process predictions
    preds = pd.DataFrame(preds, columns=["L1_preds_class", "L2_preds_class"])
    preds["L1_preds"] = [taxonomy.class_ids_to_category_name([r.L1_preds_class])
                         for r in preds.itertuples()]
    preds["L2_preds"] = [taxonomy.class_ids_to_category_name([r.L1_preds_class, r.L2_preds_class])
                         for r in preds.itertuples()]

    # aggregate results
    results = pd.concat([pd.DataFrame({
        'Scores': scores,
        'Left precision': precisions[:, 0], 
        'Right precision': precisions[:, 1], 
        'Confidence': l2_confidence_scores, 
        'L1 Confidence': l1_confidence_scores,
        'L1_target': test_data.data["L1"],
        'L2_target': test_data.data["L2"],
    }), preds], axis=1)
    results.to_csv(join(args.save_dir, "results.csv"), index=False)
