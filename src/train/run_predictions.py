import pandas as pd

from os import listdir
from os.path import join, isdir
from torch.utils.data import DataLoader

from .utils import load_best_model, DefaultLogger, load_checkpoint
from .batch_predict import batch_predict
from .predict import predict
from .evaluate import evaluate_batch_predictions, evaluate_predictions
from ..args import PredictArgs
from ..data.bert import BertDataset
from ..data.taxonomy import Taxonomy

from transformers import DistilBertTokenizer


def run_predictions(args: PredictArgs):
    """
    Construct hierarchy of models and run predictions
    """
    logger = DefaultLogger()

    # get raw test data
    test_data = pd.read_csv(args.test_path)

    # read taxonomy
    taxonomy = Taxonomy.read(args.taxonomy_dir)

    # load best models
    l1_model, l1_tokenizer = load_best_model(join(args.models_path, "L1"))
    l2_models_dict = {}  # key = class id, value = Model
    for category, class_id in taxonomy.category_to_class_id.items():
        path = join(args.models_path, category)
        if not isdir(path):
            continue

        model, _ = load_best_model(join(args.models_path, category))
        l2_models_dict[class_id] = model

    # assumes same tokenizer used on all models
    test_data = BertDataset(test_data, l1_tokenizer, args.max_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    # compute predictions and targets
    l1_targets = list(test_data.data["L1_target"])
    l2_targets = list(test_data.data["L2_target"])
    targets = list(zip(l1_targets, l2_targets))
    preds = batch_predict(l1_model, l2_models_dict, test_dataloader, args.device, strategy=args.strategy)

    # compute prediction accuracy
    overall_acc, l1_overall_acc, l1_class_accs = \
        evaluate_batch_predictions(preds, targets, taxonomy.num_classes)

    # log results
    logger.debug("Overall Accuracy:", overall_acc)
    logger.debug("L1 Accuracy:", l1_overall_acc)

    logger.debug("Accuracies by Category:")
    for class_id, l1_class_acc in l1_class_accs.items():
        class_name = taxonomy.class_id_to_category(class_id)
        logger.debug(f"L1 Category {class_name}:", l1_class_acc)
