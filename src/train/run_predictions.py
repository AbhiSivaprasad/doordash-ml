from os import listdir
from os.path import join, isdir
from torch.utils.data import DataLoader

from .utils import load_best_model, DefaultLogger
from .batch_predict import batch_predict
from .evaluate import evaluate_predictions
from ..args import PredictArgs
from ..data.bert import BertDataset
from ..data.taxonomy import read_taxonomy


def run_predictions(args: PredictArgs):
    """
    Construct hierarchy of models and run predictions
    """
    logger = DefaultLogger()

    taxonomy = read_taxonomy(args.taxonomy_dir)
    l1_model, l1_tokenizer = load_best_model(join(args.models_path, "L1"))
    l2_models_dict = {}  # key = class id, value = Model
    for category in listdir(args.models_path):
        path = join(args.models_path, category)
        if not isdir(path):
            continue

        model, _ = load_best_model(join(args.models_path, category))
        l2_models_dict[taxonomy.['category_to_class_id'][category]] = model

    # get raw test data
    test_data = pd.read_csv(args.test_path)

    # assumes same tokenizer used on all models
    test_data = BertDataset(test_data, l1_tokenizer, args.max_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=args.predict_batch_size)

    # compute predictions and accuracy
    preds = batch_predict(l1_model, l2_models_dict, test_dataloader, strategy="greedy")
    l1_targets = test_data["L1_target"]
    l2_targets = test_data["L2_target"]

    # compute prediction accuracy
    overall_acc, l1_acc, l2_accs = evaluate_predictions(preds, targets)

    # log results
    logger.debug("Overall Accuracy:", overall_acc)
    logger.debug("L1 Accuracy:", l1_acc)

    for class_id, l2_acc in l2_accs.items():
        class_name = taxonomy['category_to_class_id'][class_id]
        logger.debug(f"L2 Accuracy for Category {class_name}:", l2_acc)
