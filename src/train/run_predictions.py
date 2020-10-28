from os import listdir
from os.path import join, isdir
from torch.utils.data import DataLoader

from .utils import load_best_model, DefaultLogger
from .batch_predict import batch_predict
from .evaluate import evaluate_predictions
from ..args import PredictArgs
from ..data.bert import BertDataset
from ..data.taxonomy import Taxonomy


def run_predictions(args: PredictArgs):
    """
    Construct hierarchy of models and run predictions
    """
    logger = DefaultLogger()

    taxonomy = Taxonomy.read(args.taxonomy_dir)
    l1_model, l1_tokenizer = load_best_model(join(args.models_path, "L1"))
    l2_models_dict = {}  # key = class id, value = Model
    for class_id, category in enumerate(taxonomy.category_to_class_id):
        path = join(args.models_path, category)
        if not isdir(path):
            continue

        model, _ = load_best_model(join(args.models_path, category))
        l2_models_dict[class_id] = model

    # get raw test data
    test_data = pd.read_csv(args.test_path)

    # assumes same tokenizer used on all models
    test_data = BertDataset(test_data, l1_tokenizer, args.max_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=args.predict_batch_size)

    # compute predictions and accuracy
    preds = batch_predict(l1_model, l2_models_dict, test_dataloader, strategy="greedy", device=args.device)
    l1_targets = test_data["L1_target"]
    l2_targets = test_data["L2_target"]

    # compute prediction accuracy
    overall_acc, l1_overall_acc, l1_class_accs = evaluate_predictions(preds, targets)

    # log results
    logger.debug("Overall Accuracy:", overall_acc)
    logger.debug("L1 Accuracy:", l1_overall_acc)

    for class_id, l1_class_acc in l1_class_accs.items():
        class_name = taxonomy.category_to_class_id[class_id]
        logger.debug(f"L1 Accuracy for Category {class_name}:", l1_class_acc)
