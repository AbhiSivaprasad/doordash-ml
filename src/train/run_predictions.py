from os import listdir
from os.path import join, isdir
from torch.utils.data import DataLoader

from .utils import load_best_model, DefaultLogger
from .batch_predict import batch_predict
from .evaluate import evaluate_predictions
from ..args import PredictArgs
from ..data.bert import BertDataset


def run_predictions(args: PredictArgs):
    """
    Construct hierarchy of models and run predictions
    """
    logger = DefaultLogger()

    l1_model, l1_tokenizer = load_best_model(join(args.models_path, "L1"))
    l2_models_dict = {}  # key = class id, value = Model
    for category in listdir(args.models_path):
        path = join(args.models_path, category)
        if not isdir(path):
            continue

        model, _ = load_best_model(join(args.models_path, category))
        l2_models_dict[ind] = model

    # get raw test data
    test_data = pd.read_csv(args.test_path)

    # assumes same tokenizer used on all models
    test_data = BertDataset(test_data, l1_tokenizer, args.max_seq_length)
    test_dataloader = DataLoader(test_data, batch_size=args.predict_batch_size)

    # compute predictions and accuracy
    preds = batch_predict(l1_model, l2_models_dict, test_dataloader, strategy="greedy")
    targets = test_dataloader.targets 
    accuracy = evaluate_predictions(preds, targets)

    # log result
    logger.debug("Batch Prediction Accuracy:", accuracy)
