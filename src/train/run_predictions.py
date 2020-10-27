from os import listdir
from os.path import join, isdir

from .utils import load_best_model


def run_predictions(args: PredictArgs):
    """
    Construct hierarchy of models and run predictions
    """
    l1_model = None
    l2_models_dict = {}  # key = class id, value = Model
    for category in listdir(args.models_path):
        path = join(args.models_path, category)
        if not isdir(path):
            continue

        model = load_best_model(join(args.models_path, category))
        l2_models_dict[ind] = model


    return batch_predict(l1_model, l2_models_dict, test_dataloader, strategy="greedy")
        
