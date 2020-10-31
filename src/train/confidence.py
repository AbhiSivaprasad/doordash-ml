import numpy as np


def get_confidence_scores(class_probs: np.ndarray, strategy="max", **kwargs):
    """Given model probabilities for each class, compute a confidence score"""
    if strategy == 'max':
    elif strategy == 'max_diff':
    else:
        raise ValueError("Invalid batch prediction strategy supplied:", strategy)


def get_confidence_scores_max(class_probs: np.ndarray):
    """
    Given model probabilities for each class, return a confidence score
    equal to the max probability across classes

    :param class_probs: (batch_size, num_classes) np array of class probs
    """
    return class_probs.max(axis=1)


def get_confidence_scores_max_diff(class_probs: np.ndarray):
    """
    Given model probabilities for each class, return a confidence score 
    equal to the difference between 2 highest class probabilities

    :param class_probs: (batch_size, num_classes) np array of class probs
    """
    sorted_class_probs = class_probs.copy().sort(axis=1)
    return sorted_class_probs[:, -1] - sorted_class_probs[:, -2]
