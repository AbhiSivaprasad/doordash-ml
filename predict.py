from src.train.run_predictions import run_predictions
from src.args import PredictArgs


if __name__ == "__main__":
    run_predictions(args=PredictArgs().parse_args())
