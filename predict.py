from src.train.run_prediction import run_prediction
from src.args import PredictArgs


if __name__ == "__main__":
    run_batch_prediction(args=PredictArgs().parse_args())
