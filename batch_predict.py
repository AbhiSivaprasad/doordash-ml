from src.predict.run_batch_prediction import run_batch_prediction
from src.args import BatchPredictArgs


if __name__ == "__main__":
    run_batch_prediction(args=BatchPredictArgs().parse_args())
