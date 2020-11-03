from src.predict.run_prediction import run_prediction
from src.args import PredictArgs


if __name__ == "__main__":
    run_prediction(args=PredictArgs().parse_args())
