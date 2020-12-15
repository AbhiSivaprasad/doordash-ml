from src.train.run_training import run_training
from src.args import TrainArgs


if __name__ == "__main__":
    run_training(args=TrainArgs().parse_args())
