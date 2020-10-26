from .train/run_training import run_training


if __name__ == "__main__":
    run_training(args=TrainArgs().parse_args())
