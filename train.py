#from src.train.run_training import run_training
from src.train.train_resnet import run_training
#from src.args import TrainArgs
from src.args import ResnetTrainArgs

if __name__ == "__main__":
    run_training(args=ResnetTrainArgs().parse_args())
