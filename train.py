#from src.train.run_training import run_training
from src.train.train_resnet import run_resnet_training
#from src.args import TrainArgs
from src.args import ResnetTrainArgs

if __name__ == "__main__":
    print(ResnetTrainArgs().__dict__)
    run_resnet_training(ResnetTrainArgs())
