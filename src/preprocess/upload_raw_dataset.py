import wandb
from tap import Tap


class UploadArgs(Tap):
    project: str = "doordash"
    """Name of wandb project to upload to"""
    artifact_name: str = "raw-dataset"
    """Name of wandb artifact to create"""
    artifact_path: str
    """Path to dataset to upload as wandb artifact"""


def upload(args: UploadArgs):
    """Upload dataset as wandb artifact"""
    run = wandb.init(project=args.project, job_type="upload")
    artifact = wandb.Artifact('raw-dataset', type='dataset')
    artifact.add_file(args.artifact_path)
    run.log_artifact(artifact)    


if __name__ == '__main__':
    upload(UploadArgs().parse_args())
