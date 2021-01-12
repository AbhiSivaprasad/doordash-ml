import boto3

from ..utils import download_directory_from_s3
from ....models.handlers.hybrid import HybridHandler
from ....predict.batch_predict import batch_predict
from ....data.dataset.bert import BertDataset
from ....data.dataset.image import ImageDataset
from ....data.dataset.hybrid import HybridDataset

from torch.utils.data import DataLoader


class PythonPredictor:
    def __init__(self, config, python_client):
        # constants
        model_dir = "model"

        # fetch models from s3
        s3_resource = boto3.resource("s3")
        model_bucket = s3_resource.Bucket(config["model_bucket"])
        download_directory_from_s3(model_bucket, config["model_bucket_folder"], model_dir)

        # model versions
        with open(join(model_dir, "versions.json"), "r") as f:
            versions = json.load(f)

        # device setup
        self.device = (torch.device("cuda") 
                       if torch.cuda.is_available() 
                       else torch.device("cpu"))
        
        # setup L1, L2 models
        handler = HybridHandler.load(dir_path)

        # args
        self.max_seq_length = 100
        self.image_size = 256
        self.batch_size = 1

    def predict(self, payload, query_params, headers):
        image = 
        text = 

        text_dataset = BertDataset(data, self.tokenizer, self.max_seq_length, preserve_na=True)
        image_dataset = ImageDataset(data, self.image_dir, self.image_size, preserve_na=True)
        dataset = HybridDataset(image_dataset, text_dataset)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        preds, l2_confidence_scores = batch_predict(self.l1_model, 
                                                    self.l1_labels, 
                                                    self.l2_models_dict, 
                                                    dataloader, 
                                                    self.device, 
                                                    strategy="complete")

        # only one data point
        assert len(preds) == 1
        assert len(l2_confidence_scores) == 1

        preds = preds[0]
        l2_confidence_scores = l2_confidence_scores[0]

        # process predictions 
        def get_labels(l1_class_id, l2_class_id):
            l1_category = l1_labels[l1_class_id]
            l2_category = l2_labels_dict[l1_category][l2_class_id]
            return l1_category, l2_category

        preds = [[get_labels(l1_pred, l2_pred) for l1_pred, l2_pred in topk] for topk in preds]
        
        # build and return prediction results payload
        return [
            {
                "L1 Category": l1_pred,
                "L2 Category": l2_pred,
                "Confidence": confidence
            } for (l1_pred, l2_pred), confidence in zip(preds, l2_confidence_scores)
        ]
