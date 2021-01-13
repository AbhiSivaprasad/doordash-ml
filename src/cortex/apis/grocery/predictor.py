import cv2
import wandb
import boto3

from src.cortex.apis.utils import download_directory_from_s3
from src.models.handlers.hybrid import HybridHandler
from src.predict.batch_predict import batch_predict
from src.data.dataset.bert import BertDataset
from src.data.dataset.image import ImageDataset
from src.data.dataset.hybrid import HybridDataset
from src.data.image_utils import get_image_hashdir
from src.data.taxonomy import Taxonomy

from os.path import join
from torch.utils.data import DataLoader


class PythonPredictor:
    def __init__(self, config, python_client=None):
        # constants
        self.model_dir = "model/"
        self.download_dir = "download/"
        self.image_dir = "images/"

        # fetch taxonomy from wandb
        wandb_api = wandb.Api({"project": "main"})
        wandb_api.artifact(config['taxonomy_wandb_identifier']).download(self.download_dir)
        taxonomy = Taxonomy().from_csv(join(self.download_dir, "taxonomy.csv"))

        # fetch models from s3
        s3_resource = boto3.resource("s3")
        model_bucket = s3_resource.Bucket(config["model_bucket"])
        download_directory_from_s3(model_bucket, config["model_bucket_folder"], self.model_dir)

        # model versions
        with open(join(self.model_dir, "versions.json"), "r") as f:
            versions = json.load(f)

        # device setup
        self.device = (torch.device("cuda") 
                       if torch.cuda.is_available() 
                       else torch.device("cpu"))
        
        # setup L1, L2 models
        l1_handler = HybridHandler.load(join(self.model_dir, 'grocery'))

        self.l1_model, self.l1_labels, self.tokenizer \
            = l1_handler.model, l1_handler.labels, l1_handler.tokenizer

        self.l2_models_dict = {}  # key = class id, value = model
        self.l2_labels_dict = {}  # key = class id, value = labels

        for node, path in taxonomy.iter(skip_leaves=True):
            # skip l1
            if len(path) == 1:
                continue

            # load checkpoint
            category_dir = join(self.model_dir, node.category_id)
            handler = HybridHandler.load(category_dir)

            l2_models_dict[node.category_id] = handler.model
            l2_labels_dict[node.category_id] = handler.labels

        # args
        self.max_seq_length = 100
        self.image_size = 256
        self.batch_size = 1

    def predict(self, payload, query_params, headers):
        print(payload)

        return ["dummy"]
        # image = 
        text = self.process_payload(payload)

        text_dataset = BertDataset(data, self.tokenizer, self.max_seq_length, preserve_na=True)
        image_dataset = ImageDataset(data, self.image_dir, self.image_size, preserve_na=True, val=True)
        dataset = HybridDataset(image_dataset, text_dataset, val=True)
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
            l1_category = self.l1_labels[l1_class_id]
            l2_category = self.l2_labels_dict[l1_category][l2_class_id]
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

    def process_payload(self, payload):
        item_name = image = None
        if 'item_name' in payload:
            item_name = payload['item_name']
        elif 'image' in payload:
            image = payload['image']

        image_name = hash(image)
        data = [item_name, image_name]

        # creating appropriate hash directory in image dir and compute image path
        hash_dir = get_image_hashdir(image_name) 
        image_path = join(self.image_dir, hash_dir, image_name)

        # write image
        if not cv2.imwrite(filepath, img):
            print("could not write image")

        df = pd.DataFrame(data, columns=["Name", "Image Name"])

        return item_name, image

    def process_image():
        # process image
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        scaling_factor = 1024.0 / max(height, width)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
     
