import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, 
                 vision_model: nn.Module, 
                 text_model: nn.Module, 
                 vision_output_head: nn.Module,
                 text_output_head: nn.Module,
                 num_classes: int, 
                 hybrid_output_head: nn.Module = None
                 hidden_dim: int = 2048, 
                 dropout: float = 0.3):
        super(HybridModel, self).__init__()

        # text, vision models to ensemble
        self.vision_model = vision_model
        self.text_model = text_model

        # output heads when falling back onto solely text or vision
        self.vision_output_head = vision_output_head
        self.text_output_head = text_output_head

        # initialize new output head if hybrid head is not supplied
        if hybrid_output_head is None:
            self.hybrid_output_head = nn.Sequential([
                nn.Linear(, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ])
        else:
            self.hybrid_output_head = hybrid_output_head

    def forward(self, x):
        text, image = x
        vision_embedding = text_embedding = None

        # pure text model
        if text and not image:
            text_embedding = self.text_model(text)
            return self.text_output_head(text_embedding)

        # pure vision model
        if image and not text:
            vision_embedding = self.vision_model(image)
            return self.vision_output_head(vision_embedding)

        # concatenate embeddings
        if text and image:
            text_embedding = self.text_model(text)
            vision_embedding = self.vision_model(image)
            embedding = torch.cat((vision_embedding, text_embedding), 0)
            return self.hybrid_output_head(embedding)