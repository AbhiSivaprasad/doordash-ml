import torch
import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, 
                 vision_model: nn.Module, 
                 text_model: nn.Module, 
                 vision_output_head: nn.Module,
                 text_output_head: nn.Module,
                 num_classes: int, 
                 hybrid_embedding_dim: int = None,
                 hybrid_output_head: nn.Module = None,
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
            self.hybrid_output_head = nn.Sequential(
                nn.Linear(hybrid_embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.hybrid_output_head = hybrid_output_head

    def forward(self, x):
        text, image = x
        has_text, has_image = text is not None, image is not None
        vision_embedding = text_embedding = None

        # pure text model
        if has_text and not has_image:
            with torch.no_grad():
                text_embedding = self.text_model(text)[:, 0]
            return self.text_output_head(text_embedding)

        # pure vision model
        if has_image and not has_text:
            with torch.no_grad():
                vision_embedding = self.vision_model(image)
            return self.vision_output_head(vision_embedding)

        # concatenate embeddings
        if has_text and has_image:
            with torch.no_grad():
                text_embedding = self.text_model(text)[:, 0]
                vision_embedding = self.vision_model(image)

            embedding = torch.cat((vision_embedding, text_embedding), 1)
            return self.hybrid_output_head(embedding)
