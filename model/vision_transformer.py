from torch import nn

from model.embeddings import PositionEmbeddings
from model.encoders import Encoder


class VisionTransformer(nn.Module):
    """
    The Vision Transformer model for image classification. Combines patch embeddings,
    transformer encoder, and a final classifier head.
    """
    def __init__(self, config):
        super().__init__()

        # embedding layer for patches and positions
        self.embedding = PositionEmbeddings(config)

        # transformer encoder
        self.encoder = Encoder(config)

        # classification head (linear layer that maps hidden state to number of classes)
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])

        # initialise model weights
        self.apply(self.init_weights)

    def forward(self, x, output_attentions=False):
        # get embeddings from the input images
        embedding_output = self.embedding(x)
        # pass through the encoder
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)

        # the output corresponding to the [CLS] token for classification
        logits = self.classifier(encoder_output[:, 0, :])

        # logits and attention probabilities if requested
        return (logits, all_attentions) if output_attentions else (logits, None)

    def init_weights(self, module):
        # init weights for Linear and Conv2d layers using Xavier uniform initialization
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # initialise LayerNorm parameters
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)