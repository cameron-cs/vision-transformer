import torch

from torch import nn
from einops import rearrange, repeat


class PatchEmbeddings(nn.Module):
    """
    This module splits an image into smaller patches and projects each patch into a
    hidden vector space using a Conv2d layer.
    """
    def __init__(self, config):
        super().__init__()
        # image and patch sizes and number of channels
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # the number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # convolution layer to extract patches and project them into hidden_size dimension
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # conv2d to obtain patch embeddings
        x = self.projection(x)

        # rearrange the output to (batch_size, num_patches, hidden_size)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class PositionEmbeddings(nn.Module):
    """
    Adds the [CLS] token and positional embeddings to the patch embeddings.
    """
    def __init__(self, config):
        super().__init__()
        # Initialise patch embedding module
        self.patch_embeddings = PatchEmbeddings(config)

        # [CLS] token is added to the sequence for classification tasks
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # positional embeddings to retain positional information
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1,
                                                            config["hidden_size"]))
        # dropout for regularisation
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # patch embeddings from input images
        x = self.patch_embeddings(x)
        batch_size, num_patches, _ = x.size()

        # repeat [CLS] token for each batch
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)

        # concatenate the [CLS] token with the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional embeddings to the sequence
        x = x + self.position_embeddings

        # dropout for regularisation
        x = self.dropout(x)
        return x

