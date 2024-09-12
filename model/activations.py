import math
import torch
from torch import nn


class GELUActivation(nn.Module):
    """
    GELU Activation function used in the transformer.
    This is a smooth version of ReLU with improved convergence properties.
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
