import math
import torch
from torch import nn, optim
from einops import rearrange, repeat

from model.activations import GELUActivation


class MultiHeadAttention(nn.Module):
    """
    This is the core self-attention mechanism. It projects the input to query, key,
    and value vectors and computes the attention scores, which dictate how much
    attention each token should pay to every other token in the sequence.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # linear layer to project input to query, key, and value vectors (qkv)
        self.qkv = nn.Linear(self.hidden_size, self.all_head_size * 3)

        # dropout for attention probabilities
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])

        # linear layer to project attention output back to hidden_size
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

        # dropout for the final attention output
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # project input to query, key, and value
        qkv = self.qkv(x)
        # rearrange the qkv output to (batch_size, num_tokens, num_heads, head_size)
        query, key, value = rearrange(qkv, 'b n (three h d) -> three b n h d', three=3, h=self.num_attention_heads)

        # compute attention scores as scaled dot-product of query and key
        attention_scores = torch.einsum('b n h d, b m h d -> b h n m', query, key) / math.sqrt(self.attention_head_size)
        # softmax to get attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # dropout to the attention probabilities
        attention_probs = self.attn_dropout(attention_probs)

        # compute the weighted sum of value vectors
        attention_output = torch.einsum('b h n m, b m h d -> b n h d', attention_probs, value)
        # rearrange the attention output back to (batch_size, num_tokens, hidden_size)
        attention_output = rearrange(attention_output, 'b n h d -> b n (h d)')
        # project the attention output back to hidden_size
        attention_output = self.output_projection(attention_output)
        # apply dropout to the final output
        attention_output = self.output_dropout(attention_output)

        return (attention_output, attention_probs) if output_attentions else (attention_output, None)


class MultiLayerPerceptron(nn.Module):
    """
    The Multi-Layer Perceptron (MLP) block in each transformer layer, consisting of
    two linear layers with a non-linear activation in between.
    """
    def __init__(self, config):
        super().__init__()
        # first linear layer expands the hidden size to intermediate size
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        # GELU activation function
        self.activation = GELUActivation()
        # second linear layer projects intermediate size back to hidden size
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        # dropout for regularisation
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # pass through the first linear layer
        x = self.dense_1(x)
        # activation function
        x = self.activation(x)
        # pass through the second linear layer
        x = self.dense_2(x)
        # apply dropout
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block that consists of multi-head attention and an MLP,
    both with skip connections and layer normalisation.
    """
    def __init__(self, config):
        super().__init__()
        # multi-head self-attention
        self.attention = MultiHeadAttention(config)
        # layer normalisation before attention
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        # multi-layer perceptron  block
        self.mlp = MultiLayerPerceptron(config)
        # layer normalisation before MLP
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # layer normalisation and attention block
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)

        # add skip connection
        x = x + attention_output  # Add skip connection

        # layer normalisation and MLP block
        mlp_output = self.mlp(self.layernorm_2(x))

        # add skip connection
        x = x + mlp_output  # Add skip connection
        return (x, attention_probs) if output_attentions else (x, None)