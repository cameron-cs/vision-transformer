from torch import nn

from model.blocks import TransformerBlock


class Encoder(nn.Module):
    """
    The transformer encoder consisting of multiple blocks, each containing multi-head
    attention and MLP blocks.
    """
    def __init__(self, config):
        super().__init__()
        # stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        # pass through each transformer block
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)

            # store attention probabilities if needed
            if output_attentions:
                all_attentions.append(attention_probs)

        # final output and attentions if required
        return (x, all_attentions) if output_attentions else (x, None)