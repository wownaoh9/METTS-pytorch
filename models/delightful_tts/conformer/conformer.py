import torch
from torch import nn
from torch.nn import Module
import math

from .conformer_block import ConformerBlock


class Conformer(Module):
    r"""`Conformer` class represents the `Conformer` model which is a sequence-to-sequence model
    used in some modern automated speech recognition systems. It is composed of several `ConformerBlocks`.

    Args:
        dim (int): The number of expected features in the input.
        n_layers (int): The number of `ConformerBlocks` in the Conformer model.
        n_heads (int): The number of heads in the multiheaded self-attention mechanism in each `ConformerBlock`.
        embedding_dim (int): The dimension of the embeddings.
        p_dropout (float): The dropout probability to be used in each `ConformerBlock`.
        kernel_size_conv_mod (int): The size of the convolving kernel in the convolution module of each `ConformerBlock`.
        with_ff (bool): If True, each `ConformerBlock` uses FeedForward layer inside it.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float,
        kernel_size_conv_mod: int,
        with_ff: bool,
    ):
        super().__init__()
        self.dim = dim
        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    dim,
                    n_heads,
                    kernel_size_conv_mod=kernel_size_conv_mod,
                    dropout=p_dropout,
                    with_ff=with_ff,
                )
                for _ in range(n_layers)
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward Pass of the Conformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            mask (Tensor): The mask tensor.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, num_features).
        """
        mask = get_mask_from_lengths(x_length)
        attn_mask = mask.view((mask.shape[0], 1, 1, mask.shape[1]))
        attn_mask.to(x.device)
        encoding = positional_encoding(
            self.dim,
            x.shape[1],
        ).to(x.device)
        for enc_layer in self.layer_stack:
            x = enc_layer(
                x,
                mask=mask,
                slf_attn_mask=attn_mask,
                encoding=encoding,
            )
        return x, mask

def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    r"""Generate a mask tensor from a tensor of sequence lengths.

    Args:
        lengths (torch.Tensor): A tensor of sequence lengths of shape: (batch_size, )

    Returns:
        torch.Tensor: A mask tensor of shape: (batch_size, max_len) where max_len is the
            maximum sequence length in the provided tensor. The mask tensor has a value of
            True at each position that is more than the length of the sequence (padding positions).

    Example:
      lengths: `torch.tensor([2, 3, 1, 4])`
      Mask tensor will be: `torch.tensor([
            [False, False, True, True],
            [False, False, False, True],
            [False, True, True, True],
            [False, False, False, False]
        ])`
    """
    # Get batch size
    batch_size = lengths.shape[0]

    # Get maximum sequence length in the batch
    max_len = int(torch.max(lengths).item())

    # Generate a tensor of shape (batch_size, max_len)
    # where each row contains values from 0 to max_len
    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    # Compare each value in the ids tensor with
    # corresponding sequence length to generate a mask.
    # The mask will have True at positions where id >= sequence length,
    # indicating padding positions in the original sequences
    return ids >= lengths.unsqueeze(1).type(torch.int64).expand(-1, max_len)
def positional_encoding(
    d_model: int, length: int,
) -> torch.Tensor:
    r"""Function to calculate positional encoding for transformer model.

    Args:
        d_model (int): Dimension of the model (often corresponds to embedding size).
        length (int): Length of sequences.

    Returns:
        torch.Tensor: Tensor having positional encodings.
    """
    # Initialize placeholder for positional encoding
    pe = torch.zeros(length, d_model)

    # Generate position indices and reshape to have shape (length, 1)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)

    # Calculate term for division
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * -(math.log(10000.0) / d_model),
    )

    # Assign sin of position * div_term to even indices in the encoding matrix
    pe[:, 0::2] = torch.sin(position * div_term)

    # Assign cos of position * div_term to odd indices in the encoding matrix
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add an extra dimension to match expected output shape
    return pe.unsqueeze(0)