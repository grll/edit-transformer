import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor


# General Modules used in the encoder and decoder
class LayerNorm(nn.Module):
    """Module that perform normalization of the input x.

    Attributes:
        a_2 (nn.Parameter): learnable parameter used to project the normalization of the batch.
        b_2 (nn.Parameter): learnable parameter corresponding to the bias of the projection on the normalization.
        eps (float): epsilon float to use when dividing by the std.

    """
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        """Initialize the layer normalization module.

        Args:
            features (int): dimension of the features / model to normalize.
            eps (float): epsilon float to use when dividing by the std.

        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Perform the normalization on `x`.

        Args:
            x (Tensor): input tensor of shape `(batch, seq_len, size)`.

        Returns:
            Tensor: output normalized tensor of the same shape.

        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm.

    Notes:
         for code simplicity the norm is first as opposed to last.

    Attributes:
        norm (LayerNorm): A normalization layer defined with `size`.
        dropout (nn.Dropout): A dropout module used after the sublayer and before the residual connection.

    """
    def __init__(self, size: int, dropout: float) -> None:
        """Initialize the sublayer connection (residual connection + normalization).

        Args:
            size (int): the size used as the model dimension.
            dropout (float): the dropout percent to use before performing the residual connection.

        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """Apply residual connection to any sublayer with the same size.

        Args:
            x (Tensor): an input tensor of shape `(batch, seq_len, d_model)`.
            sublayer (nn.Module): a sublayer that will be applied on x after normalization and before residual addition.

        Returns:
            Tensor: an output tensor of the same shape as the input.

        """
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation.

    Attributes:
        w_1 (nn.Linear): a linear layer which project from `d_model` to `d_ff`.
        w_2 (nn.Linear): a linear layer which project from `d_ff` to `d_model`.
        dropout (nn.Dropout): a dropout layer used between the two linear layers.

    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize the Positionwise Feed Forward module.

        Args:
            d_model (int): dimension used by the model (output of the multi-headed attention).
            d_ff (int): dimension to use in the positionwise feed-forward.
            dropout (float): dropout rate to use between the two linear layers of the positionwise feed-forward.

        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the position wise feed-foward layer on the input tensor x.

        Args:
            x (Tensor): input tensor x of shape `(batch, seq_len, d_model)`.

        Returns:
            Tensor: output tensor of the same shape as the input tensor `x`.

        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Implement the PE function.

    Attributes:
        dropout (nn.Dropout): A dropout module used after position encoding.
        pe (nn.Parameter): fixed parameter assigned to the module corresponding to the positional embedding of shape
            `(max_len, d_model)`.

    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        """Initialize the positional encoding module.

        Args:
            d_model (int): the size used by the model.
            dropout (float): the dropout rate to use after position encoding.
            max_len (int): the maximum length that can be encoded using this module.

        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the positional encoding on the input tensor x.

        Args:
            x (Tensor): an input tensor x of shape `(batch, seq_len, d_model)`.

        Returns:
            Tensor: a position encoded tensor of the same shape as `x`.
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Clone a given module into N identical ones.

    Args:
        module (nn.Module):  an input module that will be cloned.
        n (int): the number of time that the module will be cloned.

    Returns:
        nn.ModuleList: a module list corresponding to `module` cloned `n` times.

    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# Attention module
def attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None,
              dropout: Optional[nn.Module] = None) -> Tuple[Tensor, Tensor]:
    """Compute 'Scaled Dot Product Attention' as defined in the transformer.

    Args:
        query (Tensor): a projected query tensor of shape `(batch, h, seq_len, d_k)`.
        key (Tensor): a projected key tensor of shape `(batch, h, seq_len, d_k)`.
        value (Tensor): a projected value tensor of shape `(batch, h, seq_len, d_k)`.
        mask (Optional[Tensor]): an optional mask tensor of shape `(batch, 1, 1, seq_len)` or
            `(batch, 1, seq_len, seq_len)` used in the decoder.
        dropout (Optional[nn.Module]): an optional dropout module used on the attention weights.

    Returns:
        Tuple[Tensor, Tensor]:
            - output tensor result of the attention computation between the key, query and value of shape
                `(batch, h, seq_len, d_k)`.
            - tensor of attention weights not multiplied with the values of shape `(batch, h, seq_len, seq_len)`.

    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Perform the attention previously defined in multiple subspaces.

    Attributes:
        d_k (int): dimension output in each subspace.
        h (int): number of heads to use.
        linears (nn.ModuleList): a list of linear modules to perform projection of the key, query and value and
            the final output from `d_model` to `d_model`.
        attn (Tensor): attention weight tensor of the last multi-headed forward run of shape
            `(batch, h, seq_len, seq_len)`.
        dropout (nn.Dropout): dropout module to be used in the attention function.

    """
    def __init__(self, h: int, d_model: int, dropout: float = 0.1, d_init: Optional[int] = None) -> None:
        """Take in model size and number of heads.

        Notes:
            `d_model` must be divisible by `h`.

        Args:
            h (int): number of heads to use.
            d_model (int): dimension of the model.
            dropout (float): dropout rate to use when performing attention.
            d_init (Optional[int]): initial dimension used for the key and value before projection to `d_model`
                if is different from `d_model` (when concatening with the edit vector).

        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        if d_init is not None:
            self.linears[1] = nn.Linear(d_init, d_model)
            self.linears[2] = nn.Linear(d_init, d_model)
        self.attn: Tensor = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Perform the multi-headed attention on the query, key and values parameters.

        Args:
            query (Tensor): tensor from which to project the query of shape `(batch, seq_len, d_model)`.
            key (Tensor): tensor from which to project the key of shape `(batch, seq_len, d_model)` or
                `(batch, seq_len, d_model + d_edit)` if "edit-attention".
            value (Tensor): tensor from which to project the value of shape `(batch, seq_len, d_model)`or
                `(batch, seq_len, d_model + d_edit)` if "edit-attention".
            mask (Optional[Tensor]): an optional mask tensor of shape `(batch, seq_len)` or
                `(batch, seq_len, seq_len)` used in the decoder to hide part of the sequence.

        Returns:
            Tensor: output tensor of the multi-headed attention on the query, key, and value of shape
                `(batch, seq_len, d_model)`.

        """
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1)  # mask format for  good weighted value after the softmax.

            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
