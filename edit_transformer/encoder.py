from torch import nn
from torch import Tensor

from edit_transformer.modules import LayerNorm, SublayerConnection, clones, PositionwiseFeedForward,\
    MultiHeadedAttention, PositionalEncoding


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)

        Attributes:
            self_attn (MultiHeadedAttention): multi-headed attention module used to compute the attention.
            feed_forward (PositionwiseFeedForward): position-wise feed-forward module used to project the result.
            sublayer (nn.ModuleList): 2 `SublayerConnection` used between attention and feed forward and after feed
                forward.
            size (int): the size of the output of the model (used for normalization).

    """
    def __init__(self, size: int, self_attn: MultiHeadedAttention, feed_forward: PositionwiseFeedForward,
                 dropout: float) -> None:
        """Initialize an encoder layer.

        Args:
            size (int): dimension of the model corresponds to the size of the input.
            self_attn (nn.Module): torch module defining a multi-head self-attention layer as in the transformer.
            feed_forward (nn.Module): torch module defining a point-wise feed-forward layer as in the transformer.
            dropout (float): the percentage of dropout to use in between self-attention and feed-forward.

        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward of an encoder layer of the "edit-transformer".

        Args:
            x (Tensor): input of the encoder layer of shape `(batch, seq_len, d_model)`
            mask (Tensor): mask of the input of shape `(batch, seq_len)`

        Returns:
            Tensor: the output of an encoder layer of shape `(batch, seq_len, d_model)`

        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers

    Attributes:
        layers (nn.ModuleList): `n` `EncoderLayer` module corresponding to the `n` layer of the transformer.
        norm (LayerNorm): a batch normalization layer defined by `layer.size`.
        embedding (nn.Embedding): embedding layer to embed the word index.

    """
    def __init__(self, layer: EncoderLayer, n: int, embedding: nn.Embedding, pos_encoding: PositionalEncoding) -> None:
        """Initialize the encoder of the Transformer.

        Args:
            layer (EncoderLayer): a torch module corresponding to an encoder layer.
            n (int): the number of layer to use in the encoder.
            embedding (nn.Embedding): embedding layer to embed the word index.
            pos_encoding (PositionalEncoding): positinal encoding to add on each embeded input.

        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)
        self.embedding = embedding
        self.pos_encoding = pos_encoding

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Pass the input (and mask) through each layer in turn.

        Args:
            x (Tensor): an input tensor of shape `(batch, seq_len)`.
            mask (Tensor): a mask tensor of shape `(batch, seq_len)`.

        Returns:
            Tensor: an output tensor of shape `(batch, seq_len, d_model)`.

        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
