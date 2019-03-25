import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from edit_transformer.modules import LayerNorm, SublayerConnection, clones, MultiHeadedAttention,\
    PositionwiseFeedForward, PositionalEncoding


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below).

    Attributes:
        size (int): model size used in layer normalization.
        self_attn (MultiHeadedAttention): multi-headed self-attention module previously defined.
        src_attn (MultiHeadedAttention): multi-headed attention module between the last encoder layer and the decoder
            layer.
        feed_forward (PositionwiseFeedForward): position wise feed forward module similar as in the encoder.
        dropout (nn.Dropout): dropout module used at the end of the decoder layer.

    """
    def __init__(self, size: int, self_attn: MultiHeadedAttention, src_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float) -> None:
        """Initialize a decoder layer.

        Args:
            size (int): model size used in normalization layer.
            self_attn (MultiHeadedAttention): self attention layer from the multi-head attention module.
            src_attn (MultiHeadedAttention): decoder-encoder attention layer from the multi-head attention module.
            feed_forward (PositionwiseFeedForward): feed-forward module used at the end of each decoder layer.
            dropout: percent of dropout to use at the end of the decoder layer.

        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor, edit_embed: Tensor) -> Tensor:
        """Forward through a decoder layer.

        Args:
            x (Tensor): decoder input tensor of shape `(batch, tgt_seq_len + 1, word_dim)`.
            memory (Tensor): memory / encoding of the source sequence tensor of shape `(batch, src_seq_len, d_model)`.
            src_mask (Tensor): mask of the source sequences of shape `(batch, src_seq_len, 1)`.
            tgt_mask (Tensor): triangular mask over the target sequence `(batch, tgt_seq_len + 1, 1)`.
            edit_embed (Tensor): edition embedding tensor for each sequence of shape `(batch, d_edit)`.

        Returns:
            Tensor: a decoder layer output tensor of shape `(batch, tgt_seq_len, d_model)`.

        """
        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        # concat edit_embed
        # batch_size, tgt_seq_len, _ = x.shape
        # edit_dim = edit_embed.size(1)
        # emb = torch.cat((self.norm(x), edit_embed.unsqueeze(dim=1).expand(batch_size, tgt_seq_len, edit_dim)), dim=2)

        return self.dropout(self.feed_forward(self.norm(x)))


class Decoder(nn.Module):
    """Generic N layer decoder with masking.

    Attributes:
        layers (nn.ModuleList): a list of `n` `DecoderLayer` module used in the decoder of the model.
        norm (LayerNorm): a normalization layer used a the end of the decoder.
        embedding (nn.Embedding): embedding layer to embed the word index.
        pos_encoding (PositionalEncoding): positinal encoding to add on each embeded input.

    """
    def __init__(self, layer: DecoderLayer, n: int, embedding: nn.Embedding, pos_encoding: PositionalEncoding) -> None:
        """Initialize the decoder.

        Args:
            layer (DecoderLayer): the `DecoderLayer` module that will be cloned `n` times.
            n (int): number of layers to use in the decoder.
            embedding (nn.Embedding): embedding layer to embed the word index.
            pos_encoding (PositionalEncoding): positinal encoding to add on each embeded input.

        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)
        self.embedding = embedding
        self.pos_encoding = pos_encoding

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor, edit_embed: Tensor) -> Tensor:
        """Forward through the decoder.

        Args:
            x (Tensor): decoder input tensor of shape `(batch, tgt_seq_len + 1, 1)`.
            memory (Tensor): memory / encoding of the source sequence tensor of shape `(batch, src_seq_len, d_model)`.
            src_mask (Tensor): mask of the source sequences of shape `(batch, src_seq_len, 1)`.
            tgt_mask (Tensor): triangular mask over the target sequence `(batch, tgt_seq_len, 1)`.
            edit_embed (Tensor): edition embedding tensor for each sequence of shape `(batch, d_edit)`.

        Returns:
            Tensor: a decoder layer output tensor of shape `(batch, tgt_seq_len, d_model)`.

        """
        x = self.embedding(x)
        x = self.pos_encoding(x)

        seq_len = memory.shape[1]
        edit_embed = edit_embed.unsqueeze(1).expand(-1, seq_len, -1)
        combined_input = torch.cat((memory, edit_embed), dim=-1)

        for layer in self.layers:
            x = layer(x, combined_input, src_mask, tgt_mask, edit_embed)
        return self.norm(x)


class Generator(nn.Module):
    """Generate a probability over the vocabulary for each timestep.

    Attributes:
        vocab_projection_pos (nn.Linear): linear projection from model output to the word embedding dimension.
        vocab_projection_neg (nn.Linear): linear projection from model output to the word embedding dimension.
        word_embeddings (nn.Embedding): Embedding matrix of shape `(num_words, word_embeddings_size)`.

    """
    def __init__(self, d_model: int, word_embeddings: nn.Embedding) -> None:
        """Initialize the generator.

        Args:
            d_model (int): dimension of the model output.
            word_embeddings (nn.Embedding): word embeddings matrix.

        """
        super(Generator, self).__init__()
        word_dim = word_embeddings.shape[-1]
        self.vocab_projection_pos = nn.Linear(d_model, word_dim)
        self.vocab_projection_neg = nn.Linear(d_model, word_dim)
        self.word_embeddings = word_embeddings

    def forward(self, z: Tensor) -> Tensor:
        """Forward through the generator.

        Args:
            z (Tensor): output tensor of the decoder of shape `(batch, tgt_seq_len, d_model)`.

        Returns:
            Tensor: output probability of the generator tensor of shape `(batch, tgt_seq_len, d_vocab)`.

        """
        batch_size, tgt_seq_len, _ = z.shape

        vocab_query_pos = self.vocab_projection_pos(z)
        vocab_query_neg = self.vocab_projection_neg(z)

        vocab_logit_pos = F.relu(torch.matmul(vocab_query_pos, self.word_embeddings.weight.t()))
        vocab_logit_neg = F.relu(torch.matmul(vocab_query_neg, self.word_embeddings.weight.t()))

        return F.softmax(vocab_logit_pos - vocab_logit_neg, dim=-1)
