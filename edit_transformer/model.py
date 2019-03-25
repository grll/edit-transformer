import copy

import torch
from torch import Tensor
import torch.nn as nn

from edit_transformer.encoder import Encoder, EncoderLayer
from edit_transformer.edit_encoder import EditEncoder
from edit_transformer.decoder import Decoder, DecoderLayer
from edit_transformer.decoder import Generator
from edit_transformer.modules import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EditTransformer(nn.Module):
    """Define the 'edit-transformer' model.

    Attributes:
        encoder (Encoder): encoder module of the transformer.
        edit_encoder (EditEncoder): edit-encoder module of the 'edit-transformer'.
        decoder (Decoder): decoder module of the transformer.
        generator (Generator): generator module that generate word probability from the model output.

    """
    def __init__(self, encoder: Encoder, edit_encoder: EditEncoder, decoder: Decoder, generator: Generator) -> None:
        super(EditTransformer, self).__init__()
        self.encoder = encoder
        self.edit_encoder = edit_encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor, insert: Tensor, delete: Tensor,
                draw_samples: bool = True, draw_p: bool = False) -> Tensor:
        """Process masked src and target sequences to return an output of probability over the dictionary.

        Args:
            src (Tensor): Tensor of the source sequences with shape `(batch_size, src_seq_len, 1)`.
            src_mask (Tensor): mask Tensor over the source sequences with shape `(batch_size, src_seq_len, 1)`.
            tgt (Tensor): Tensor of the target sentences with shape `(batch_size, tgt_seq_len + 1, 1)` with a <start>
                token at the beginning of the sequence.
            tgt_mask (Tensor): mask Tensor over the target sentences with shape `(batch_size, tgt_seq_len + 1, 1)`.
            insert (Tensor): tensor of insertions of shape `(batch, insert_seq_len, 1)`.
            delete (Tensor): tensor of deletions of shape `(batch, delete_seq_len, 1)`.
            draw_samples (bool): Weather to draw samples VAE style or not (keep True for training).
            draw_p (bool): Edit vector drawn from random prior distribution (keep False for training).

        Returns:
            Tensor: Tensor of probability output of the generator of shape `(batch_size, tgt_seq_len, d_vocab)`.

        """
        # encode
        source_embed = self.encoder(src, src_mask)
        edit_embed = self.edit_encoder(insert, delete, draw_samples=draw_samples, draw_p=draw_p)

        # decode
        # special triangle mask for decoding step by step:
        batch_size, tgt_seq_len, _ = tgt.shape
        triangle_mask = torch.ones([tgt_seq_len, tgt_seq_len], device=device).tril().unsqueeze(dim=0)
        tgt_mask = tgt_mask * triangle_mask
        logits = self.decoder(tgt, source_embed, src_mask, tgt_mask, edit_embed)

        return self.generator(logits)


def make_model(embedding: nn.Embedding, edit_dim: int = 128, n: int = 2, d_ff: int = 2048, h: int = 6,
               dropout: float = 0.1, lamb_reg: float = 100.0, norm_eps: float = 0.1,
               norm_max: float = 14.0) -> EditTransformer:
    """Helper: Construct a model from hyperparameters.

    Args:
        embedding (nn.Embedding): The embedding layer corresponding to the vocabulary used.
        edit_dim (int): dimension of the edit vector (default 128).
        n (int): number of encoder and decoder layers (default 2).
        d_ff (int): size of the feed-forward layer projection at the end of each encoder/decoder modules (default 2048).
        h (int): Number of heads for multi-head attention (needs to be a divisor of word_dim / d_model) (default 6).
        dropout (float): dropout rate used in between the different layers (default 0.1).
        lamb_reg (float): dispersion regularization term of for the edit-encoder vMF distribution (default 100.0).
        norm_eps (float): epsilon used to sample the random noise added to the norm of the edit_vector (default 0.1).
        norm_max (float):scalar used to rescale the norm samples (corresponds to the maximum norm) (default 14.0).

    Returns:
        EditTransformer: the corresponding EditTransformer torch module.

    """
    word_dim = embedding.weight.shape[-1]
    d_model = word_dim

    c = copy.deepcopy

    # create modules
    self_attn = MultiHeadedAttention(h, d_model, dropout)
    source_attn = MultiHeadedAttention(h, d_model, dropout, d_model + edit_dim)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_encoding = PositionalEncoding(d_model, dropout)

    # create encoder
    encoder_layer = EncoderLayer(d_model, c(self_attn), c(ff), dropout)
    encoder = Encoder(encoder_layer, n, embedding, pos_encoding)

    # create edit encoder
    edit_encoder = EditEncoder(embedding, edit_dim, norm_max, lamb_reg, norm_eps)

    # create decoder
    decoder_layer = DecoderLayer(d_model, c(self_attn), c(source_attn), c(ff), dropout)
    decoder = Decoder(decoder_layer, n, embedding, pos_encoding)

    # create generator
    generator = Generator(d_model, embedding)

    # create the model
    model = EditTransformer(
        encoder,
        edit_encoder,
        decoder,
        generator)

    if torch.cuda.is_available():
        model.cuda()

    return model
