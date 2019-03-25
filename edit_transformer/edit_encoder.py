import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EditEncoder(nn.Module):
    """Encode editions using a noisy version of the sum of insert and delete.

    Attributes:
        embedding (nn.Embedding): embedding layer to convert indices into word embeddings.
        linear_prenoise (nn.Linear): linear projection without bias used before introducing noise.

    """
    def __init__(self, embedding: nn.Embedding, edit_dim: int = 128, norm_max: float = 14.0, kappa_init: float = 100.0,
                 norm_eps: float = 0.1) -> None:
        """Initialize the edit encoder.

        Args:
            embedding (nn.Embedding): embedding layer to convert indices into word embeddings.
            edit_dim (int): dimension of the final edit vector.
            norm_max (float): scalar used to rescale the norm samples (corresponds to the maximum norm).
            kappa_init (float): control the dispersion of the vMF sampling
            norm_eps (float): epsilon used to sample the random noise added to the norm of the edit_vector

        """
        self.embedding = embedding
        word_dim = embedding.shape[-1]
        self.linear_prenoise = nn.Linear(word_dim, edit_dim // 2, bias=False)
        self.norm_max = norm_max
        self.noise_scaler = kappa_init
        self.norm_eps = norm_eps
        self.normclip = nn.Hardtanh(0, self.norm_max - norm_eps)

    def forward(self, insert: Tensor, delete: Tensor, draw_samples: bool = True, draw_p: bool = False) -> Tensor:
        """Forward through the edit encoder.

        Args:
            insert (Tensor): tensor of insertions of shape `(batch, insert_seq_len, 1)`.
            delete (Tensor): tensor of deletions of shape `(batch, delete_seq_len, 1)`.
            draw_samples (bool): Weather to draw samples VAE style or not (keep True for training).
            draw_p (bool): Edit vector drawn from random prior distribution (keep False for training).

        Returns:
            Tensor: edit embedding vector of shape `(batch, d_edit)`.

        """
        insert_embed = self.embedding(insert)
        delete_embed = self.embedding(delete)

        insert_embed.sum_(dim=-2)
        delete_embed.sum_(dim=-2)

        insert_set = self.linear_prenoise(insert_embed)
        delete_set = self.linear_prenoise(delete_embed)

        combined_map = torch.cat([insert_set, delete_set], 1)
        if draw_samples:
            if draw_p:
                batch_size, edit_dim = combined_map.size()
                combined_map = self.draw_p_noise(batch_size, edit_dim)
            else:
                combined_map = self.sample_vMF(combined_map, self.noise_scaler)
        edit_embed = combined_map
        return edit_embed

    def draw_p_noise(self, batch_size: int, edit_dim: int) -> Tensor:
        """ Sample `batch_size` vector of size `edit_dim` from the prior distribution.

        Args:
            batch_size (int): number of vector to sample.
            edit_dim (float): dimension of the vector to sample.

        Returns:
            Tensor: sampled tensor matrice of shape `(batch, edit_dim)`.

        """
        rand_draw = torch.randn(batch_size, edit_dim, device=device)
        rand_draw = rand_draw / torch.norm(rand_draw, p=2, dim=1, keepdim=True).expand(batch_size, edit_dim)
        rand_norms = (torch.rand(batch_size, 1, device=device) * self.norm_max).expand(batch_size, edit_dim)
        return rand_draw * rand_norms

    def add_norm_noise(self, munorm: Tensor, eps: float) -> Tensor:
        """ Add noise on the norm of the edit vector.

        Notes:
            KL loss is - log(maxvalue/eps)
            cut at maxvalue-eps, and add [0,eps] noise.

        Args:
            munorm (Tensor): norm of a sample expanded of shape `(2*word_dim)`.
            eps (float): epsilon of the distribbution from which to random sample.

        Returns:
            Tensor: noisy version of the norm of the edit vector of shape `(2*word_dim)`.

        """
        trand = torch.rand(1, device=device).expand(munorm.size())*eps
        return self.normclip(munorm) + trand

    def sample_vMF(self, mu: Tensor, kappa: float):
        """vMF sampler in pytorch.

        http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python

        Args:
            mu (Tensor): of shape `(batch_size, 2*word_dim)`.
            kappa (float): controls dispersion. kappa of zero is no dispersion.

        Returns:
            Tensor: noisy version of mu of shape `(batch_size, 2*word_dim)`.

        """
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            munoise = self.add_norm_noise(munorm, self.norm_eps)
            if mu[i].norm().item() > 1e-10:
                # sample offset from center (on sphere) with spread kappa
                w = self._sample_weight(kappa, id_dim)
                wtorch = w * torch.ones(id_dim, device=device)

                # sample a point v on the unit sphere that's orthogonal to mu
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)

                # compute new point
                scale_factr = torch.sqrt(torch.ones(id_dim, device=device) - torch.pow(wtorch,2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munoise
            else:
                rand_draw = torch.randn(id_dim, device=device)
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
                rand_norms = (torch.rand(1, device=device) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * rand_norms #mu[i]
            result_list.append(sampled_vec)

        return torch.stack(result_list, 0)

    def _sample_weight(self, kappa: float, dim: int) -> float:
        """Rejection sampling scheme for sampling distance from center on surface of the sphere.

        Args:
            kappa (float): dispersion parameter.
            dim (int): dimension of the sphere on which to sample.

        Returns:
            float: sampled weight

        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa) # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            z = np.random.beta(dim / 2., dim / 2.)  #concentrates towards 0.5 as d-> inf
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u): #thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return w

    def _sample_orthonormal_to(self, mu: Tensor, dim: int) -> Tensor:
        """Sample point on sphere orthogonal to mu.

        Args:
            mu (Tensor): vector on the unit sphere of shape `(word_dim*2)`.
            dim (int): dimension of the vector (word_dim*2).

        Returns:
            Tensor: orthonormal vector to mu of shape `(word_dim*2)`.

        """
        v = torch.randn(dim, device=device)
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)
