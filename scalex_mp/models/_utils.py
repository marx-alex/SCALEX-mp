from typing import Union, List
from itertools import combinations

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


def get_activation_fn(activation: str) -> Union[nn.Module, None]:
    if isinstance(activation, str):
        activation = activation.lower()
    avail_act = ["tanhshrink", "tanh", "relu", "gelu", "sigmoid", "identity", None]
    assert (
        activation in avail_act
    ), f"'act' must be one of {avail_act}, instead got {activation}"

    if activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanhshrink":
        return nn.Tanhshrink()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "identity":
        return nn.Identity()
    return None


class DomainSpecificBatchNorm1d(nn.Module):
    """Domain-specific Batch Normalization
    """
    def __init__(self, num_features: int, num_domains: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Parameters
        ----------
        num_features : int
            Number of features or channels of the input
        num_domains : int
            Number of domains
        eps : float
            A value added to the denominator for numerical stability
        momentum : float
            The value used for the running_mean and running_var computation
        """
        super().__init__()
        self.num_domains = num_domains
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for _ in range(num_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)
        for i in range(self.num_domains):
            domain_mask = d == i

            if domain_mask.sum() > 1:
                out[domain_mask, :] = self.bns[i](x[domain_mask, :])
            else:
                out[domain_mask, :] = x[domain_mask, :]
        return out


class ArgsSequential(nn.Sequential):
    """Sequential module with multiple arguments.

    This module allows passing multiple arguments to all submodules.

    Parameters
    ----------
    *args
        `pytorch.nn.Module`
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Pass a tensor through all modules.

        Parameters
        ----------
        x : torch.Tensor
        *args
            Arguments are passed to every module

        Returns
        -------
            torch.Tensor
        """
        for module in self:
            x = module(x, *args)
        return x


class KLDLoss(nn.Module):
    """Kullback-Leibler Divergence Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, mu: torch.Tensor, var: torch.Tensor):
        """
        Args:
            mu: Mean
            var: Variance

        Returns:
            Kullback-Leibler Divergence
        """
        return kl_divergence(
            Normal(mu, var.sqrt()),
            Normal(torch.zeros_like(mu), torch.ones_like(var))
        ).sum(dim=1).mean()


# https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
def rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Gaussian kernel between two tensors.
    """
    x_n, y_n = x.size(0), y.size(0)
    n_dim = x.size(1)
    x = x.unsqueeze(1)  # [x_n, 1, n_dim]
    y = y.unsqueeze(0)  # [1, y_n, n_dim)
    tiled_x = x.expand(x_n, y_n, n_dim)
    tiled_y = y.expand(x_n, y_n, n_dim)
    return torch.exp(-(tiled_x - tiled_y).pow(2).mean(2) / float(n_dim))


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy Loss."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_mmd(x, y):
        x_kernel = rbf_kernel(x, x)
        y_kernel = rbf_kernel(y, y)
        xy_kernel = rbf_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Source features in Hilbert space.

        Returns:
            Maximum mean discrepancy.
        """
        loss = self._compute_mmd(torch.randn_like(x), x)
        return loss


class DomainMMDLoss(nn.Module):
    """Maximum Mean Discrepancy Loss between domains."""

    def __init__(self, num_domains: int, reduction: str = 'mean'):
        super().__init__()

        self.num_domains = num_domains
        avail_reductions = ['mean', 'sum']
        assert reduction in avail_reductions, f"reduction must be one of {avail_reductions}, instead got {reduction}"
        self.reduction = reduction

    @staticmethod
    def _partition(x: torch.Tensor, y: torch.Tensor, n_partitions) -> List[torch.Tensor]:
        partitions = []
        y = y.flatten()

        for i in range(n_partitions):
            mask = y == i
            partitions.append(x[mask, ...])

        return partitions

    @staticmethod
    def _compute_mmd(x, y):
        x_kernel = rbf_kernel(x, x)
        y_kernel = rbf_kernel(y, y)
        xy_kernel = rbf_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        Args:
            x: Source features in Hilbert space.
            d: Domains

        Returns:
            Maximum mean discrepancy between kernel embeddings of source and target.
        """
        loss = torch.tensor(0.0, device=x.device)

        partitions = self._partition(x, d, self.num_domains)
        partitions = [partition for partition in partitions if partition.shape[0] > 0]
        combs = list(combinations(list(range(self.num_domains)), r=2))
        for i, j in combs:
            loss += self._compute_mmd(partitions[i], partitions[j])

        if self.reduction == 'mean':
            if len(combs) > 0:
                loss = loss / len(combs)
        return loss


class MultiCoralLoss(nn.Module):
    """Coral Loss for two or more domains.

    References
    ----------
    [1] https://arxiv.org/pdf/1607.01719.pdf
    """

    def __init__(self, num_domains: int, reduction: str = 'mean'):
        super().__init__()

        self.num_domains = num_domains
        avail_reductions = ['mean', 'sum']
        assert reduction in avail_reductions, f"reduction must be one of {avail_reductions}, instead got {reduction}"
        self.reduction = reduction

    @staticmethod
    def _partition(x: torch.Tensor, y: torch.Tensor, n_partitions) -> List[torch.Tensor]:
        partitions = []
        y = y.flatten()

        for i in range(n_partitions):
            mask = y == i
            partitions.append(x[mask, ...])

        return partitions

    @staticmethod
    def _compute_coral(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        n_features = source.size(1)

        source_cov = torch.cov(source.T)
        target_cov = torch.cov(target.T)

        squared_frob = torch.sum((source_cov - target_cov)**2)

        return squared_frob / (4 * n_features * n_features)

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        Args:
            x: Source features in Hilbert space.
            d: Domains

        Returns:
            Maximum mean discrepancy between kernel embeddings of source and target.
        """
        loss = torch.tensor(0.0, device=x.device)

        partitions = self._partition(x, d, self.num_domains)
        partitions = [partition for partition in partitions if partition.shape[0] > 0]
        combs = list(combinations(list(range(self.num_domains)), r=2))
        for i, j in combs:
            loss += self._compute_coral(partitions[i], partitions[j])

        if self.reduction == 'mean':
            if len(combs) > 0:
                loss = loss / len(combs)
        return loss
