from typing import Union

import torch
from torch import nn
from torch.distributions import Normal


def get_activation_fn(activation: str) -> Union[nn.Module, None]:
    if isinstance(activation, str):
        activation = activation.lower()
    avail_act = ["tanhshrink", "tanh", "relu", "gelu", "sigmoid", None]
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
    return None


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    var = torch.exp(logvar) + eps
    return Normal(mu, var.sqrt()).rsample()


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
