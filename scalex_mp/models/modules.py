from typing import Tuple, Optional

import torch
from torch import nn
from torch.distributions import Normal

from scalex_mp.models import DomainSpecificBatchNorm1d, ArgsSequential, get_activation_fn


class Block(nn.Module):
    """NN Block.

    Parameters
    ----------
    in_features : int
        Size of each input sample
    out_features : int
        Size of each output sample
    norm : str, optional
        'bn' for Batch normalization, 'dsbn' for domain-specific batch normalization
    n_batches : int
        Number of batches, is needed for domain-specific batch normalization
    act : str, optional
        Activation functions
    dropout : float
        Add dropout layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: Optional[str] = 'bn',
        n_batches: int = 1,
        act: Optional[str] = 'relu',
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True
        )

        if norm == 'bn':
            self.norm_layer = nn.BatchNorm1d(out_features)
        elif norm == 'dsbn':
            self.norm_layer = DomainSpecificBatchNorm1d(out_features, num_domains=n_batches)
        self.norm = norm

        self.act_layer = get_activation_fn(act)

        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self, x: torch.Tensor, d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[batch size, number of features]`
        d : torch.Tensor
            Tensor of shape `[batch size, 1]` with batch (domain) labels

        Returns
        -------
        torch.Tensor
        """
        x = self.fc(x)

        if self.norm == 'bn':
            x = self.norm_layer(x)
        elif self.norm == 'dsbn':
            x = self.norm_layer(x, d)

        if self.act_layer is not None:
            x = self.act_layer(x)

        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        return x


class Encoder(nn.Module):
    """Encoder Module.

    Parameters
    ----------
    layer_dims : list
        List with dimensions in hidden layers
    dropout : float
        Add dropout layer
    """

    def __init__(
        self,
        layer_dims: list,
        dropout: float = None,
    ) -> None:
        super().__init__()
        # dynamically append modules
        self.encode = None
        if len(layer_dims) > 1:
            self.encode = []

            for i, (in_features, out_features) in enumerate(
                zip(layer_dims[:-1], layer_dims[1:])
            ):

                self.encode.append(
                    Block(
                        in_features=in_features,
                        out_features=out_features,
                        norm='bn',
                        act='relu',
                        dropout=dropout
                    )
                )

            self.encode = nn.Sequential(*self.encode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[batch size, number of features]`

        Returns
        -------
        torch.Tensor
        """
        if self.encode is not None:
            return self.encode(x)
        return x


class VAEEncoder(Encoder):
    """Encoder Module for a Variational Autoencoder.

    Parameters
    ----------
    n_features : int
        Number of features of the input AnnData object
    latent_dim : int
        Number of latent features
    layer_dims : list
        List with dimensions in hidden layers
    dropout : float
        Add dropout layer
    """

    def __init__(
            self,
            n_features: int,
            latent_dim: int,
            layer_dims: list,
            dropout: float = None
    ):
        super().__init__(layer_dims=[n_features] + layer_dims, dropout=dropout)
        self.mu_encoder = nn.Linear(layer_dims[-1], latent_dim)
        self.var_encoder = nn.Linear(layer_dims[-1], latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass tensor through module.

        Parameters
        x : torch.Tensor
            Tensor of shape `[batch size, number of features]`

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Means and variances
        """
        if self.encode is not None:
            x = self.encode(x)

        mu = self.mu_encoder(x)
        var = torch.exp(self.var_encoder(x)) + 1e-5
        z = self.reparameterize(mu, var)
        return z, mu, var


class Decoder(nn.Module):
    """Decoder Module.

    Parameters
    ----------
    n_batches : int
        Number of batches, is needed for domain-specific batch normalization
    n_features : int
        Number of features of the input AnnData object
    latent_dim : int
        Number of latent features
    layer_dims : list
        List with dimensions in hidden layers
    recon_loss : str
        Reconstruction loss. Either `mse` for Mean Squared Error or `bce` for Binary Cross Entropy.
    dropout : float
        Add dropout layer
    """

    def __init__(
        self,
        n_batches: int,
        n_features: int,
        latent_dim: int,
        layer_dims: list,
        recon_loss: str = 'mse',
        dropout: float = None,
    ) -> None:
        super().__init__()
        layer_dims = [latent_dim] + layer_dims + [n_features]
        if recon_loss == 'mse':
            last_layer = 'identity'
        elif recon_loss == 'bce':
            last_layer = 'sigmoid'
        else:
            assert False, f'`recon_loss` must be either `mse` or `bce`, instead got {recon_loss}'

        # dynamically append modules
        self.decode = None
        if len(layer_dims) > 1:
            self.decode = []

            for i, (in_features, out_features) in enumerate(
                zip(layer_dims[:-1], layer_dims[1:])
            ):

                self.decode.append(
                    Block(
                        in_features=in_features,
                        out_features=out_features,
                        norm='dsbn',
                        n_batches=n_batches,
                        # gaussian assumption for x_hat
                        act='relu' if (i + 1) < (len(layer_dims) - 1) else last_layer,
                        dropout=dropout
                    )
                )

            self.decode = ArgsSequential(*self.decode)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[batch size, number of features]`
        d : torch.Tensor
            Tensor of shape `[batch size, 1]` with batch (domain) labels

        Returns
        -------
        torch.Tensor
        """
        if self.decode is not None:
            x = self.decode(x, d)
        return x
