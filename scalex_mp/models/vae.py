from typing import Optional, Union, Tuple, List
from collections import defaultdict

import torch
from torch import nn
import pytorch_lightning as pl

from scalex_mp.models._utils import KLDLoss, DomainMMDLoss, MMDLoss
from scalex_mp.models.modules import VAEEncoder, Decoder


class SCALEX(pl.LightningModule):
    """Variational Autoencoder for batch integration of single cell data.

    Parameters
    ----------
    n_features : int
        Number of features of the input AnnData object
    n_batches : int
        Number of batches to integrate
    latent_dim : int
        Number of latent features
    encoder_layer_dims : list, optional
        Dimensions of hidden layers in the encoder part.
        The length of the sequences it equal to the number of hidden layers.
    decoder_layer_dims : list, optional
        Dimensions of hidden layers in the decoder part.
        The length of the sequences it equal to the number of hidden layers.
    beta : float
        Coefficient for the KLD-loss
    beta_norm : bool
        Normalize KLD-loss beta
    regul_loss : str
        Regularization loss. Either 'kld' for Kullback-Leibler Divergence or `mmd` for Maximum Mean Discrepancy.
    recon_loss : str
        Reconstruction loss. Either `mse` for Mean Squared Error or `bce` for Binary Cross Entropy.
    dropout : float
        Dropout rate
    l2 : float
        L2 regularization
    learning_rate : float
        Learning rate during training
    optimizer : str
        Optimizer for the training process.
        Can be `Adam` or `AdamW`.
    """
    def __init__(
        self,
        n_features: int,
        n_batches: int,
        latent_dim: int = 10,
        encoder_layer_dims: Optional[List[int]] = None,
        decoder_layer_dims: Optional[List[int]] = None,
        regul_loss: str = 'kld',
        recon_loss: str = 'mse',
        dropout: Optional[float] = None,
        l2: float = 5e-4,
        beta: float = 0.5,
        beta_norm: bool = True,
        learning_rate: float = 2e-4,
        optimizer: str = "Adam",
    ):
        super().__init__()

        self.save_hyperparameters()

        if encoder_layer_dims is None:
            encoder_layer_dims = [1024]

        if decoder_layer_dims is None:
            decoder_layer_dims = []

        # parts of the VAE
        self.encoder = VAEEncoder(
            n_features=n_features,
            latent_dim=latent_dim,
            layer_dims=encoder_layer_dims,
            dropout=dropout
        )
        self.decoder = Decoder(
            n_batches=n_batches,
            n_features=n_features,
            latent_dim=latent_dim,
            layer_dims=decoder_layer_dims,
            recon_loss=recon_loss,
            dropout=dropout
        )

        if recon_loss == 'mse':
            self.recon_loss_func = nn.MSELoss()
        elif recon_loss == 'bce':
            self.recon_loss_func = nn.BCELoss()
        else:
            assert False, f'`recon_loss` must be either `mse` or `bce`, instead got {recon_loss}'
        if regul_loss == 'kld':
            self.regul_loss_func = KLDLoss()
        elif regul_loss == 'mmd':
            self.regul_loss_func = MMDLoss()
        else:
            assert False, f'`regul_loss` must be either `kld` or `mmd`, instead got {regul_loss}'
        self.mmd = DomainMMDLoss(num_domains=n_batches)

    def forward_features(
            self, x: torch.Tensor, return_statistics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z, mu, var = self.encoder(x)
        if return_statistics:
            return z, mu, var
        return z

    def forward(
            self, x: torch.Tensor, d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, var = self.forward_features(x, return_statistics=True)
        x_hat = self.decoder(z, d=d)
        return z, mu, var, x_hat

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.l2)
        elif self.hparams.optimizer == "AdamW":
            opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.l2)
        else:
            assert False, f"{self.hparams.optimizer=} is not Adam or AdamW"

        return opt

    def _common_step(self, batch):
        x, d, _ = batch
        z, mu, var, x_hat = self(x, d=d)

        # loss
        recon_loss = self.recon_loss_func(x_hat, x)

        if self.hparams.beta_norm:
            beta = (self.hparams.beta * self.hparams.latent_dim) / self.hparams.n_features
        else:
            beta = self.hparams.beta

        if self.hparams.regul_loss == 'kld':
            regul_loss = self.regul_loss_func(mu, var) * beta
        else:
            regul_loss = self.regul_loss_func(z) * beta

        with torch.no_grad():
            mmd = self.mmd(mu, d=d)

        loss = recon_loss + regul_loss

        self.log_dict(
            {
                "loss": loss,
                "recon_loss": recon_loss,
                "regul_loss": regul_loss,
                "inter-batch-mmd": mmd
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return loss
