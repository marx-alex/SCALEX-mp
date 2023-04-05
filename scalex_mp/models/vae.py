from typing import Optional, Union, Tuple, List

import torch
from torch import nn
import pytorch_lightning as pl

from scalex_mp.models import VAEEncoder, Decoder


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
    mmd_beta : float
        Coefficient for the MMD-Loss
    mmd_beta : float
        Coefficient for the KLD-Loss
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
        dropout: float = 0.1,
        l2: float = 1e-4,
        kld_beta: float = 1,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ):
        super().__init__()

        self.save_hyperparameters()

        if encoder_layer_dims is None:
            encoder_layer_dims = [512, 256]

        if decoder_layer_dims is None:
            decoder_layer_dims = [256, 512]

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
            dropout=dropout
        )

        self.mse = nn.MSELoss()

    def forward_features(self, x: torch.Tensor, return_statistics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z, mu, var = self.encoder(x)
        if return_statistics:
            return z, mu, var
        return z

    def forward(
            self, x: torch.Tensor, d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.forward_features(x, return_statistics=True)
        x_hat = self.decoder(z, d=d)
        return z, mu, logvar, x_hat

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
        x = batch['x']
        d = batch['d']
        self.log('batch_size', x.size()[0])

        z, mu, logvar, x_hat = self(x, d=d)

        # loss
        mse = self.mse(x_hat, x)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

        loss = mse + (kld * self.hparams.kld_beta)

        self.log_dict(
            {
                "loss": loss,
                "mse": mse,
                "kld": kld * self.hparams.kld_beta,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return loss
