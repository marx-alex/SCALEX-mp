from typing import Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder

from scalex_mp.models import SCALEX
from scalex_mp.data import AnnDataModule


class SCALEXLogic:
    """SCALEX Logic class for convenient training and integration.

    Parameters
    ----------
    adata : anndata.AnnData or str
        Path to AnnData object or AnnData object
    batch_key : str, optional
        Batch variable
    latent_dim : int
        Number of latent features
    encoder_layer_dims : list, optional
        Dimensions of hidden layers in the encoder part.
        The length of the sequences it equal to the number of hidden layers.
        The default is `[512, 256]`.
    decoder_layer_dims : list, optional
        Dimensions of hidden layers in the decoder part.
        The length of the sequences it equal to the number of hidden layers.
        The default is `[256, 512]`.
    kld_beta : float
        Coefficient for the KLD-Loss
    dropout : float
        Dropout rate
    l2 : float
        L2 regularization
    batch_size : int
        Size of mini batches
    num_workers : int
        Number of subprocesses
    learning_rate : float
        Learning rate during training
    optimizer : str
        Optimizer for the training process.
        Can be `Adam` or `AdamW`.
    model : pytorch_lightning.LightningModule
        Pretrained Module
    """

    def __init__(
        self,
        adata: ad.AnnData,
        batch_key: str = 'batch',
        latent_dim: int = 10,
        encoder_layer_dims: Optional[list] = None,
        decoder_layer_dims: Optional[list] = None,
        kld_beta: float = 1,
        dropout: float = 0.1,
        l2: float = 1e-4,
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
        model: Optional[pl.LightningModule] = None
    ):
        # load data
        encoder_batch = LabelEncoder()
        encoder_batch.fit(adata.obs[batch_key])

        encoders = {
            "obs": {
                batch_key: encoder_batch.transform,
            }
        }

        self.data = AnnDataModule(
            adata,
            batch_key=batch_key,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataloader_opts=dict(convert=encoders),
            test_dataloader_opts=dict(convert=encoders),
        )

        if model is None:
            self.model = SCALEX(
                n_features=self.data.n_features,
                n_batches=self.data.n_batches,
                latent_dim=latent_dim,
                encoder_layer_dims=encoder_layer_dims,
                decoder_layer_dims=decoder_layer_dims,
                dropout=dropout,
                l2=l2,
                kld_beta=kld_beta,
                learning_rate=learning_rate,
                optimizer=optimizer,
            )
        else:
            self.model = model

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, adata: ad.AnnData, **kwargs):
        """Load Logic from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        adata : anndata.AnnData or str
        Path to AnnData object or AnnData object
        """
        model = SCALEX.load_from_checkpoint(checkpoint_path=checkpoint_path)
        return cls(adata=adata, model=model, **kwargs)

    def fit(
        self,
        logpath: Optional[str] = None,
        callbacks: Optional[pl.Callback] = None,
        wandb_log: bool = False,
        wandb_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Fits the model and creates checkpoint.

        Parameters
        ----------
        logpath : str
            Path where to store log files
        callbacks : list
            List of pytorch lightning callbacks.
            The model with the best loss and the last model is saved by default.
        wandb_log : bool
            Use WandB-Logger
        wandb_kwargs : dict
            Keyword arguments that are passed to WandbLogger
        **kwargs : dict
            Keyword arguments that are passed to pytorch_lightning.Trainer
        """
        if wandb_log:
            default_wandb_kwargs = dict(
                project="SCALEX",
                save_dir='.' if logpath is None else logpath,
            )
            if wandb_kwargs is not None:
                default_wandb_kwargs.update(wandb_kwargs)

            logger = WandbLogger(**default_wandb_kwargs)
        else:
            logger = False

        default_trainer_kwargs = dict(max_epochs=1000, log_every_n_steps=1)

        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="loss", min_delta=0.005, patience=50
                ),
                ModelCheckpoint(
                    dirpath=logpath,
                    filename="{epoch}_best_loss",
                    monitor="loss",
                ),
                ModelCheckpoint(
                    dirpath=logpath,
                    filename="last_model",
                )
            ]

        default_trainer_kwargs.update(dict(logger=logger, callbacks=callbacks))
        default_trainer_kwargs.update(kwargs)

        trainer = pl.Trainer(**default_trainer_kwargs)
        trainer.fit(self.model, self.data)

        if wandb_log:
            logger.experiment.finish()

    def get_latent(self) -> Union[ad.AnnData, dict]:
        """Stores latent representation under `.obsm['X_latent']`

        Returns
        -------
        anndata.AnnData
        """
        latent = np.zeros((len(self.data.adata), self.model.hparams.latent_dim))

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data.test_dataloader(), desc="Get latent"):
                x = batch['x']
                idx = batch['idx']
                z = self.model.forward_features(x, return_statistics=False).cpu().detach().numpy()
                latent[idx, :] = z
        self.model.train()

        self.data.adata.obsm['X_latent'] = latent
        return self.data.adata
