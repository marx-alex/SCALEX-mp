import argparse
import logging
from typing import Optional, Union

from scalex_mp import SCALEXLogic
import anndata as ad

logger = logging.getLogger("trainSCALEX")
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def run(
    adata: Union[str, ad.AnnData],
    batch_key: str = 'batch',
    latent_dim: int = 10,
    encoder_layer_dims: Optional[list] = None,
    decoder_layer_dims: Optional[list] = None,
    beta: float = 0.5,
    beta_norm: bool = True,
    regul_loss: str = 'kld',
    recon_loss: str = 'mse',
    dropout: Optional[float] = None,
    l2: float = 5e-4,
    batch_size: int = 64,
    num_workers: int = 4,
    learning_rate: float = 2e-4,
    optimizer: str = "Adam",
    max_epochs: int = 100,
    wandb_log: bool = True
):
    """
    Train a SCALEX model.
    """

    logic = SCALEXLogic(
        adata=adata,
        batch_key=batch_key,
        latent_dim=latent_dim,
        encoder_layer_dims=encoder_layer_dims,
        decoder_layer_dims=decoder_layer_dims,
        beta=beta,
        beta_norm=beta_norm,
        regul_loss=regul_loss,
        recon_loss=recon_loss,
        dropout=dropout,
        l2=l2,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        optimizer=optimizer
    )
    logic.fit(max_epochs=max_epochs, wandb_log=wandb_log)


def main(args=None):
    """Implements the commandline tool to train a SCALEX model."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(
        prog="trainSCALEX",
        description="Train a SCALEX model."
    )

    parser.add_argument(
        "--adata",
        type=str,
        help="Path to AnnData object or AnnData object",
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="batch",
        help="Batch variable",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=10,
        help="Number of latent features",
    )
    parser.add_argument(
        "--encoder_layer_dims",
        nargs='*',
        type=int,
        default=[1024],
        help="Dimensions of hidden layers in the encoder part. "
             "The length of the sequences it equal to the number of hidden layers. "
             "The default is `[1024]`.",
    )
    parser.add_argument(
        "--decoder_layer_dims",
        nargs='*',
        type=int,
        default=[],
        help="Dimensions of hidden layers in the decoder part. "
             "The length of the sequences it equal to the number of hidden layers. The default is `[]`.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Coefficient for the KLD-Loss",
    )
    parser.add_argument(
        "--beta_norm",
        type=bool,
        default=False,
        help="Normalize KLD-loss beta",
    )
    parser.add_argument(
        "--regul_loss",
        type=str,
        default='kld',
        help="Regularization loss. "
             "Either 'kld' for Kullback-Leibler Divergence or `mmd` for Maximum Mean Discrepancy.",
    )
    parser.add_argument(
        "--recon_loss",
        type=str,
        default='mse',
        help="Reconstruction loss. Either `mse` for Mean Squared Error or `bce` for Binary Cross Entropy. "
             "If reconstruction loss is `mse`, identity transform is used for the activation of the last decoder layer."
             "Otherwise the sigmoid function is used. Use `bce` for data scaled between 0 and 1.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=5e-4,
        help="L2 regularization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Size of mini batches",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of subprocesses",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=64,
        help="Learning rate during training",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='Adam',
        help="Optimizer for the training process. Can be `Adam` or `AdamW`.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default='100',
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--wandb_log",
        type=bool,
        default='True',
        help="Log to Weights and Biases",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(
        adata=args.adata,
        batch_key=args.batch_key,
        latent_dim=args.latent_dim,
        encoder_layer_dims=args.encoder_layer_dims,
        decoder_layer_dims=args.decoder_layer_dims,
        beta=args.beta,
        beta_norm=args.beta_norm,
        regul_loss=args.regul_loss,
        recon_loss=args.recon_loss,
        dropout=args.dropout,
        l2=args.l2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        max_epochs=args.max_epochs,
        wandb_log=args.wandb_log
    )
