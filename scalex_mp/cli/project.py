import argparse
import logging
import os
from typing import Union

from scalex_mp import SCALEXLogic
from scalex_mp.cli._utils import str_to_bool
import anndata as ad

logger = logging.getLogger("projectSCALEX")
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def run(
        model_path: str,
        adata: Union[str, ad.AnnData],
        out_dir: str = ".",
        batch_key: str = 'batch',
        batch_size: int = 64,
        num_workers: int = 4,
        return_mean: bool = True
):
    """
    Project using a SCALEX model.
    """
    logic = SCALEXLogic.from_checkpoint(
        model_path,
        adata=adata,
        batch_key=batch_key,
        batch_size=batch_size,
        num_workers=num_workers)
    adata = logic.get_latent(return_mean=return_mean)

    fname = "adata_scalex"
    sfx = ".h5ad"
    v = 0
    if os.path.exists(os.path.join(out_dir, fname + sfx)):
        while os.path.exists(os.path.join(out_dir, f"{fname}_{v}" + sfx)):
            v += 1
        fname = f"{fname}_{v}"

    adata.write(os.path.join(out_dir, fname + sfx))


def main(args=None):
    """Implements the commandline tool to project from a SCALEX model."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(
        prog="projectSCALEX",
        description="Project from a SCALEX model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the checkpoint file",
        required=True
    )
    parser.add_argument(
        "--adata",
        type=str,
        help="Path to AnnData object or AnnData object",
        required=True
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="batch",
        help="Batch variable",
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
        "--return_mean",
        type=str_to_bool,
        default=True,
        help="Return the mean or a reparameterized sample of the latent space.",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(
        model_path=args.model_path,
        adata=args.adata,
        out_dir=args.out_dir,
        batch_key=args.batch_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_mean=args.return_mean
    )
