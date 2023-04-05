from typing import Union, Optional

import anndata as ad
import pytorch_lightning as pl
from anndata.experimental import AnnLoader
from scalex_mp.data import AnnDataCollator


class AnnDataModule(pl.LightningDataModule):
    """Data Loading of AnnData Views.

    Parameters
    ----------
    adata : anndata.AnnData or str
        Path to anndata object or anndata object
    batch_key : str, optional
        Batch variable
    batch_size : int
        Size of mini batches
    num_workers : int
        Number of subprocesses
    train_dataloader_opts : dict, optional
        Additional arguments for training dataloader
    test_dataloader_opts : dict, optional
        Additional arguments for testing dataloader
    """

    def __init__(
        self,
        adata: Union[str, ad.AnnData],
        batch_key: str = "batch",
        batch_size: int = 32,
        num_workers: int = 0,
        train_dataloader_opts: Optional[dict] = None,
        test_dataloader_opts: Optional[dict] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        if isinstance(adata, str):
            self.adata = ad.read_h5ad(adata)
        else:
            self.adata = adata
        # convert string obs to categories
        self.adata.strings_to_categoricals()
        self.n_features = adata.n_vars
        self.n_batches = len(adata.obs[batch_key].unique())

        # collator
        collator = AnnDataCollator(batch_key=batch_key)

        # defining options for data loaders
        self.train_dataloader_opts = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "shuffle": True,
            "collate_fn": collator,
        }

        self.test_dataloader_opts = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "shuffle": False,
            "collate_fn": collator,
        }

        if train_dataloader_opts is not None:
            self.train_dataloader_opts.update(train_dataloader_opts)

        if test_dataloader_opts is not None:
            self.test_dataloader_opts.update(test_dataloader_opts)

    def train_dataloader(self) -> AnnLoader:
        return AnnLoader(self.adata, **self.train_dataloader_opts)

    def test_dataloader(self) -> AnnLoader:
        return AnnLoader(self.adata, **self.test_dataloader_opts)
