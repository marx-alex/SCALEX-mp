from typing import Union

import numpy as np
import anndata as ad
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class AnnDataModule(pl.LightningDataModule):
    """Data Loading of AnnData Views.

    Parameters
    ----------
    adata : anndata.AnnData or str
        Path to AnnData object or AnnData object
    batch_key : str, optional
        Batch variable
    batch_size : int
        Size of mini batches
    num_workers : int
        Number of subprocesses
    """

    def __init__(
        self,
        adata: Union[str, ad.AnnData],
        batch_key: str = "batch",
        batch_size: int = 64,
        num_workers: int = 0
    ):
        super().__init__()

        # dataset
        if isinstance(adata, str):
            adata = ad.read_h5ad(adata)
        self.dataset = AnnDataset(adata, batch_key=batch_key)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_features = adata.n_vars
        self.n_batches = len(adata.obs[batch_key].unique())
        self.batch_key = batch_key

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False
        )


class AnnDataset(Dataset):
    """Dataset with AnnData.

    Parameters
    ----------
    adata : anndata.AnnData or str
        AnnData object with single cell data
    batch_key : str, optional
        Batch variable
    """

    def __init__(self, adata: ad.AnnData, batch_key: str = 'batch'):
        adata.obs[batch_key] = adata.obs[batch_key].astype('category')
        self.adata = adata
        self.batch_key = batch_key

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        if isinstance(self.adata.X[idx], np.ndarray):
            x = self.adata.X[idx].squeeze()
        else:
            x = self.adata.X[idx].toarray().squeeze()
        # float32 precision is required by the linear layer
        x = x.astype('float32')
        domain = self.adata.obs[self.batch_key].cat.codes[idx]
        return x, domain, idx
