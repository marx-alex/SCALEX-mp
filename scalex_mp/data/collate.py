class AnnDataCollator:
    """Collator class for the AnnDataModule.

    This class contains the pytorch collate logic, which
    can be initiated once and called for every batch.

    Parameters
    ----------
    batch_key : str, optional
        Batch variable
    """

    def __init__(
        self,
        batch_key: str = "batch",
    ) -> None:
        self.batch_key = batch_key

    def __call__(self, batch) -> dict:
        """Called for every batch.

        If AdataCollator is called, `x` and `batch` is returned as a tuple.

        Parameters
        ----------
        batch : AnnCollectionView
            Batch given by anndata.experimental.AnnLoader

        Returns
        -------
        dict
            Converted batch to dictionary
        """
        out = dict()

        out["x"] = batch.X
        out["idx"] = batch.oidx
        out['d'] = batch.obs[self.batch_key].long()
        return out
