SCALEX-mp
==============================

SCALEX is a Variational Autoencoder for the integration of single cell data from different batches.
It was originally designed for single cell sequencing data. SCALEX-md is an adaption
for morphological (image-based) profiles. Pytorch Lightning and Weights and Biases for 
logging is used under the hood.

## Installation
Install the latest version from GitHub:

    pip install git+https://github.com/marx-alex/SCALEX-md.git

Or git clone and install:

    git clone git://github.com/marx-alex/SCALEX-md.git
    cd SCALEX-md
    python setup.py install

## Usage

```
import anndata as ad
from scalex_mp import SCALEXLogic
```
Load the AnnData object with morphological profiles (typically normally distributed)
and initialize the SCALEX procedure.
```
adata = ad.read_h5ad('your_file.h5ad')

logic = SCALEXLogic(
    adata=adata,
    batch_key='batch',              # name of observation
    latent_dim=10,                  # number of dimensions of the latent space
    encoder_layer_dims=[512, 256],  # dimensions of the hidden layer in the encoder
    decoder_layer_dims=[256, 512],  # dimensions of the hidden layer in the decoder
    kld_beta=0.001,                 # coefficient of the KLD loss
    dropout=0.1,                    # dropout rate
    l2=1e-4,                        # l2 regularization
    batch_size=32,                  # batch size for training
    num_workers=4,                  # number of workers for training
    learning_rate=1e-4,             # learning rate for training
    optimizer="Adam",               # `Adam` or `AdamW` optimizer
)
```
If you want to use Weights and Biases for logging, log-in before training.
```
wandb login
```
Start training.
```
logic.fit(max_epochs=20, wandb_log=True)
```
Load a pretrained model.
```
model_path = "./checkpoints/last_model.ckpt"
logic = SCALEXLogic.from_checkpoint(model_path, adata=adata)
```
Load the latent features into your AnnData object.
```
adata = logic.get_latent()
```

## Reference

    Xiong, L., Tian, K., Li, Y. et al. Online single-cell data integration through projecting heterogeneous 
    datasets into a common cell-embedding space. 
    Nat Commun 13, 6118 (2022). https://doi.org/10.1038/s41467-022-33758-z
