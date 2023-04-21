SCALEX-mp
==============================

SCALEX is a Variational Autoencoder for the integration of single cell data from different batches.
It was originally designed for single cell sequencing data. SCALEX-mp is an adaption
for morphological (image-based) profiles. Pytorch Lightning and Weights and Biases for 
logging is used under the hood.

The main difference to the published SCALEX model is that the last decoder layer uses 
an identity transform as activation, because in contrast to count data morphological 
profiles are expected to be gaussian distributed.
Mean squared error is therefore used as reconstruction loss. The size and number of
hidden layers can be individually adapted.

## Installation
Install the latest version from GitHub:

    git clone git://github.com/marx-alex/SCALEX-mp.git
    cd SCALEX-mp
    pip install -r requirements.txt

## Usage

```
import anndata as ad
from scalex_mp import SCALEXLogic
```
Load the AnnData object with morphological profiles (typically normally distributed)
and initialize the SCALEX procedure.
The default parameters are set to the parameters from the original publication (see references).
```
adata = ad.read_h5ad('your_file.h5ad')

logic = SCALEXLogic(
    adata=adata,
    batch_key='batch',              # name of observation
    latent_dim=10,                  # number of dimensions of the latent space
    encoder_layer_dims=[1024],      # dimensions of the hidden layers in the encoder
    decoder_layer_dims=[],          # dimensions of the hidden layers in the decoder
    kld_beta=0.5,                   # coefficient of the KLD loss
    dropout=None,                   # dropout rate
    l2=5e-4,                        # l2 regularization
    batch_size=64,                  # batch size for training
    num_workers=4,                  # number of workers for training
    learning_rate=2e-4,             # learning rate for training
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
