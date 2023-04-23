SCALEX-mp
==============================

SCALEX is a Variational Autoencoder for the integration of single cell data from different batches.
It was originally designed for single cell sequencing data. SCALEX-mp is an adaption
for morphological (image-based) profiles. This implementation is based on
Pytorch Lightning and Weights and Biases for logging.

The main difference to the published SCALEX model is that the last decoder layer uses 
an identity transform as activation, because in contrast to count data morphological 
profiles are expected to be gaussian distributed.
Mean squared error is therefore used for the reconstruction loss. 

All hyperparameters can be individually adapted, including the size and number of
hidden layers. The reconstruction loss can also be changed to binary cross entropy as it 
is used in the original paper. Instead of the KLD-Loss in the regularization term the 
Maximum Mean Discrepancy can also be used to get a MMD-VAE.

## 💻 Installation
Install the latest version from GitHub:

    git clone git://github.com/marx-alex/SCALEX-mp.git
    cd SCALEX-mp
    pip install -r requirements.txt

## Usage

### 💾 Logging

All experiments can be logged with Weights and Biases.
Log in with your credentials before training.

```
wandb login
```

### 🚩 Main functions

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
    beta=0.5,                       # coefficient of the KLD loss
    beta_norm=False,                # normalization of the beta coefficient
    regul_loss='kld',               # regularization loss: either 'kld' or 'mmd'
    recon_loss='mse',               # reconstruction loss: either 'mse' or 'bce'
    dropout=None,                   # dropout rate
    l2=5e-4,                        # l2 regularization
    batch_size=64,                  # batch size for training
    num_workers=4,                  # number of workers for training
    learning_rate=2e-4,             # learning rate for training
    optimizer="Adam",               # `Adam` or `AdamW` optimizer
)
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

### Command Line Interface

There are two scripts that can be used from the command line.
`trainSCALEX` takes an AnnData object and trains a model. The training can be logged with 
Weights and Biases. `projectSCALEX` takes a pretrained model from a checkpoint file and
an AnnData object to project the data onto the latent space.

```
>> trainSCALEX -h

usage: trainSCALEX [-h] [--adata ADATA] [--batch_key BATCH_KEY] [--latent_dim LATENT_DIM] [--encoder_layer_dims [ENCODER_LAYER_DIMS ...]]
                   [--decoder_layer_dims [DECODER_LAYER_DIMS ...]] [--beta BETA] [--beta_norm BETA_NORM] [--regul_loss REGUL_LOSS] [--recon_loss RECON_LOSS] [--dropout DROPOUT]
                   [--l2 L2] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--learning_rate LEARNING_RATE] [--optimizer OPTIMIZER] [--max_epochs MAX_EPOCHS]
                   [--wandb_log WANDB_LOG]

Train a SCALEX model.

optional arguments:
  -h, --help            show this help message and exit
  --adata ADATA         Path to AnnData object or AnnData object
  --batch_key BATCH_KEY
                        Batch variable
  --latent_dim LATENT_DIM
                        Number of latent features
  --encoder_layer_dims [ENCODER_LAYER_DIMS ...]
                        Dimensions of hidden layers in the encoder part. The length of the sequences it equal to the number of hidden layers. The default is `[1024]`.
  --decoder_layer_dims [DECODER_LAYER_DIMS ...]
                        Dimensions of hidden layers in the decoder part. The length of the sequences it equal to the number of hidden layers. The default is `[]`.
  --beta BETA           Coefficient for the KLD-Loss
  --beta_norm BETA_NORM
                        Normalize KLD-loss beta
  --regul_loss REGUL_LOSS
                        Regularization loss. Either 'kld' for Kullback-Leibler Divergence or `mmd` for Maximum Mean Discrepancy.
  --recon_loss RECON_LOSS
                        Reconstruction loss. Either `mse` for Mean Squared Error or `bce` for Binary Cross Entropy. If reconstruction loss is `mse`, identity transform is used for
                        the activation of the last decoder layer.Otherwise the sigmoid function is used. Use `bce` for data scaled between 0 and 1.
  --dropout DROPOUT     Dropout rate
  --l2 L2               L2 regularization
  --batch_size BATCH_SIZE
                        Size of mini batches
  --num_workers NUM_WORKERS
                        Number of subprocesses
  --learning_rate LEARNING_RATE
                        Learning rate during training
  --optimizer OPTIMIZER
                        Optimizer for the training process. Can be `Adam` or `AdamW`.
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs
  --wandb_log WANDB_LOG
                        Log to Weights and Biases
```

```
>> projectSCALEX -h

usage: projectSCALEX [-h] [--model_path MODEL_PATH] [--adata ADATA] [--out_dir OUT_DIR] [--batch_key BATCH_KEY] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                     [--return_mean RETURN_MEAN]

Project from a SCALEX model.

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the checkpoint file
  --adata ADATA         Path to AnnData object or AnnData object
  --out_dir OUT_DIR     Output directory
  --batch_key BATCH_KEY
                        Batch variable
  --batch_size BATCH_SIZE
                        Size of mini batches
  --num_workers NUM_WORKERS
                        Number of subprocesses
  --return_mean RETURN_MEAN
                        Return the mean or a reparameterized sample of the latent space.
```

## 📃 Reference

    Xiong, L., Tian, K., Li, Y. et al. Online single-cell data integration through projecting heterogeneous 
    datasets into a common cell-embedding space. 
    Nat Commun 13, 6118 (2022). https://doi.org/10.1038/s41467-022-33758-z
