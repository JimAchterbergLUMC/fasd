# FASD: Tabular Fidelity Agnostic Synthetic Data Generation 
![Alt text](images/image.png)

FASD is a Python package to generate fidelity agnostic (tabular) synthetic data. It directly optimizes synthetic data for high utility in predicting a target variable, without necessarily optimizing for high fidelity. 


# Motivation

Fidelity Agnostic Synthetic Data can be helpful in terms of privacy-preservation, since reducing fidelity can mitigate privacy concerns due to the inherent fidelity-privacy tradeoff in synthetic data. FASD computes supervised representations, after which it generates synthetic data in this representation space, and decodes these back to the original data space. By separating the supervised signal from the generative task - as in, e.g., supervised VAEs - only information relevant to the prediction is retained. 


# Key Features

- Support for regression and (multi-class) classification tasks
- Simple API to generate synthetic data in a few lines of code
- Built-in support for missing values, i.e., we randomly impute missings and synthesize an indicator column to reinstate missing values after generation


# Installation

`pip install fasd`

# Quick Start

```python 
from sklearn.datasets import load_breast_cancer
from fasd import TabularFASD
X = load_breast_cancer(as_frame=True).frame
generator = TabularFASD(target_column="target")
generator.fit(X)
syn = generator.generate(len(X))
```

See the tutorial.ipynb notebook for a tutorial. 

# Parameters

FASD consists of three separate neural networks, i.e., a predictor MLP, a Variational Auto-Encoder, and a decoder MLP. We present a unified vocabulary to set the network architectures and training parameters for all three of these networks as follows:

`[network]_[parameter]`

So, for example, to set the amount of training epochs, the corresponding parameter is `predictor_epochs` for the predictor MLP, `vae_epochs` for the Variational Auto-Encoder, and `decoder_epochs` for the decoder MLP. 


## Initialization:
Here we present a list of all parameters (and default values) which FASD takes as input during initialization:
- `target_column` (str): the name of the target column
- `representation_dim` (int): dimension of the supervised representations, which corresponds to the final hidden layer of the predictor MLP (default=100)
- `predictor_hidden_layers` (list): nodes of hidden layers in the predictor excluding the final hidden layer (which is already given by representation_dim) (default=[ ])
- `predictor_nonlin` (str): nonlinear activation function in the predictor (default="relu")
- `representations_nonlin` (str): nonlinear activation function at the final hidden layer of the predictor MLP (default="tanh")
- `predictor_dropout` (float): dropout in the predictor MLP (default=0)
- `predictor_epochs` (int): training epochs for the predictor MLP (default=100)
- `predictor_batch_size` (int): batch size for the predictor MLP (default=512)
- `predictor_lr` (float): learning rate for the predictor MLP (default=1e-3)
- `predictor_opt_betas` (tuple): exponential decay rates of moving averages in Adam optimizer in the predictor MLP (default=(0.9, 0.999))
- `predictor_weight_decay` (float): weight decay in Adam optimizer in the predictor MLP (default=1e-3)
- `predictor_early_stopping` (bool): whether to implement early stopping during training of the predictor MLP (default=True)
- `predictor_n_iter_min` (int): minimum number of epochs to train the predictor MLP (default=10)
- `predictor_patience` (int): number of epochs without improvement on a validation set before early stopping training of the predictor MLP, if early stopping is set to True (default=50)
- `predictor_clipping_value` (float): gradient clipping value in the predictor MLP (0 disables gradient clipping) (default=0)
- `decoder_hidden_layers` (list): list of hidden layer nodes in the decoder MLP (default=[100])
- `decoder_nonlin` (str): nonlinear activations in hidden layers of decoder MLP (default='relu')
- `decoder_dropout` (float): dropout in the decoder MLP (default=0)
- `decoder_epochs` (int): training epochs for the decoder MLP (default=100)
- `decoder_batch_size` (int): batch size for the decoder MLP (default=512)
- `decoder_lr` (float): learning rate for the decoder MLP (default=1e-3)
- `decoder_opt_betas` (tuple): exponential decay rates of moving averages in Adam optimizer in the decoder MLP (default=(0.9, 0.999))
- `decoder_weight_decay` (float): weight decay in Adam optimizer in the decoder MLP (default=1e-3)
- `decoder_early_stopping` (bool): whether to implement early stopping during training of the decoder MLP (default=True)
- `decoder_n_iter_min` (int): minimum number of epochs to train the decoder MLP (default=10)
- `decoder_patience` (int): number of epochs without improvement on a validation set before early stopping training of the decoder MLP, if early stopping is set to True (default=50)
- `decoder_clipping_value` (float): gradient clipping value in the decoder MLP (0 disables gradient clipping) (default=0)
- `vae_encoder_hidden_layers` (list): list of hidden layer nodes in the VAE encoder (default=[128,128])
- `vae_encoder_nonlin` (str): nonlinear activations in hidden layers of VAE encoder (default='relu')
- `vae_encoder_dropout` (float): dropout in the VAE encoder (default=0)
- `vae_decoder_hidden_layers` (list): list of hidden layer nodes in the VAE decoder (default=[128,128])
- `vae_decoder_nonlin` (str): nonlinear activations in hidden layers of VAE decoder (default='relu')
- `vae_decoder_dropout` (float): dropout in the VAE decoder (default=0)
- `vae_embedding_size` (int): embedding size in the VAE (default=128)
- `vae_loss_factor` (float): loss factor in β-VAE, i.e., factor to multiply reconstruction loss (default=1)
- `vae_epochs` (int): training epochs for the VAE (default=100)
- `vae_batch_size` (int): batch size for the VAE (default=512)
- `vae_lr` (float): learning rate for the VAE (default=1e-3)
- `vae_opt_betas` (tuple): exponential decay rates of moving averages in Adam optimizer in the VAE (default=(0.9, 0.999))
- `vae_weight_decay` (float): weight decay in Adam optimizer in the VAE (default=1e-3)
- `vae_early_stopping` (bool): whether to implement early stopping during training of the VAE (default=True)
- `vae_n_iter_min` (int): minimum number of epochs to train the VAE (default=10)
- `vae_patience` (int): number of epochs without improvement on a validation set before early stopping training of the VAE, if early stopping is set to True (default=50)
- `vae_clipping_value` (float): gradient clipping value in the VAE (0 disables gradient clipping) (default=0)

## Fitting:
Here we present a list of all parameters which FASD takes as input when fitting:
- `X` (pd.DataFrame): dataframe of real data to base synthetic data generation on
- `discrete_features` (list, optional): list of discrete feature names. If none are provided, FASD automatically detects non-numerical features as discrete, along with any numerical feature which has less than 10 unique values (default=None)

## Generating:
Here we present a list of all parameters which FASD takes as input when generating:
- `n` (int): number of synthetic samples to generate 


# Citing

If you use this package, please cite the corresponding paper:
```bibtex
@article{achterberg2025fidelity,
  title={Fidelity-agnostic synthetic data generation improves utility while retaining privacy},
  author={Achterberg, Jim and Haas, Marcel and van Dijk, Bram and Spruit, Marco},
  journal={Patterns},
  year={2025},
  publisher={Elsevier}
}
```