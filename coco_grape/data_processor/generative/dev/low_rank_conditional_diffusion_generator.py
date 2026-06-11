import math
import os
import sys
import logging
import warnings
import contextlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from sklearn.base import BaseEstimator, TransformerMixin


# --- Utility Context Manager ---
@contextlib.contextmanager
def suppress_output():
    """
    Context manager to temporarily suppress stdout and stderr.
    
    This can be useful during operations that generate unwanted console output.
    """
    # Open a null device to redirect output.
    with open(os.devnull, 'w') as devnull:
        # Save current stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            # Redirect stdout and stderr to devnull
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            # Restore original stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# --- Variational Autoencoder (VAE) for y ---
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for modeling the conditioning information y.
    
    This VAE encodes input y into a latent space and then decodes it back.
    
    Attributes:
        leaky_relu_slope (float): Slope parameter for LeakyReLU activations.
        encoder (nn.Sequential): Sequential encoder network.
        fc_mu (nn.Linear): Fully-connected layer to compute the mean of the latent distribution.
        fc_logvar (nn.Linear): Fully-connected layer to compute the log variance.
        decoder (nn.Sequential): Sequential decoder network.
        fc_out (nn.Linear): Output layer to reconstruct y.
    """
    def __init__(self, y_dim, latent_dim=32, hidden_dims=[128, 64], leaky_relu_slope=0.2):
        super(VAE, self).__init__()
        self.leaky_relu_slope = leaky_relu_slope
        
        # Build encoder network.
        encoder_modules = []
        last_dim = y_dim
        for h_dim in hidden_dims:
            encoder_modules.append(nn.Linear(last_dim, h_dim))
            encoder_modules.append(nn.LeakyReLU(self.leaky_relu_slope))
            last_dim = h_dim
        self.encoder = nn.Sequential(*encoder_modules)
        # Fully connected layers to produce latent distribution parameters.
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)
        
        # Build decoder network.
        decoder_modules = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_modules.append(nn.Linear(last_dim, h_dim))
            decoder_modules.append(nn.LeakyReLU(self.leaky_relu_slope))
            last_dim = h_dim
        self.decoder = nn.Sequential(*decoder_modules)
        # Output layer to reconstruct y.
        self.fc_out = nn.Linear(last_dim, y_dim)
    
    def encode(self, y):
        """
        Encodes the input y into latent space parameters.
        
        Args:
            y (Tensor): Input tensor of conditioning information.
            
        Returns:
            mu (Tensor): Mean of the latent distribution.
            logvar (Tensor): Log variance of the latent distribution.
        """
        x = self.encoder(y)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (Tensor): Mean tensor.
            logvar (Tensor): Log variance tensor.
        
        Returns:
            z (Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decodes the latent vector z to reconstruct y.
        
        Args:
            z (Tensor): Latent vector.
            
        Returns:
            Tensor: Reconstructed y.
        """
        x = self.decoder(z)
        return self.fc_out(x)
    
    def forward(self, y):
        """
        Forward pass through the VAE.
        
        Args:
            y (Tensor): Input tensor of conditioning information.
            
        Returns:
            y_recon (Tensor): Reconstructed y.
            mu (Tensor): Mean of the latent distribution.
            logvar (Tensor): Log variance of the latent distribution.
        """
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)
        y_recon = self.decode(z)
        return y_recon, mu, logvar


# --- Low-Rank Network Components ---
############################################
# Custom low‐rank linear layer (LowRankLinear)
############################################
class LowRankLinear(nn.Module):
    """
    A linear layer whose weight matrix is factorized as A @ B,
    where A is (in_features x thin_size) and B is (thin_size x out_features).
    
    This factorization can reduce the number of parameters and computational cost.
    
    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        thin_size (int): Inner dimension for factorization.
        A (Parameter): Left factor matrix.
        B (Parameter): Right factor matrix.
        bias (Parameter or None): Optional bias parameter.
    """
    def __init__(self, in_features, out_features, thin_size, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.thin_size = thin_size
        self.A = nn.Parameter(torch.Tensor(in_features, thin_size))
        self.B = nn.Parameter(torch.Tensor(thin_size, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass of the low-rank linear layer.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output after applying the factorized linear transformation.
        """
        out = x @ self.A @ self.B  # Apply factorized weight multiplication
        if self.bias is not None:
            out = out + self.bias  # Add bias if applicable
        return out


############################################
# Residual block with dropout and LeakyReLU
############################################
class ResidualBlock(nn.Module):
    """
    A residual block that applies a low-rank linear layer, dropout,
    and LeakyReLU activation. If the input and output dimensions differ,
    a low-rank skip projection is applied.
    
    Attributes:
        linear (LowRankLinear): The main low-rank linear transformation.
        dropout (nn.Dropout): Dropout layer for regularization.
        activation (nn.LeakyReLU): LeakyReLU activation.
        skip (nn.Module): Identity or low-rank linear layer to match dimensions.
    """
    def __init__(self, in_features, out_features, thin_size, dropout_prob, negative_slope):
        super().__init__()
        self.linear = LowRankLinear(in_features, out_features, thin_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.LeakyReLU(negative_slope)
        
        # If dimensions differ, use a skip connection with projection.
        if in_features != out_features:
            self.skip = LowRankLinear(in_features, out_features, thin_size)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass of the residual block.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after residual addition.
        """
        out = self.linear(x)
        out = self.dropout(out)
        out = self.activation(out)
        # Add skip connection.
        return out + self.skip(x)


############################################
# LowRankMLP Network using Residual Blocks or Plain Layers
############################################
class LowRankMLPNet(nn.Module):
    """
    A multi-layer perceptron built from low-rank layers.
    
    Depending on the 'use_residual' flag, each hidden layer is either built
    as a residual block or a plain sequence of layers.
    
    Parameters:
      - input_dim (int): Dimension of the input.
      - output_dim (int): Dimension of the output.
      - hidden_layers (int): Number of hidden layers.
      - hidden_dim (int): Width of the hidden layers.
      - thin_size (int): Inner dimension for weight factorization.
      - dropout (float): Dropout probability.
      - negative_slope (float): Negative slope for LeakyReLU activation.
      - use_residual (bool): Flag to use residual blocks.
    """
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_dim, thin_size,
                 dropout, negative_slope, use_residual=True):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.use_residual = use_residual
        layers = []
        # Build the first hidden layer.
        if hidden_layers > 0:
            if use_residual:
                layers.append(ResidualBlock(input_dim, hidden_dim, thin_size, dropout, negative_slope))
            else:
                layers.append(nn.Sequential(
                    LowRankLinear(input_dim, hidden_dim, thin_size),
                    nn.LeakyReLU(negative_slope),
                    nn.Dropout(dropout)
                ))
            # Build remaining hidden layers.
            for _ in range(hidden_layers - 1):
                if use_residual:
                    layers.append(ResidualBlock(hidden_dim, hidden_dim, thin_size, dropout, negative_slope))
                else:
                    layers.append(nn.Sequential(
                        LowRankLinear(hidden_dim, hidden_dim, thin_size),
                        nn.LeakyReLU(negative_slope),
                        nn.Dropout(dropout)
                    ))
            self.blocks = nn.ModuleList(layers)
            # Output layer to map hidden representation to desired output.
            self.out_layer = LowRankLinear(hidden_dim, output_dim, thin_size)
        else:
            # If no hidden layers, directly map input to output.
            self.out_layer = LowRankLinear(input_dim, output_dim, thin_size)
        
    def forward(self, x):
        """
        Forward pass of the MLP network.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output prediction.
        """
        if self.hidden_layers > 0:
            for block in self.blocks:
                x = block(x)
        return self.out_layer(x)


############################################
# Separate Low-Rank Embeddings for Each Input
############################################

class LowRankInputEmbedding(nn.Module):
    """
    A low-rank linear layer for embedding the noisy input x.
    
    This module reduces the dimensionality of x using a factorized linear transformation.
    """
    def __init__(self, input_dim, input_emb_dim, thin_size):
        super().__init__()
        self.lowrank = LowRankLinear(input_dim, input_emb_dim, thin_size)
    
    def forward(self, x):
        """
        Forward pass to compute the input embedding.
        
        Args:
            x (Tensor): Noisy input tensor.
            
        Returns:
            Tensor: Embedded representation of x.
        """
        return self.lowrank(x)

class LowRankConditioningEmbedding(nn.Module):
    """
    A low-rank module for transforming the conditioning information y.
    
    Applies a low-rank linear transformation, followed by activation and dropout.
    """
    def __init__(self, y_dim, cond_emb_dim, thin_size, dropout=0.2, negative_slope=0.2):
        super().__init__()
        self.lowrank = LowRankLinear(y_dim, cond_emb_dim, thin_size)
        self.activation = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, y):
        """
        Forward pass to compute the conditioning embedding.
        
        Args:
            y (Tensor): Conditioning information.
            
        Returns:
            Tensor: Embedded representation of y.
        """
        emb = self.lowrank(y)
        emb = self.activation(emb)
        emb = self.dropout(emb)
        return emb

class LowRankTimeEmbedding(nn.Module):
    """
    A low-rank module for transforming the time step t.
    
    This module converts a scalar time step into an embedding using a low-rank layer.
    """
    def __init__(self, time_emb_dim, thin_size, dropout=0.2, negative_slope=0.2):
        super().__init__()
        self.lowrank = LowRankLinear(1, time_emb_dim, thin_size)
        self.activation = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, t):
        """
        Forward pass to compute the time embedding.
        
        Args:
            t (Tensor): Tensor of time steps with shape (batch_size,).
            
        Returns:
            Tensor: Time embeddings.
        """
        # t has shape (batch_size,) -> unsqueeze to (batch_size, 1)
        t = t.unsqueeze(1).float()
        emb = self.lowrank(t)
        emb = self.activation(emb)
        emb = self.dropout(emb)
        return emb


############################################
# Final Diffusion Network with Separate Embeddings
############################################
class LowRankDiffusionNetSeparate(nn.Module):
    """
    Processes x, y, and t via separate low-rank embeddings and then
    concatenates them. The concatenated representation is passed 
    through a low-rank MLP to predict the denoised x.
    
    Parameters:
      - input_dim (int): Dimension of x.
      - y_dim (int): Dimension of conditioning y.
      - input_emb_dim (int): Embedding dimension for x.
      - cond_emb_dim (int): Embedding dimension for y.
      - time_emb_dim (int): Embedding dimension for t.
      - hidden_layers, hidden_dim, thin_size, dropout, negative_slope: Parameters for the final MLP.
      - use_residual (bool): Whether to use residual blocks in the MLP.
    """
    def __init__(self, input_dim, y_dim,
                 input_emb_dim, cond_emb_dim, time_emb_dim,
                 hidden_layers, hidden_dim, thin_size, dropout, negative_slope, use_residual=True):
        super().__init__()
        # Separate embeddings for each input type.
        self.input_emb = LowRankInputEmbedding(input_dim, input_emb_dim, thin_size)
        self.cond_emb = LowRankConditioningEmbedding(y_dim, cond_emb_dim, thin_size, dropout, negative_slope)
        self.time_emb = LowRankTimeEmbedding(time_emb_dim, thin_size, dropout, negative_slope)
        # Total dimension after concatenating embeddings.
        total_emb_dim = input_emb_dim + cond_emb_dim + time_emb_dim
        # Final MLP to predict the denoised x.
        self.net = LowRankMLPNet(total_emb_dim, input_dim, hidden_layers, hidden_dim, thin_size, dropout, negative_slope, use_residual)
    
    def forward(self, x, t, y):
        """
        Forward pass through the diffusion network.
        
        Args:
            x (Tensor): Noisy input.
            t (Tensor): Time step indices.
            y (Tensor): Conditioning information.
            
        Returns:
            Tensor: Predicted noise (denoising output).
        """
        x_emb = self.input_emb(x)
        y_emb = self.cond_emb(y)
        t_emb = self.time_emb(t)
        # Concatenate embeddings along the feature dimension.
        combined = torch.cat([x_emb, y_emb, t_emb], dim=1)
        return self.net(combined)


# --- Conditional Diffusion Model ---
class ConditionalDiffusionModel(pl.LightningModule, BaseEstimator, TransformerMixin):
    """
    A conditional diffusion model implemented using PyTorch Lightning.
    
    This model combines a diffusion network with a variational autoencoder (VAE)
    to perform conditional generation and denoising of input data.
    
    The model includes separate low-rank embeddings for the noisy input (x),
    conditioning information (y), and time steps (t), which are then concatenated
    and passed through a low-rank MLP.
    
    It also includes methods for training (with a diffusion loss and a VAE reconstruction loss),
    validation, and generating samples.
    
    Parameters:
      - mlp_num_layers (int): Number of hidden layers in the MLP.
      - mlp_hidden_dim (int): Hidden dimension for the MLP.
      - mlp_thin_size (int): Inner dimension for low-rank factorization in the MLP.
      - mlp_dropout_rate (float): Dropout rate in the MLP.
      - mlp_use_residual (bool): Whether to use residual blocks in the MLP.
      - mlp_leaky_relu_slope (float): Negative slope for LeakyReLU in the MLP.
      - time_emb_dim (int): Embedding dimension for the time step.
      - cond_emb_dim (int): Embedding dimension for conditioning information.
      - input_emb_dim (int): Embedding dimension for the input.
      - vae_latent_dim (int): Latent dimension for the VAE.
      - vae_hidden_dims (list of int): Hidden dimensions for the VAE encoder/decoder.
      - lr (float): Learning rate.
      - betas (tuple): Betas for the Adam optimizer.
      - num_steps (int): Number of diffusion steps.
      - batch_size (int): Batch size for training.
      - max_epochs (int): Maximum number of training epochs.
      - vae_recon_loss_weight (float): Weight for the VAE reconstruction loss.
      - mask_tradeoff (float): Weight tradeoff when using a feature mask.
      - verbose (bool): Whether to output verbose logging.
      - enable_progress_bar (bool): Whether to enable a rich progress bar.
      - feature_mask (iterable or None): Optional feature mask for the diffusion loss.
      - flattened_indices (iterable or None): Indices used for additional embedding loss.
      - embedding_loss_weight (float): Weight for the embedding loss.
      - low_variance_threshold (float): Threshold to remove low-variance features.
    """
    def __init__(self, 
                 mlp_num_layers=1,
                 mlp_hidden_dim=128,
                 mlp_thin_size=128 // 4,
                 mlp_dropout_rate=0.2,
                 mlp_use_residual=True,
                 mlp_leaky_relu_slope=0.2,
                 time_emb_dim=128,
                 cond_emb_dim=128,
                 input_emb_dim=128,
                 vae_latent_dim=128 // 4,
                 vae_hidden_dims=[128, 128 // 2],
                 lr=1e-4,
                 betas=(0.9, 0.999),
                 num_steps=500,
                 batch_size=64,
                 max_epochs=250,
                 vae_recon_loss_weight=2.0,
                 mask_tradeoff=0.5,
                 verbose=False,
                 enable_progress_bar=False,
                 feature_mask=None,
                 flattened_indices=None,         # New parameter for additional embedding loss.
                 embedding_loss_weight=1.0,        # New parameter for additional embedding loss.
                 low_variance_threshold=1e-6
                ):
        super(ConditionalDiffusionModel, self).__init__()
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        pl.LightningModule.__init__(self)
        
        # Save hyperparameters.
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_thin_size = mlp_thin_size
        self.mlp_dropout_rate = mlp_dropout_rate
        self.mlp_use_residual = mlp_use_residual
        self.mlp_leaky_relu_slope = mlp_leaky_relu_slope
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim
        self.input_emb_dim = input_emb_dim

        self.lr = lr
        self.betas = betas
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.vae_recon_loss_weight = vae_recon_loss_weight
        self.verbose = verbose
        self.enable_progress_bar = enable_progress_bar
        self.mask_tradeoff = mask_tradeoff
        self.feature_mask = feature_mask
        self.low_variance_threshold = low_variance_threshold
        
        # Store flattened_indices as a torch.LongTensor (if provided).
        self.flattened_indices = None
        if flattened_indices is not None: 
            self.set_flattened_indices(flattened_indices)
        self.embedding_loss_weight = embedding_loss_weight

        # Placeholders for sub-models and dimensions.
        self.model = None
        self.vae = None
        self.input_dim = None         # Reduced input dimension after low-variance feature removal.
        self.full_input_dim = None    # Original full input dimension.
        self.y_dim = None

        # Diffusion schedule: linearly spaced beta values.
        beta_start = 1e-4
        beta_end = 0.02
        beta_schedule = torch.linspace(beta_start, beta_end, num_steps)
        self.register_buffer('beta_schedule', beta_schedule)
        self.alpha = 1.0 - beta_schedule
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alpha, dim=0))

        # Buffers for scaling (computed in fit).
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))

        # Placeholders for low variance features (computed during fit).
        self.low_variance_mask = None
        self.low_variance_means = None

        # Verbose logging: store training and validation losses.
        if self.verbose:
            self.train_losses = []
            self.val_losses = []
            self.current_train_losses = []
            self.current_val_losses = []

        # Suppress logging if not verbose.
        if not self.verbose:
            logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
            logging.getLogger("lightning").setLevel(logging.CRITICAL)
            logging.getLogger("pytorch_lightning").propagate = False
            logging.getLogger("lightning").propagate = False

    def set_flattened_indices(self, flattened_indices):
        """
        Set the flattened indices used for the additional embedding loss.
        
        Args:
            flattened_indices (iterable): Indices to be used for gathering features.
        """
        self.flattened_indices = torch.tensor(flattened_indices, dtype=torch.long)

    def training_step(self, batch, batch_idx):
        """
        Defines a single training step.
        
        This method performs:
          - Data scaling.
          - Forward pass through the VAE and diffusion network.
          - Calculation of VAE reconstruction loss, KL loss, and diffusion loss.
          - Optionally, additional embedding loss.
        
        Args:
            batch (tuple): Tuple containing inputs x and conditioning y.
            batch_idx (int): Batch index.
            
        Returns:
            total_loss (Tensor): Combined loss for backpropagation.
        """
        x, y = batch
        # Normalize input using pre-computed mean and std.
        x_scaled = (x - self.mean) / self.std

        # VAE forward pass for conditioning information y.
        y_recon, mu, logvar = self.vae(y)
        recon_loss = F.mse_loss(y_recon, y, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss

        # Sample a random diffusion time step for each sample.
        t = torch.randint(0, self.num_steps, (x_scaled.size(0),), device=self.device).long()
        # Add noise to the input based on the diffusion process.
        x_noisy, noise = self.forward_diffusion(x_scaled, t)
        # Predict the noise using the diffusion network.
        noise_pred = self.model(x_noisy, t, y)

        # Compute diffusion loss (with or without feature mask).
        if self.feature_mask is not None:
            se = (noise_pred - noise) ** 2
            mask = self.feature_mask.to(noise_pred.device).unsqueeze(0)
            num_masked = mask.sum()
            num_unmasked = (1 - mask).sum()
            masked_loss = (se * mask).sum() / (num_masked * noise_pred.size(0)) if num_masked > 0 else 0.0
            unmasked_loss = (se * (1 - mask)).sum() / (num_unmasked * noise_pred.size(0)) if num_unmasked > 0 else 0.0
            diffusion_loss = self.mask_tradeoff * masked_loss + (1 - self.mask_tradeoff) * unmasked_loss
        else:
            diffusion_loss = F.mse_loss(noise_pred, noise, reduction='mean')

        total_loss = diffusion_loss + self.vae_recon_loss_weight * vae_loss

        # Additional embedding loss using flattened_indices.
        if self.flattened_indices is not None:
            # Compute the denoised prediction in the reduced normalized space.
            sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(1)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(1)
            x_pred = (x_noisy - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
            # Convert the prediction back to the full feature space.
            x_pred_full = self.restore_low_variance_features(x_pred * self.std + self.mean)
            # Use flattened_indices to gather specific features.
            indices = self.flattened_indices.to(x_pred_full.device)
            gathered = x_pred_full[:, indices.view(-1)]
            gathered = gathered.view(x_pred_full.shape[0], indices.shape[0], indices.shape[1])
            x_pred_sum = gathered.sum(dim=2)  # Sum over nodes.
            embedding_loss = F.mse_loss(x_pred_sum, y, reduction='mean')
            total_loss = total_loss + self.embedding_loss_weight * embedding_loss
            if self.verbose:
                self.log('train_embedding_loss', embedding_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.verbose:
            # Log all relevant losses.
            self.log('train_diffusion_loss', diffusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_vae_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_vae_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.current_train_losses.append(total_loss.item())

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single validation step.
        
        Similar to training_step but without backpropagation. It computes and logs
        the diffusion loss, VAE reconstruction loss, KL loss, and optionally the embedding loss.
        
        Args:
            batch (tuple): Tuple containing inputs x and conditioning y.
            batch_idx (int): Batch index.
            
        Returns:
            total_loss (Tensor): Combined loss.
        """
        x, y = batch
        x_scaled = (x - self.mean) / self.std

        y_recon, mu, logvar = self.vae(y)
        recon_loss = F.mse_loss(y_recon, y, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss

        t = torch.randint(0, self.num_steps, (x_scaled.size(0),), device=self.device).long()
        x_noisy, noise = self.forward_diffusion(x_scaled, t)
        noise_pred = self.model(x_noisy, t, y)

        if self.feature_mask is not None:
            se = (noise_pred - noise) ** 2
            mask = self.feature_mask.to(noise_pred.device).unsqueeze(0)
            num_masked = mask.sum()
            num_unmasked = (1 - mask).sum()
            masked_loss = (se * mask).sum() / (num_masked * noise_pred.size(0)) if num_masked > 0 else 0.0
            unmasked_loss = (se * (1 - mask)).sum() / (num_unmasked * noise_pred.size(0)) if num_unmasked > 0 else 0.0
            diffusion_loss = self.mask_tradeoff * masked_loss + (1 - self.mask_tradeoff) * unmasked_loss
        else:
            diffusion_loss = F.mse_loss(noise_pred, noise, reduction='mean')

        total_loss = diffusion_loss + self.vae_recon_loss_weight * vae_loss

        # Compute additional embedding loss (if applicable).
        if self.flattened_indices is not None:
            sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(1)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(1)
            x_pred = (x_noisy - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
            x_pred_full = self.restore_low_variance_features(x_pred * self.std + self.mean)
            indices = self.flattened_indices.to(x_pred_full.device)
            gathered = x_pred_full[:, indices.view(-1)]
            gathered = gathered.view(x_pred_full.shape[0], indices.shape[0], indices.shape[1])
            x_pred_sum = gathered.sum(dim=2)
            embedding_loss = F.mse_loss(x_pred_sum, y, reduction='mean')
            total_loss = total_loss + self.embedding_loss_weight * embedding_loss
            self.log('val_embedding_loss', embedding_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log validation losses.
        self.log('val_diffusion_loss', diffusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_vae_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_vae_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.verbose:
            self.current_val_losses.append(total_loss.item())

        return total_loss

    def linear_beta_schedule(self, timesteps):
        """
        Generates a linear beta schedule over the given number of timesteps.
        
        Args:
            timesteps (int): Number of diffusion steps.
            
        Returns:
            Tensor: Linearly spaced beta values.
        """
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    def forward_diffusion(self, x0, t):
        """
        Applies the forward diffusion process to add noise to the input.
        
        Args:
            x0 (Tensor): Original input tensor.
            t (Tensor): Time steps for each sample.
            
        Returns:
            tuple: (x_noisy, noise) where x_noisy is the noisy version of x0 and noise is the added noise.
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise

    def reverse_diffusion_step(self, x, t, y):
        """
        Performs a single reverse diffusion step.
        
        Uses the model's prediction to update x toward denoising.
        
        Args:
            x (Tensor): Current noisy input.
            t (Tensor): Current time step indices.
            y (Tensor): Conditioning information.
            
        Returns:
            Tensor: Updated x after one reverse diffusion step.
        """
        noise_pred = self.model(x, t, y)
        beta_t = self.beta_schedule[t].view(-1, 1).to(x.device)
        alpha_t = self.alpha[t].view(-1, 1).to(x.device)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1).to(x.device)

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
        # If t > 0, add random noise; else, no noise.
        noise = torch.randn_like(x) if t[0].item() > 0 else torch.zeros_like(x)
        updated_x = coef1 * (x - coef2 * noise_pred) + torch.sqrt(beta_t) * noise
        return updated_x

    def remove_low_variance_features(self, X):
        """
        Removes features from X that have variance below a specified threshold.
        
        Args:
            X (Tensor): Input tensor with features.
            
        Returns:
            tuple: (X_reduced, low_var_mask) where X_reduced is X with low-variance features removed
                   and low_var_mask is a boolean mask indicating which features were removed.
        """
        var = torch.var(X, dim=0, unbiased=False)
        low_var_mask = var < self.low_variance_threshold
        self.low_variance_means = torch.mean(X, dim=0)[low_var_mask]
        X_reduced = X[:, ~low_var_mask]
        return X_reduced, low_var_mask

    def restore_low_variance_features(self, X_reduced):
        """
        Restores the full feature space by reintroducing the low-variance features.
        
        Args:
            X_reduced (Tensor): Tensor with low-variance features removed.
            
        Returns:
            Tensor: Full tensor with original dimensions, where low-variance features are set to their stored means.
        """
        n_samples = X_reduced.shape[0]
        full_dim = self.full_input_dim
        X_full = torch.empty(n_samples, full_dim, device=X_reduced.device, dtype=X_reduced.dtype)
        X_full[:, ~self.low_variance_mask] = X_reduced
        stored_means = self.low_variance_means.unsqueeze(0).expand(n_samples, -1)
        X_full[:, self.low_variance_mask] = stored_means
        return X_full

    def process_data(self, X, y):
        """
        Processes and converts input data to torch tensors.
        
        Args:
            X (numpy.ndarray or Tensor): Input features.
            y (numpy.ndarray or Tensor): Conditioning information.
            
        Returns:
            tuple: (X_tensor, y_tensor) as float tensors.
            
        Raises:
            ValueError: If input types are not numpy arrays or torch tensors.
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X.astype(np.float32))
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise ValueError("Input X must be a numpy array or a torch tensor.")
        if y is not None:
            if isinstance(y, np.ndarray):
                y_tensor = torch.tensor(y.astype(np.float32))
            elif isinstance(y, torch.Tensor):
                y_tensor = y.float()
            else:
                raise ValueError("Input y must be a numpy array or a torch tensor.")
        else:
            raise ValueError("Conditioning information y must be provided for conditional diffusion.")
        # If a low variance mask exists, adjust the input dimensions accordingly.
        if self.low_variance_mask is not None:
            if X_tensor.shape[1] == self.full_input_dim:
                X_tensor = X_tensor[:, ~self.low_variance_mask]
            elif X_tensor.shape[1] != self.input_dim:
                raise ValueError(f"Unexpected input dimension {X_tensor.shape[1]}; expected either full dim {self.full_input_dim} or reduced dim {self.input_dim}.")
        return X_tensor, y_tensor

    def setup_model(self, input_dim, y_dim):
        """
        Instantiates and sets up the diffusion network and VAE if not already done.
        
        Args:
            input_dim (int): Dimension of the (reduced) input x.
            y_dim (int): Dimension of conditioning information y.
        """
        if self.model is None:
            self.model = LowRankDiffusionNetSeparate(
                input_dim=input_dim,
                y_dim=y_dim,
                input_emb_dim=self.input_emb_dim,
                cond_emb_dim=self.cond_emb_dim,
                time_emb_dim=self.time_emb_dim,
                hidden_layers=self.mlp_num_layers,
                hidden_dim=self.mlp_hidden_dim,
                thin_size=self.mlp_thin_size,
                dropout=self.mlp_dropout_rate,
                negative_slope=self.mlp_leaky_relu_slope,
                use_residual=self.mlp_use_residual
            )
            self.model.to(self.device)
            if self.verbose:
                print(f'LowRankDiffusionNetSeparate instantiated with output_dim: {input_dim}')
        if self.vae is None:
            self.vae = VAE(
                y_dim=y_dim,
                latent_dim=self.vae_latent_dim,
                hidden_dims=self.vae_hidden_dims,
                leaky_relu_slope=self.mlp_leaky_relu_slope
            )
            self.vae.to(self.device)

    def on_train_epoch_end(self):
        """
        Callback at the end of a training epoch.
        
        If verbose, stores the average training loss for the epoch.
        """
        if self.verbose and self.current_train_losses:
            epoch_loss = sum(self.current_train_losses) / len(self.current_train_losses)
            self.train_losses.append(epoch_loss)
            self.current_train_losses = []

    def on_validation_epoch_end(self):
        """
        Callback at the end of a validation epoch.
        
        If verbose, stores the average validation loss for the epoch.
        """
        if self.verbose and self.current_val_losses:
            epoch_loss = sum(self.current_val_losses) / len(self.current_val_losses)
            self.val_losses.append(epoch_loss)
            self.current_val_losses = []

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        
        Returns:
            Optimizer: Adam optimizer configured with the specified learning rate and betas.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

    def fit(self, X, y=None, **fit_params):
        """
        Fits the conditional diffusion model to the provided data.
        
        This method performs:
          - Data processing and scaling.
          - Removal of low-variance features.
          - Setup of the diffusion network and VAE.
          - Splitting the data into training (and optionally validation) sets.
          - Training using a PyTorch Lightning Trainer.
        
        Args:
            X (numpy.ndarray or Tensor): Input features.
            y (numpy.ndarray or Tensor): Conditioning information.
            **fit_params: Additional parameters to be passed.
            
        Returns:
            self: Fitted model instance.
            
        Raises:
            ValueError: If conditioning information y is not provided.
        """
        if y is None:
            raise ValueError("Conditioning information y must be provided for conditional diffusion.")
        X_tensor, y_tensor = self.process_data(X, y)
        self.full_input_dim = X_tensor.shape[1]
        # Compute full mean and std before removing low-variance features.
        full_mean = torch.mean(X_tensor, dim=0, keepdim=True)
        full_std = torch.std(X_tensor, dim=0, keepdim=True)
        X_tensor, self.low_variance_mask = self.remove_low_variance_features(X_tensor)
        self.input_dim = X_tensor.shape[1]
        # Save scaling factors for the reduced feature space.
        self.mean = full_mean[:, ~self.low_variance_mask]
        self.std = full_std[:, ~self.low_variance_mask]
        # Adjust feature_mask if provided.
        if self.feature_mask is not None:
            if len(self.feature_mask) == self.full_input_dim:
                self.feature_mask = torch.tensor(self.feature_mask, dtype=torch.float32)[~self.low_variance_mask]
            elif len(self.feature_mask) != self.input_dim:
                raise ValueError(f"Length of feature_mask ({len(self.feature_mask)}) must equal input_dim ({self.input_dim}).")
        normalized_X = (X_tensor - self.mean) / self.std
        self.y_dim = y_tensor.shape[1]
        self.setup_model(self.input_dim, self.y_dim)
        dataset = TensorDataset(X_tensor, y_tensor)
        # Configure progress bar and callbacks based on verbosity.
        if self.enable_progress_bar:
            progress_bar = RichProgressBar(refresh_rate=10)
            callbacks = [progress_bar]
        else:
            callbacks = []
        if self.verbose:
            val_size = int(0.2 * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            enable_progress_bar = self.enable_progress_bar
        else:
            train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_dataloader = None
            enable_progress_bar = False
        if not self.verbose:
            logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
            logging.getLogger("lightning").setLevel(logging.CRITICAL)
        # Choose accelerator based on available hardware.
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1
        trainer = pl.Trainer(
            max_epochs=self.max_epochs, 
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=True,
            overfit_batches=0.0,
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=10
        )
        # Start training with or without validation dataloader.
        if self.verbose:
            trainer.fit(self, train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self, train_dataloader)
        return self

    def transform(self, X, y=None, normalized=False):
        """
        Transforms the input data by denoising via reverse diffusion.
        
        Args:
            X (numpy.ndarray or Tensor): Input features.
            y (numpy.ndarray or Tensor): Conditioning information.
            normalized (bool): Whether X is already normalized.
            
        Returns:
            numpy.ndarray: Generated, denoised output.
            
        Raises:
            ValueError: If the model has not been fitted or if dimensions mismatch.
        """
        if self.model is None or self.vae is None or self.input_dim is None or self.y_dim is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")
        if y is None:
            raise ValueError("Conditioning information y must be provided for conditional diffusion.")
        X_tensor, y_tensor = self.process_data(X, y)
        if not normalized:
            x_scaled = (X_tensor - self.mean) / self.std
        else:
            x_scaled = X_tensor
        x_scaled = x_scaled.to(self.device)
        y_tensor = y_tensor.to(self.device)
        if X_tensor.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {X_tensor.shape[1]}.")
        if y_tensor.shape[1] != self.y_dim:
            raise ValueError(f"Conditioning dimension mismatch. Expected {self.y_dim}, got {y_tensor.shape[1]}.")
        if X_tensor.shape[0] != y_tensor.shape[0]:
            raise ValueError(f"Number of samples in X ({X_tensor.shape[0]}) and y ({y_tensor.shape[0]}) must match.")
        self.model.eval()
        with torch.no_grad():
            batch_size = x_scaled.size(0)
            generated = x_scaled.clone()
            # Reverse diffusion process: iterate backwards over time steps.
            for step in reversed(range(self.num_steps)):
                t_tensor = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
                generated = self.reverse_diffusion_step(generated, t_tensor, y_tensor)
            # Rescale the output back to original scale.
            generated = generated * self.std + self.mean
            if self.low_variance_mask is not None:
                generated = self.restore_low_variance_features(generated)
            return generated.cpu().numpy()

    def predict(self, y):
        """
        Generates samples conditioned on the provided y.
        
        Args:
            y (numpy.ndarray or Tensor): Conditioning information.
            
        Returns:
            numpy.ndarray: Generated samples.
        """
        return self.conditional_sample(y)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits the model to X and y, then transforms X.
        
        Args:
            X (numpy.ndarray or Tensor): Input features.
            y (numpy.ndarray or Tensor): Conditioning information.
            **fit_params: Additional parameters for fit.
            
        Returns:
            numpy.ndarray: Transformed (denoised) output.
        """
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def conditional_sample(self, y, num_samples=None):
        """
        Generates samples conditioned on y using the diffusion process.
        
        Args:
            y (numpy.ndarray or Tensor): Conditioning information.
            num_samples (int or None): Number of samples to generate. If y contains a single sample,
                                       it is repeated num_samples times.
            
        Returns:
            numpy.ndarray: Generated samples.
            
        Raises:
            ValueError: If the model is not fitted or if dimensions mismatch.
        """
        if self.model is None or self.vae is None or self.input_dim is None or self.y_dim is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")
        if isinstance(y, np.ndarray):
            y = y.astype(np.float32)
            y_tensor = torch.tensor(y)
        elif isinstance(y, torch.Tensor):
            y_tensor = y.float()
        else:
            raise ValueError("Input y must be a numpy array or a torch tensor.")
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(0)
        if num_samples is not None:
            if y_tensor.size(0) == 1:
                y_tensor = y_tensor.repeat(num_samples, 1)
            elif y_tensor.size(0) != num_samples:
                raise ValueError(f"Number of samples in y ({y_tensor.size(0)}) does not match num_samples ({num_samples}).")
        else:
            num_samples = y_tensor.size(0)
        if y_tensor.shape[1] != self.y_dim:
            raise ValueError(f"Conditioning dimension mismatch. Expected {self.y_dim}, got {y_tensor.shape[1]}.")
        # Start with random noise in the input space.
        x_noise = torch.randn(num_samples, self.input_dim).to(self.device)
        generated = self.transform(x_noise.detach().cpu().numpy(),
                                   y_tensor.detach().cpu().numpy(),
                                   normalized=True)
        return generated

    def sample(self, num_samples=1):
        """
        Generates samples by first decoding a latent vector through the VAE,
        then sampling using the conditional diffusion process.
        
        Args:
            num_samples (int): Number of samples to generate.
            
        Returns:
            tuple: (x_generated_np, y_generated_np) where x_generated_np is the generated sample
                   and y_generated_np is the conditioning information from the VAE.
        """
        if self.vae is None or self.model is None or self.input_dim is None or self.y_dim is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")
        self.vae.eval()
        with torch.no_grad():
            # Sample latent variables and decode them to get y.
            z = torch.randn(num_samples, self.vae_latent_dim).to(self.device)
            y_generated = self.vae.decode(z)
            y_generated_np = y_generated.cpu().numpy()
            # Generate corresponding x using conditional sampling.
            x_generated_np = self.conditional_sample(y_generated_np, num_samples=num_samples)
            return x_generated_np, y_generated_np

    def on_train_end(self):
        """
        Callback at the end of training.
        
        If verbose, plots the training and validation losses over epochs.
        """
        if not self.verbose:
            return
        train_len = len(self.train_losses)
        val_len = len(self.val_losses)
        min_length = min(train_len, val_len)
        if min_length == 0:
            print("No training or validation losses recorded.")
            return
        if train_len != val_len:
            print(f"Warning: Train losses length ({train_len}) != Validation losses length ({val_len}). Trimming to minimum length ({min_length}).")
            self.train_losses = self.train_losses[:min_length]
            self.val_losses = self.val_losses[:min_length]
        skip_first = 5 if min_length > 5 else 0
        trimmed_train_losses = self.train_losses[skip_first:]
        trimmed_val_losses = self.val_losses[skip_first:]
        epochs = range(skip_first + 1, min_length + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, trimmed_train_losses, label='Train Loss')
        plt.plot(epochs, trimmed_val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()
