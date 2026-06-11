import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting
import warnings
import contextlib
import os
import sys
import logging


# === Utility Context Manager ===

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# === Variational Autoencoder (VAE) for y ===

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for modeling the conditioning information y.
    """
    def __init__(self, y_dim, latent_dim=32, hidden_dims=[128, 64], leaky_relu_slope=0.2):
        super(VAE, self).__init__()
        self.leaky_relu_slope = leaky_relu_slope
        
        # Encoder
        encoder_modules = []
        last_dim = y_dim
        for h_dim in hidden_dims:
            encoder_modules.append(nn.Linear(last_dim, h_dim))
            encoder_modules.append(nn.LeakyReLU(self.leaky_relu_slope))
            last_dim = h_dim
        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)
        
        # Decoder
        decoder_modules = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_modules.append(nn.Linear(last_dim, h_dim))
            decoder_modules.append(nn.LeakyReLU(self.leaky_relu_slope))
            last_dim = h_dim
        self.decoder = nn.Sequential(*decoder_modules)
        self.fc_out = nn.Linear(last_dim, y_dim)
    
    def encode(self, y):
        """
        Encodes the input y into latent space.
        """
        x = self.encoder(y)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decodes the latent vector z to reconstruct y.
        """
        x = self.decoder(z)
        return self.fc_out(x)
    
    def forward(self, y):
        """
        Forward pass through the VAE.
        """
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)
        y_recon = self.decode(z)
        return y_recon, mu, logvar

# === Modular Embedding Components ===

class TimeEmbedding(nn.Module):
    """
    Embeds the time step t into a higher-dimensional space.
    """
    def __init__(self, time_emb_dim, leaky_relu_slope=0.2):
        super(TimeEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, t):
        """
        Args:
            t (Tensor): Time steps tensor of shape (batch_size,).

        Returns:
            Tensor: Embedded time steps of shape (batch_size, time_emb_dim).
        """
        t = t.unsqueeze(-1).float()  # Shape: (batch_size, 1)
        return self.embedding(t)      # Shape: (batch_size, time_emb_dim)


class ConditioningEmbedding(nn.Module):
    """
    Embeds the conditioning information y into a higher-dimensional space.
    """
    def __init__(self, y_dim, cond_emb_dim, leaky_relu_slope=0.2):
        super(ConditioningEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(y_dim, cond_emb_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(cond_emb_dim, cond_emb_dim)
        )

    def forward(self, y):
        """
        Args:
            y (Tensor): Conditioning information tensor of shape (batch_size, y_dim).

        Returns:
            Tensor: Embedded conditioning information of shape (batch_size, cond_emb_dim).
        """
        return self.embedding(y)


class InputEmbedding(nn.Module):
    """
    Embeds the input vector X into a higher-dimensional space.
    """
    def __init__(self, input_dim, input_emb_dim):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, input_emb_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Embedded input tensor of shape (batch_size, input_emb_dim).
        """
        return self.embedding(x)

# === Generalized MLP Architecture ===

class ResidualBlock(nn.Module):
    """
    A residual block consisting of two linear layers with LeakyReLU activations and dropout.
    Optionally includes a residual (skip) connection.
    """
    def __init__(self, hidden_dim, dropout_rate, use_residual, leaky_relu_slope=0.2):
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation1 = nn.LeakyReLU(leaky_relu_slope)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.LeakyReLU(leaky_relu_slope)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the residual block.
        """
        residual = x
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.dropout2(out)
        if self.use_residual:
            out += residual
        return out


class GeneralizedMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) with configurable layers, activations,
    dropout, and optional residual connections.
    """
    def __init__(self, input_dim, y_dim, hidden_dim=256, time_emb_dim=128,
                 cond_emb_dim=128, input_emb_dim=128, output_dim=None,
                 num_layers=3, dropout_rate=0.1, use_residual=True, leaky_relu_slope=0.2):
        super(GeneralizedMLP, self).__init__()
        self.input_emb = InputEmbedding(input_dim, input_emb_dim)
        self.cond_emb = ConditioningEmbedding(y_dim, cond_emb_dim, leaky_relu_slope)
        self.time_emb = TimeEmbedding(time_emb_dim, leaky_relu_slope)
        
        # Total embedding dimension after concatenation
        total_emb_dim = input_emb_dim + cond_emb_dim + time_emb_dim

        # First layer
        layers = [nn.Linear(total_emb_dim, hidden_dim),
                  nn.LeakyReLU(leaky_relu_slope),
                  nn.Dropout(dropout_rate)]

        # Hidden layers with Residual Blocks
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout_rate, use_residual, leaky_relu_slope))

        # Output layer
        output_layer = nn.Linear(hidden_dim, output_dim if output_dim else input_dim)
        layers.append(output_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x, t, y):
        """
        Forward pass of the generalized MLP.
        """
        x_emb = self.input_emb(x)  # Shape: (batch_size, input_emb_dim)
        y_emb = self.cond_emb(y)    # Shape: (batch_size, cond_emb_dim)
        t_emb = self.time_emb(t)    # Shape: (batch_size, time_emb_dim)
        combined = torch.cat([x_emb, y_emb, t_emb], dim=1)  # Shape: (batch_size, total_emb_dim)
        return self.net(combined)  # Shape: (batch_size, output_dim)

# --- Learnable Normalization Module for Masked Features ---

class LearnableNormalizationForMaskedFeatures(nn.Module):
    def __init__(self, feature_mask, init_scale=1.0, init_shift=0.0, init_threshold=1.0):
        """
        Learn an affine transformation for the features indicated by feature_mask, 
        with a learnable threshold that softly clips extreme deviations.

        Args:
            feature_mask (array-like or Tensor): Boolean mask of shape (num_features,)
                                                 indicating which features to normalize.
            init_scale (float): Initial scaling factor.
            init_shift (float): Initial shifting factor.
            init_threshold (float): Initial threshold for clipping extreme deviations.
                                    Smaller values mean more aggressive clipping.
        """
        super(LearnableNormalizationForMaskedFeatures, self).__init__()
        # Register the feature mask as a buffer so it moves with the model
        self.register_buffer("feature_mask", torch.tensor(feature_mask, dtype=torch.bool))
        self.num_masked = self.feature_mask.sum().item()
        # Learnable parameters for scaling and shifting (only for masked features)
        self.scale = nn.Parameter(torch.ones(self.num_masked) * init_scale)
        self.shift = nn.Parameter(torch.zeros(self.num_masked) + init_shift)
        # Learnable threshold: controls how strongly extreme deviations are clipped.
        self.threshold = nn.Parameter(torch.ones(self.num_masked) * init_threshold)

    def forward(self, x):
        """
        Apply the learnable affine transformation to the masked features, then
        apply a scaled tanh nonlinearity to penalize extreme deviations.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_features)

        Returns:
            Tensor: Output tensor with the masked features normalized.
        """
        # Extract the masked features.
        x_masked = x[:, self.feature_mask]
        # Apply the learnable affine transformation.
        x_affine = self.scale * x_masked + self.shift
        # Apply a soft-clipping nonlinearity:
        # For each masked feature, the transformation f(u)= c * tanh(u / c)
        # ensures that for |u| << c the mapping is nearly linear,
        # while for |u| >> c, f(u) saturates.
        c = self.threshold
        x_norm = c * torch.tanh(x_affine / c)
        # Replace the masked features with the normalized values.
        x_out = x.clone()
        x_out[:, self.feature_mask] = x_norm
        return x_out

    def inverse(self, x):
        """
        Invert the learnable normalization on the masked features.
        (Note: because of the tanh nonlinearity, inversion requires using atanh,
         and inputs must lie within the open interval (-c, c).)
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, num_features)
            
        Returns:
            Tensor: Tensor with the inverse-transformation applied on the masked features.
        """
        x_masked = x[:, self.feature_mask]
        epsilon = 1e-6  # for numerical stability
        c = self.threshold
        # Clip input to avoid numerical issues with atanh.
        clipped = torch.clamp(x_masked / (c + epsilon), -0.999, 0.999)
        # Invert the tanh nonlinearity: atanh(x) recovers the pre-activation.
        x_affine = c * torch.atanh(clipped)
        # Invert the affine transformation.
        x_orig = (x_affine - self.shift) / (self.scale + epsilon)
        x_inv = x.clone()
        x_inv[:, self.feature_mask] = x_orig
        return x_inv


# --- Conditional Diffusion Model with Learnable Normalization ---
class ConditionalDiffusionModel(pl.LightningModule, BaseEstimator, TransformerMixin):
    def __init__(self, 
                 mlp_num_layers=3,
                 mlp_hidden_dim=256,
                 mlp_dropout_rate=0.1,
                 mlp_use_residual=True,
                 mlp_leaky_relu_slope=0.2,
                 time_emb_dim=128,
                 cond_emb_dim=128,
                 input_emb_dim=128,
                 vae_latent_dim=32,
                 vae_hidden_dims=[128, 64],
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 num_steps=1000,
                 batch_size=64,
                 max_epochs=100,
                 vae_recon_loss_weight=1.0,
                 verbose=False,
                 enable_progress_bar=False,
                 feature_mask=None,    # binary mask over features (e.g. degree feature positions)
                 mask_tradeoff=0.5):
        super(ConditionalDiffusionModel, self).__init__()
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        pl.LightningModule.__init__(self)

        # Hyperparameters for MLP and training
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_dropout_rate = mlp_dropout_rate
        self.mlp_use_residual = mlp_use_residual
        self.mlp_leaky_relu_slope = mlp_leaky_relu_slope
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim
        self.input_emb_dim = input_emb_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.lr = lr
        self.betas = betas
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.vae_recon_loss_weight = vae_recon_loss_weight
        self.verbose = verbose
        self.enable_progress_bar = enable_progress_bar

        # Parameters for feature masking in the diffusion loss
        self.feature_mask = feature_mask  # Expecting a binary mask (e.g. a list or numpy array)
        self.mask_tradeoff = mask_tradeoff

        # Placeholders for models and dimensions
        self.model = None
        self.vae = None
        self.input_dim = None
        self.y_dim = None

        # Diffusion schedule
        beta_schedule = self.linear_beta_schedule(num_steps)
        self.register_buffer('beta_schedule', beta_schedule)
        self.alpha = 1.0 - beta_schedule
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alpha, dim=0))

        # Buffers for scaling inputs
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))

        if self.verbose:
            self.train_losses = []
            self.val_losses = []
            self.current_train_losses = []
            self.current_val_losses = []

        if not self.verbose:
            logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
            logging.getLogger("lightning").setLevel(logging.CRITICAL)
            logging.getLogger("pytorch_lightning").propagate = False
            logging.getLogger("lightning").propagate = False

        # Initialize the learnable normalization module if a feature_mask is provided.
        self.learnable_norm = None
        self.set_learnable_norm()
        
    def set_learnable_norm(self):
        if self.feature_mask is not None:
            self.learnable_norm = LearnableNormalizationForMaskedFeatures(feature_mask=self.feature_mask, init_scale=0.1, init_shift=0.0, init_threshold=20)
            
    def linear_beta_schedule(self, timesteps):
        beta_start = 1e-4
        beta_end = 0.02
        beta_schedule = torch.linspace(beta_start, beta_end, timesteps)
        return beta_schedule

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise

    def reverse_diffusion_step(self, x, t, y):
        noise_pred = self.model(x, t, y)
        beta_t = self.beta_schedule[t].to(x.device)
        alpha_t = self.alpha[t].to(x.device)
        alpha_cumprod_t = self.alpha_cumprod[t].to(x.device)
        current_step = t[0].item()
        if current_step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        coef1 = (1 / torch.sqrt(alpha_t)).unsqueeze(1)
        coef2 = ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)).unsqueeze(1)
        updated_x = coef1 * (x - coef2 * noise_pred) + torch.sqrt(beta_t).unsqueeze(1) * noise
        return updated_x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Apply learnable normalization to x if available
        if self.learnable_norm is not None:
            x = self.learnable_norm(x)

        # Scale input data
        x_scaled = (x - self.mean) / self.std

        # --- VAE Forward Pass ---
        y_recon, mu, logvar = self.vae(y)
        recon_loss = F.mse_loss(y_recon, y, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss

        # --- Diffusion Process ---
        t = torch.randint(0, self.num_steps, (x_scaled.size(0),), device=self.device).long()
        x_noisy, noise = self.forward_diffusion(x_scaled, t)
        noise_pred = self.model(x_noisy, t, y)
        
        # Compute diffusion loss with feature masking if provided
        if self.feature_mask is not None:
            se = (noise_pred - noise) ** 2
            mask = self.feature_mask.to(noise_pred.device).unsqueeze(0)
            num_masked = mask.sum()
            num_unmasked = (1 - mask).sum()
            if num_masked > 0:
                masked_loss = (se * mask).sum() / (num_masked * noise_pred.size(0))
            else:
                masked_loss = 0.0
            if num_unmasked > 0:
                unmasked_loss = (se * (1 - mask)).sum() / (num_unmasked * noise_pred.size(0))
            else:
                unmasked_loss = 0.0
            diffusion_loss = self.mask_tradeoff * masked_loss + (1 - self.mask_tradeoff) * unmasked_loss
        else:
            diffusion_loss = F.mse_loss(noise_pred, noise, reduction='mean')

        total_loss = diffusion_loss + self.vae_recon_loss_weight * vae_loss

        if self.verbose:
            self.log('train_diffusion_loss', diffusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_vae_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_vae_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.current_train_losses.append(total_loss.item())

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.learnable_norm is not None:
            x = self.learnable_norm(x)
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
            if num_masked > 0:
                masked_loss = (se * mask).sum() / (num_masked * noise_pred.size(0))
            else:
                masked_loss = 0.0
            if num_unmasked > 0:
                unmasked_loss = (se * (1 - mask)).sum() / (num_unmasked * noise_pred.size(0))
            else:
                unmasked_loss = 0.0
            diffusion_loss = self.mask_tradeoff * masked_loss + (1 - self.mask_tradeoff) * unmasked_loss
        else:
            diffusion_loss = F.mse_loss(noise_pred, noise, reduction='mean')

        total_loss = diffusion_loss + self.vae_recon_loss_weight * vae_loss

        self.log('val_diffusion_loss', diffusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_vae_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_vae_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.verbose:
            self.current_val_losses.append(total_loss.item())

        return total_loss

    def on_train_epoch_end(self):
        if self.verbose and self.current_train_losses:
            epoch_train_loss = np.mean(self.current_train_losses)
            self.train_losses.append(epoch_train_loss)
            self.current_train_losses = []

    def on_validation_epoch_end(self):
        if self.verbose and self.current_val_losses:
            epoch_val_loss = np.mean(self.current_val_losses)
            self.val_losses.append(epoch_val_loss)
            self.current_val_losses = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

    def setup_model(self, input_dim, y_dim):
        if self.model is None:
            self.model = GeneralizedMLP(
                input_dim=input_dim, 
                y_dim=y_dim, 
                hidden_dim=self.mlp_hidden_dim, 
                time_emb_dim=self.time_emb_dim,
                cond_emb_dim=self.cond_emb_dim,
                input_emb_dim=self.input_emb_dim,
                output_dim=input_dim,
                num_layers=self.mlp_num_layers,
                dropout_rate=self.mlp_dropout_rate,
                use_residual=self.mlp_use_residual,
                leaky_relu_slope=self.mlp_leaky_relu_slope
            )
            self.model.to(self.device)
            if self.verbose:
                print(f'GeneralizedMLP output_dim: {self.model.net[-1].out_features}')
        if self.vae is None:
            self.vae = VAE(
                y_dim=y_dim,
                latent_dim=self.vae_latent_dim,
                hidden_dims=self.vae_hidden_dims,
                leaky_relu_slope=self.mlp_leaky_relu_slope
            )
            self.vae.to(self.device)

    def process_data(self, X, y):
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
            X_tensor = torch.tensor(X)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise ValueError("Input X must be a numpy array or a torch tensor.")
        if y is not None:
            if isinstance(y, np.ndarray):
                y = y.astype(np.float32)
                y_tensor = torch.tensor(y)
            elif isinstance(y, torch.Tensor):
                y_tensor = y.float()
            else:
                raise ValueError("Input y must be a numpy array or a torch tensor.")
        else:
            raise ValueError("Conditioning information y must be provided for conditional diffusion.")
        return X_tensor, y_tensor

    def fit(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("Conditioning information y must be provided for conditional diffusion.")
        X_tensor, y_tensor = self.process_data(X, y)
        self.input_dim = X_tensor.shape[1]
        self.y_dim = y_tensor.shape[1]
        if self.feature_mask is not None:
            if len(self.feature_mask) != self.input_dim:
                raise ValueError(f"Length of feature_mask ({len(self.feature_mask)}) must equal input_dim ({self.input_dim}).")
            self.feature_mask = torch.tensor(self.feature_mask, dtype=torch.float32)
        self.setup_model(self.input_dim, self.y_dim)
        self.mean = torch.mean(X_tensor, dim=0, keepdim=True)
        self.std = torch.std(X_tensor, dim=0, keepdim=True)
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        dataset = TensorDataset(X_tensor, y_tensor)
        if self.enable_progress_bar:
            from pytorch_lightning.callbacks import RichProgressBar
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
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
        else:
            accelerator = "cpu"
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
        if self.verbose:
            trainer.fit(self, train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self, train_dataloader)
        return self

    def on_train_end(self):
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
        import matplotlib.pyplot as plt
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

    def transform(self, X, y=None):
        if self.model is None or self.vae is None or self.input_dim is None or self.y_dim is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")
        if y is None:
            raise ValueError("Conditioning information y must be provided for conditional diffusion.")
        X_tensor, y_tensor = self.process_data(X, y)
        if self.learnable_norm is not None:
            X_tensor = self.learnable_norm(X_tensor)
        x_scaled = (X_tensor - self.mean) / self.std
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
            for step in reversed(range(self.num_steps)):
                t_tensor = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
                generated = self.reverse_diffusion_step(generated, t_tensor, y_tensor)
            generated = generated * self.std + self.mean
            if self.learnable_norm is not None:
                generated = self.learnable_norm.inverse(generated)
            return generated.cpu().numpy()

    def predict(self, y):
        return self.conditional_sample(y)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def conditional_sample(self, y, num_samples=None):
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
            else:
                if y_tensor.size(0) != num_samples:
                    raise ValueError(f"Number of samples in y ({y_tensor.size(0)}) does not match num_samples ({num_samples}).")
        else:
            num_samples = y_tensor.size(0)
        if y_tensor.shape[1] != self.y_dim:
            raise ValueError(f"Conditioning dimension mismatch. Expected {self.y_dim}, got {y_tensor.shape[1]}.")
        x_noise = torch.randn(num_samples, self.input_dim).to(self.device)
        if self.learnable_norm is not None:
            x_noise = self.learnable_norm(x_noise)
        x_noise_scaled = (x_noise - self.mean) / self.std
        generated = self.transform(x_noise_scaled.detach().cpu().numpy(), y_tensor.detach().cpu().numpy())

        return generated

    def sample(self, num_samples=1):
        if self.vae is None or self.model is None or self.input_dim is None or self.y_dim is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' first.")
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.vae_latent_dim).to(self.device)
            y_generated = self.vae.decode(z)
            y_generated_np = y_generated.cpu().numpy()
            x_generated_np = self.conditional_sample(y_generated_np, num_samples=num_samples)
            return x_generated_np, y_generated_np
