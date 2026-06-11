import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import contextlib, os, sys
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.base import BaseEstimator


# --- Modified Custom Robust Scaler ---
class CustomRobustScaler:
    """
    A custom robust scaler that performs feature-wise scaling using the median and IQR.
    For features with near-zero variance (IQR < epsilon), the scaler removes them.
    During fit, it computes and stores the median and IQR for the nonconstant features,
    as well as the means for the constant ones. The transform method returns only the
    scaled nonconstant features, and the inverse_transform method reconstructs the
    full-dimensional data by inserting the constant features (filled with their mean)
    in their original positions.
    """
    def __init__(self, quantile_range=(5.0, 95.0), epsilon=1e-8):
        self.quantile_range = quantile_range
        self.epsilon = epsilon
        self.median_ = None   # For nonconstant features only.
        self.iqr_ = None      # For nonconstant features only.
        self.constant_mask = None  # Boolean mask for constant features.
        self.nonconstant_mask = None
        self.constant_means = None
        self.original_dim = None

    def fit(self, X):
        # X: shape (n_samples, n_features)
        self.original_dim = X.shape[1]
        medians = np.median(X, axis=0)
        q_low = np.percentile(X, self.quantile_range[0], axis=0)
        q_high = np.percentile(X, self.quantile_range[1], axis=0)
        iqrs = q_high - q_low
        # Flag features with very small IQR as constant.
        self.constant_mask = iqrs < self.epsilon
        self.nonconstant_mask = ~self.constant_mask
        # For constant features, store their mean.
        all_means = np.mean(X, axis=0)
        self.constant_means = all_means[self.constant_mask]
        # Store median and IQR for nonconstant features.
        self.median_ = medians[self.nonconstant_mask]
        self.iqr_ = iqrs[self.nonconstant_mask]
        return self

    def transform(self, X):
        # Only scale nonconstant features.
        return (X[:, self.nonconstant_mask] - self.median_) / self.iqr_

    def inverse_transform(self, X_scaled):
        """
        X_scaled: array of shape (n_samples, n_nonconstant_features)
        Returns an array of shape (n_samples, original_dim) where constant features are
        filled with their computed mean.
        """
        n_samples = X_scaled.shape[0]
        X_orig = np.empty((n_samples, self.original_dim))
        # Inverse-transform nonconstant features.
        X_orig[:, self.nonconstant_mask] = X_scaled * self.iqr_ + self.median_
        # Fill constant features with their stored mean.
        X_orig[:, self.constant_mask] = np.tile(self.constant_means, (n_samples, 1))
        return X_orig

    def map_feature_index(self, original_index: int) -> int:
        """
        Maps a feature index from the original space to the reduced space (after removing constant features).
        
        Parameters:
            original_index (int): The index in the original feature space.
            
        Returns:
            int: The corresponding index in the reduced feature space.
            
        Raises:
            ValueError: If the index is out of range or the feature is constant.
        """
        if original_index < 0 or original_index >= self.original_dim:
            raise ValueError("Feature index out of range.")
        if not self.nonconstant_mask[original_index]:
            raise ValueError(f"Feature at original index {original_index} is considered constant and was removed.")
        # New index is the count of nonconstant features before the original_index.
        new_index = int(np.sum(self.nonconstant_mask[:original_index]))
        return new_index


# --- Utility Context Manager ---
@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr output.
    Useful when running code that prints too much output during training.
    """
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


# --- FiLM Module: Feature-wise Linear Modulation ---
class FeatureWiseLinearModulation(nn.Module):
    """
    Applies FiLM (Feature-wise Linear Modulation) to an input tensor.
    The modulation is defined as: FiLM(X) = gamma * X + beta.
    
    Parameters:
        target_feature_dimension (int): The dimensionality of the features to modulate.
        conditioning_vector_dimension (int): The dimensionality of the conditioning vector.
    """
    def __init__(self, target_feature_dimension: int, conditioning_vector_dimension: int):
        super().__init__()
        self.linear_generate_film_parameters = nn.Linear(
            conditioning_vector_dimension,
            target_feature_dimension * 2  # Generate both gamma and beta.
        )
    
    def forward(self, input_features: torch.Tensor, conditioning_vector: torch.Tensor) -> torch.Tensor:
        film_params = self.linear_generate_film_parameters(conditioning_vector)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * input_features + beta


# --- Time Embedding Module ---
class TimeEmbedding(nn.Module):
    """
    Encodes a scalar diffusion time step into a higher-dimensional embedding.
    
    Parameters:
        time_embedding_dimension (int): The dimensionality of the time embedding.
    """
    def __init__(self, time_embedding_dimension: int):
        super().__init__()
        self.linear_time = nn.Linear(1, time_embedding_dimension)
    
    def forward(self, diffusion_time_step: torch.Tensor) -> torch.Tensor:
        return self.linear_time(diffusion_time_step)


# --- Transformer Encoder Layer with Cross-Attention ---
class TransformerEncoderLayerWithCrossAttention(nn.Module):
    """
    A custom transformer encoder layer that incorporates self-attention,
    cross-attention with conditioning tokens, and a feed-forward network.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerEncoderLayerWithCrossAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, token_condition):
        # Self-attention over the main input tokens.
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        # Cross-attention: main tokens attend to conditioning tokens.
        src2, _ = self.cross_attn(src, token_condition, token_condition)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        # Feed-forward network.
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm3(src)
        return src


# --- Hybrid Diffusion Transformer Model ---
class HybridDiffusionTransformerModel(pl.LightningModule):
    """
    A hybrid diffusion transformer model that incorporates both global and token-level conditioning,
    dynamic scheduling for the important feature loss, and a feature-specific learnable noise schedule.
    
    The noise schedule for the important feature follows a power-mean function:
        sigma_imp(t) = (sigma_min^(1/ρ_imp) + t*(sigma_max^(1/ρ_imp) - sigma_min^(1/ρ_imp)))^(ρ_imp)
    where ρ_imp is a learnable parameter.
    
    Dynamic scheduling for the important feature loss is computed based on a linear warmup and an adjustment
    based on the validation variance of the important feature.
    
    Parameters:
        number_of_rows_per_example (int): Number of rows per input example.
        input_feature_dimension (int): Dimensionality of input features.
        conditioning_num_rows (int): Number of rows for the conditioning data.
        conditioning_feature_dimension (int): Dimensionality of conditioning features.
        latent_embedding_dimension (int): Dimensionality of latent embeddings.
        number_of_transformer_layers (int): Number of transformer encoder layers.
        transformer_attention_head_count (int): Number of attention heads in each transformer layer.
        transformer_dropout (float, optional): Dropout rate in transformer layers. Default is 0.1.
        time_embedding_dimension (int, optional): Dimensionality of the time embedding. Default is 64.
        learning_rate (float, optional): Learning rate for optimization. Default is 1e-3.
        verbose (bool, optional): If True, training and validation loss plots are displayed. Default is False.
        important_feature_index (int, optional): Index of the important feature to be dynamically reweighted. Default is 1.
        feature_reweight_factor (float, optional): Maximum reweight factor for the important feature loss. Default is 1.0.
        warmup_epochs (int, optional): Number of epochs over which to linearly increase the dynamic weight. Default is 10.
        sigma_min (float, optional): Minimum noise scale for the important feature. Default is 0.1.
        sigma_max (float, optional): Maximum noise scale for the important feature. Default is 1.0.
    """
    def __init__(self,
                 number_of_rows_per_example: int,
                 input_feature_dimension: int,
                 conditioning_num_rows: int,
                 conditioning_feature_dimension: int,
                 latent_embedding_dimension: int,
                 number_of_transformer_layers: int,
                 transformer_attention_head_count: int,
                 transformer_dropout: float = 0.1,
                 time_embedding_dimension: int = 64,
                 learning_rate: float = 1e-3,
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 feature_reweight_factor: float = 1.0,
                 warmup_epochs: int = 10,
                 sigma_min: float = 0.1,
                 sigma_max: float = 1.0):
        super(HybridDiffusionTransformerModel, self).__init__()
        self.save_hyperparameters()
        self.number_of_rows_per_example = number_of_rows_per_example
        self.input_feature_dimension = input_feature_dimension
        self.conditioning_num_rows = conditioning_num_rows
        self.conditioning_feature_dimension = conditioning_feature_dimension
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.time_embedding_dimension = time_embedding_dimension
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.feature_reweight_factor = feature_reweight_factor
        self.warmup_epochs = warmup_epochs
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Dynamic weight and target variance for the important feature loss.
        self.dynamic_feature_weight = 1.0
        self.target_variance = None  # To be set during data fitting.

        # Loss histories for plotting both global and important feature losses.
        self.train_losses = []
        self.val_losses = []
        self.current_train_losses = []
        self.current_val_losses = []
        self.train_imp_losses = []
        self.val_imp_losses = []
        self.current_train_imp_losses = []
        self.current_val_imp_losses = []
        self.current_val_imp_vars = []

        # Encoder for main input.
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        # FiLM layer for modulation.
        self.film_layer = FeatureWiseLinearModulation(latent_embedding_dimension, latent_embedding_dimension)
        # Time embedding.
        self.time_embedding_module = TimeEmbedding(time_embedding_dimension)
        # Global conditioning projection.
        self.global_condition_projection = nn.Linear(conditioning_feature_dimension, latent_embedding_dimension)
        # Token-level conditioning projection.
        self.token_condition_projection = nn.Linear(conditioning_feature_dimension, latent_embedding_dimension)

        # Create transformer layers with cross-attention.
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(latent_embedding_dimension, transformer_attention_head_count, transformer_dropout)
            for _ in range(number_of_transformer_layers)
        ])

        # Decoder to map latent tokens back to input feature space.
        self.linear_decoder_latent_to_output = nn.Linear(latent_embedding_dimension, input_feature_dimension)

        # Learnable parameter for the noise schedule (ρ_imp) for the important feature.
        self.important_noise_param = nn.Parameter(torch.tensor(1.0))

    def apply_noise_schedule(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to input x using a feature-specific noise schedule for the important feature.
        For the important feature (at index self.important_feature_index), the noise is scaled
        according to a learnable power-mean schedule:
            sigma_imp(t) = (sigma_min^(1/ρ_imp) + t*(sigma_max^(1/ρ_imp) - sigma_min^(1/ρ_imp)))^(ρ_imp)
        Args:
            x (torch.Tensor): Input tensor of shape [batch, num_rows, input_feature_dimension].
            t (torch.Tensor): Diffusion time step tensor of shape [batch, 1].
        Returns:
            torch.Tensor: The input with added noise, where the important feature receives scaled noise.
        """
        noise = torch.randn_like(x)
        rho = torch.clamp(self.important_noise_param, min=1e-6)
        sigma_imp = (self.sigma_min ** (1.0 / rho) +
                     t * (self.sigma_max ** (1.0 / rho) - self.sigma_min ** (1.0 / rho))) ** rho
        noise_scale = torch.ones_like(x)
        # Scale the noise for the important feature across all rows.
        noise_scale[..., self.important_feature_index] = sigma_imp.expand(x.size(0), x.size(1)).squeeze(-1)
        return x + noise * noise_scale

    def compute_dynamic_weight(self, avg_val_variance: float) -> float:
        """
        Computes a dynamic weight for the important feature loss based on a linear warmup and validation variance.
        The weight is computed as:
            weight = warmup_weight * adjustment,
        where warmup_weight = 1 + (feature_reweight_factor - 1) * min(current_epoch, warmup_epochs) / warmup_epochs,
        and adjustment = min(1.0, avg_val_variance / target_variance) if target_variance > 0.
        Args:
            avg_val_variance (float): Average variance of the important feature in the validation set.
        Returns:
            float: The computed dynamic weight.
        """
        target_weight = self.feature_reweight_factor
        warmup_weight = 1 + (target_weight - 1) * min(self.current_epoch, self.warmup_epochs) / self.warmup_epochs
        adjustment = 1.0
        if self.target_variance is not None and self.target_variance > 0:
            adjustment = min(1.0, avg_val_variance / self.target_variance)
        return warmup_weight * adjustment

    def compute_weighted_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error loss with an extra dynamic weight applied to the important feature.
        Args:
            prediction (torch.Tensor): The predicted output.
            target (torch.Tensor): The ground truth.
        Returns:
            torch.Tensor: The weighted loss.
        """
        error = (prediction - target) ** 2
        weights = torch.ones(self.input_feature_dimension, device=self.device)
        weights[self.important_feature_index] = self.dynamic_feature_weight
        weighted_error = error * weights
        loss = weighted_error.mean()
        return loss

    def forward(self, input_rows: torch.Tensor, conditioning_tensor: torch.Tensor, diffusion_time_step: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid diffusion transformer model.
        1. Encodes the main input.
        2. Projects and combines global conditioning (summed over conditioning tokens) with time embedding.
        3. Applies FiLM modulation to the latent tokens.
        4. Applies token-level conditioning via cross-attention in transformer layers.
        5. Decodes the latent tokens back to input feature space.
        
        Args:
            input_rows (torch.Tensor): Main input of shape [batch, n, input_feature_dimension].
            conditioning_tensor (torch.Tensor): Conditioning input of shape [batch, m, conditioning_feature_dimension].
            diffusion_time_step (torch.Tensor): Diffusion time step of shape [batch, 1].
        Returns:
            torch.Tensor: The predicted output.
        """
        latent_tokens = self.linear_encoder_input_to_latent(input_rows)  # (B, n, latent_embedding_dimension)
        global_condition = conditioning_tensor.sum(dim=1)  # (B, conditioning_feature_dimension)
        global_condition = self.global_condition_projection(global_condition)  # (B, latent_embedding_dimension)
        time_embedding = self.time_embedding_module(diffusion_time_step)  # (B, time_embedding_dimension)
        if time_embedding.size(-1) != self.latent_embedding_dimension:
            proj = nn.Linear(time_embedding.size(-1), self.latent_embedding_dimension).to(self.device)
            time_embedding = proj(time_embedding)
        combined_condition = global_condition + time_embedding  # (B, latent_embedding_dimension)
        latent_tokens = self.film_layer(latent_tokens, combined_condition)  # (B, n, latent_embedding_dimension)
        # Token-level conditioning.
        token_condition = self.token_condition_projection(conditioning_tensor)  # (B, m, latent_embedding_dimension)
        latent_tokens = latent_tokens.transpose(0, 1)  # (n, B, latent_embedding_dimension)
        token_condition = token_condition.transpose(0, 1)  # (m, B, latent_embedding_dimension)
        for layer in self.transformer_layers:
            latent_tokens = layer(latent_tokens, token_condition)
        latent_tokens = latent_tokens.transpose(0, 1)  # (B, n, latent_embedding_dimension)
        predicted_output = self.linear_decoder_latent_to_output(latent_tokens)  # (B, n, input_feature_dimension)
        return predicted_output

    def training_step(self, batch, batch_idx):
        input_examples, conditioning = batch
        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)
        # Apply feature-specific noise schedule.
        noisy_input = self.apply_noise_schedule(input_examples, diffusion_time_step)
        predicted_output = self.forward(noisy_input, conditioning, diffusion_time_step)
        loss = self.compute_weighted_loss(predicted_output, input_examples)
        self.current_train_losses.append(loss.item())
        # Record MSE loss for the important feature.
        imp_loss = F.mse_loss(
            predicted_output[..., self.important_feature_index],
            input_examples[..., self.important_feature_index],
            reduction='mean'
        )
        self.current_train_imp_losses.append(imp_loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_examples, conditioning = batch
        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)
        noisy_input = self.apply_noise_schedule(input_examples, diffusion_time_step)
        predicted_output = self.forward(noisy_input, conditioning, diffusion_time_step)
        loss = self.compute_weighted_loss(predicted_output, input_examples)
        self.current_val_losses.append(loss.item())
        # Record important feature variance and loss.
        imp_feature = predicted_output[..., self.important_feature_index]
        batch_imp_variance = imp_feature.var().item()
        self.current_val_imp_vars.append(batch_imp_variance)
        imp_loss = F.mse_loss(
            predicted_output[..., self.important_feature_index],
            input_examples[..., self.important_feature_index],
            reduction='mean'
        )
        self.current_val_imp_losses.append(imp_loss.item())
        self.log("val_loss", loss)
        self.log("val_imp_var", batch_imp_variance)
        return loss

    def on_train_epoch_end(self):
        if self.current_train_losses:
            epoch_loss = sum(self.current_train_losses) / len(self.current_train_losses)
            self.train_losses.append(epoch_loss)
            self.current_train_losses = []
        if self.current_train_imp_losses:
            epoch_imp_loss = sum(self.current_train_imp_losses) / len(self.current_train_imp_losses)
            self.train_imp_losses.append(epoch_imp_loss)
            self.current_train_imp_losses = []

    def on_validation_epoch_end(self):
        if self.current_val_losses:
            epoch_loss = sum(self.current_val_losses) / len(self.current_val_losses)
            self.val_losses.append(epoch_loss)
            self.current_val_losses = []
        if self.current_val_imp_vars:
            avg_val_imp_var = sum(self.current_val_imp_vars) / len(self.current_val_imp_vars)
            # Update dynamic weight based on validation variance.
            self.dynamic_feature_weight = self.compute_dynamic_weight(avg_val_imp_var)
            self.current_val_imp_vars = []
        if self.current_val_imp_losses:
            avg_val_imp_loss = sum(self.current_val_imp_losses) / len(self.current_val_imp_losses)
            self.val_imp_losses.append(avg_val_imp_loss)
            self.current_val_imp_losses = []

    def on_train_end(self):
        if not self.verbose:
            return
        min_length = min(len(self.train_losses), len(self.val_losses),
                        len(self.train_imp_losses), len(self.val_imp_losses))
        if min_length == 0:
            print("No training or validation losses recorded.")
            return
        self.train_losses = self.train_losses[:min_length]
        self.val_losses = self.val_losses[:min_length]
        self.train_imp_losses = self.train_imp_losses[:min_length]
        self.val_imp_losses = self.val_imp_losses[:min_length]
        skip_first = 5 if min_length > 5 else 0
        epochs = range(skip_first + 1, min_length + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        # Plot global losses on the left y-axis.
        ax1.plot(epochs, self.train_losses[skip_first:], label='Train Global Loss', color='blue')
        ax1.plot(epochs, self.val_losses[skip_first:], label='Val Global Loss', color='navy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Global Loss', color='blue')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a twin axis for the important feature losses.
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.train_imp_losses[skip_first:], label='Train Imp Loss', color='red')
        ax2.plot(epochs, self.val_imp_losses[skip_first:], label='Val Imp Loss', color='orange')
        ax2.set_ylabel('Important Feature Loss', color='red')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends from both axes.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title('Training and Validation Losses')
        ax1.grid(True)
        plt.show()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def generate(self, conditioning_tensor: torch.Tensor, total_diffusion_steps: int = 1000) -> torch.Tensor:
        batch_size = conditioning_tensor.size(0)
        generated = torch.randn(
            batch_size,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=conditioning_tensor.device
        )
        for step in range(total_diffusion_steps):
            diffusion_time_step = torch.full(
                (batch_size, 1),
                step / total_diffusion_steps,
                device=conditioning_tensor.device
            )
            generated = self.forward(generated, conditioning_tensor, diffusion_time_step)
        return generated

class TransformerConditionalDiffusionGenerator(BaseEstimator):
    """
    A scikit-learn style generator that wraps a hybrid diffusion transformer model with dynamic loss 
    scheduling and feature-specific noise injection. This model applies robust scaling to both input 
    and conditioning data, and during training it uses a dynamic weight for the important feature loss 
    (computed via a linear warmup and variance adjustment) and a learnable noise schedule for the 
    important feature. The noise schedule for the important feature is defined as:
    
        sigma_imp(t) = (sigma_min^(1/ρ_imp) + t*(sigma_max^(1/ρ_imp) - sigma_min^(1/ρ_imp)))^(ρ_imp)
    
    where ρ_imp is a learnable parameter.
    
    Parameters:
        latent_embedding_dimension (int): Dimensionality of the latent embeddings in the transformer.
        number_of_transformer_layers (int): Number of layers in the transformer encoder.
        transformer_attention_head_count (int): Number of attention heads per transformer layer.
        transformer_dropout (float): Dropout rate in the transformer layers.
        time_embedding_dimension (int): Dimensionality of the time embedding.
        learning_rate (float): Learning rate used during training.
        maximum_epochs (int): Maximum number of epochs for training.
        training_batch_size (int): Batch size used for training.
        total_diffusion_steps (int): Number of diffusion steps to perform during generation.
        verbose (bool): If True, training and validation loss plots (with dual y-axes) are displayed.
        important_feature_index (int): Index of the important feature (after robust scaling) whose loss is dynamically reweighted.
        feature_reweight_factor (float): Maximum reweight factor applied to the loss of the important feature.
        warmup_epochs (int): Number of epochs over which the dynamic weight is linearly increased.
        robust_epsilon (float): Epsilon for the CustomRobustScaler to determine near-constant features.
    """
    def __init__(self,
                 latent_embedding_dimension: int = 128,
                 number_of_transformer_layers: int = 4,
                 transformer_attention_head_count: int = 4,
                 transformer_dropout: float = 0.1,
                 time_embedding_dimension: int = 128,
                 learning_rate: float = 1e-3,
                 maximum_epochs: int = 10,
                 training_batch_size: int = 32,
                 total_diffusion_steps: int = 1000,
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 feature_reweight_factor: float = 1.0,
                 warmup_epochs: int = 10,
                 robust_epsilon: float = 1e-8):
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.time_embedding_dimension = time_embedding_dimension
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.training_batch_size = training_batch_size
        self.total_diffusion_steps = total_diffusion_steps
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.feature_reweight_factor = feature_reweight_factor
        self.warmup_epochs = warmup_epochs
        self.robust_epsilon = robust_epsilon

        # These will be set during fitting.
        self.number_of_rows_per_example = None  # For main input X.
        self.conditioning_num_rows = None       # For conditioning y.
        self.input_feature_dimension = None
        self.conditioning_feature_dimension = None

        self.model = None
        self.input_scaler = None
        self.cond_scaler = None

    def _fit_scalers(self, X_array, y_array):
        """
        Fit robust scalers on input X and conditioning y.
        """
        B, n, d = X_array.shape
        X_reshaped = X_array.reshape(-1, d)
        self.input_scaler = CustomRobustScaler(epsilon=self.robust_epsilon).fit(X_reshaped)
        # Map the important feature index from original to the reduced space.
        self.important_feature_index = self.input_scaler.map_feature_index(self.important_feature_index)
        B, m, f = y_array.shape
        y_reshaped = y_array.reshape(-1, f)
        self.cond_scaler = CustomRobustScaler(epsilon=self.robust_epsilon).fit(y_reshaped)

    def _transform_data(self, X_array, y_array):
        """
        Transforms the data using the fitted robust scalers.
        """
        B, n, d = X_array.shape
        X_reshaped = X_array.reshape(-1, d)
        X_scaled_temp = self.input_scaler.transform(X_reshaped)
        new_d = X_scaled_temp.shape[1]
        X_scaled = X_scaled_temp.reshape(B, n, new_d)
        
        B, m, f = y_array.shape
        y_reshaped = y_array.reshape(-1, f)
        y_scaled_temp = self.cond_scaler.transform(y_reshaped)
        new_f = y_scaled_temp.shape[1]
        y_scaled = y_scaled_temp.reshape(B, m, new_f)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array):
        """
        Reconstructs the original input features from the scaled version.
        """
        B, n, new_d = X_array.shape
        X_reshaped = X_array.reshape(-1, new_d)
        X_orig = self.input_scaler.inverse_transform(X_reshaped).reshape(B, n, self.input_scaler.original_dim)
        return X_orig

    def fit(self, X, y):
        """
        Fit the hybrid diffusion transformer model on input X and conditioning y.
        
        Parameters:
            X: List of numpy arrays with shape (n_i, d) per instance.
            y: List of numpy arrays with shape (m_i, f) per instance.
        
        Returns:
            self
        """
        # Pad X arrays to have a consistent number of rows.
        max_X_rows = max(x.shape[0] for x in X)
        self.number_of_rows_per_example = max_X_rows
        original_d = X[0].shape[1]
        X_padded = []
        for x in X:
            n_rows = x.shape[0]
            if n_rows < max_X_rows:
                pad_width = ((0, max_X_rows - n_rows), (0, 0))
                x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
            X_padded.append(x)
        X_array = np.stack(X_padded, axis=0)

        # Pad y arrays to have a consistent number of rows.
        max_y_rows = max(arr.shape[0] for arr in y)
        self.conditioning_num_rows = max_y_rows
        original_f = y[0].shape[1]
        y_padded = []
        for arr in y:
            n_rows = arr.shape[0]
            if n_rows < max_y_rows:
                pad_width = ((0, max_y_rows - n_rows), (0, 0))
                arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
            y_padded.append(arr)
        y_array = np.stack(y_padded, axis=0)
        
        self._fit_scalers(X_array, y_array)
        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        self.conditioning_feature_dimension = y_scaled.shape[2]
        
        # Compute the target variance for the important feature.
        self.target_variance = np.var(X_scaled[..., self.important_feature_index])
        
        # Instantiate the modified hybrid diffusion transformer model.
        from_modified_model = HybridDiffusionTransformerModel  # Assumes this class is defined (as above).
        self.model = from_modified_model(
            number_of_rows_per_example=self.number_of_rows_per_example,
            input_feature_dimension=self.input_feature_dimension,
            conditioning_num_rows=self.conditioning_num_rows,
            conditioning_feature_dimension=self.conditioning_feature_dimension,
            latent_embedding_dimension=self.latent_embedding_dimension,
            number_of_transformer_layers=self.number_of_transformer_layers,
            transformer_attention_head_count=self.transformer_attention_head_count,
            transformer_dropout=self.transformer_dropout,
            time_embedding_dimension=self.time_embedding_dimension,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            important_feature_index=self.important_feature_index,
            feature_reweight_factor=self.feature_reweight_factor,
            warmup_epochs=self.warmup_epochs,
            sigma_min=0.1,      # You can also parameterize these if desired.
            sigma_max=1.0
        )
        self.model.target_variance = self.target_variance
        
        # Create datasets and dataloaders.
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        total_size = len(dataset)
        val_size = int(total_size * 0.1)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=self.training_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.training_batch_size, shuffle=False)
        
        trainer = pl.Trainer(max_epochs=self.maximum_epochs,
                             logger=False,
                             enable_checkpointing=False,
                             enable_progress_bar=False)
        if not self.verbose:
            with suppress_output():
                trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        return self

    def predict(self, y):
        """
        Generate predictions given conditioning y.
        
        Parameters:
            y: List of numpy arrays with shape (m_i, f) per instance.
            
        Returns:
            List of generated outputs (numpy arrays) corresponding to each instance.
        """
        # Pad y arrays to match the fitted conditioning length.
        max_y_rows = self.conditioning_num_rows
        y_padded = []
        for arr in y:
            n_rows = arr.shape[0]
            if n_rows < max_y_rows:
                pad_width = ((0, max_y_rows - n_rows), (0, 0))
                arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
            else:
                arr = arr[:max_y_rows]
            y_padded.append(arr)
        y_array = np.stack(y_padded, axis=0)
        # Transform conditioning data.
        B, m, f = y_array.shape
        y_scaled_temp = self.cond_scaler.transform(y_array.reshape(-1, f))
        y_scaled = y_scaled_temp.reshape(B, m, -1)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)
            generated = self.model.generate(y_tensor, total_diffusion_steps=self.total_diffusion_steps)
            generated_np = generated.cpu().numpy()
            generated_orig = self._inverse_transform_input(generated_np)
            return [generated_orig[i] for i in range(generated_orig.shape[0])]

    def sample(self, n_samples: int):
        """
        Generate n_samples using the diffusion model's sampling procedure.
        
        Parameters:
            n_samples (int): Number of samples to generate.
            
        Returns:
            List of generated outputs (numpy arrays).
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.eval()
        with torch.no_grad():
            # Sample conditioning from the trained conditional generator.
            y_sample_np = self.conditional_generator_estimator.sample(n_samples)
            y_sample_tensor = torch.tensor(y_sample_np, dtype=torch.float32, device=device)
            X_generated = self.model.generate(y_sample_tensor, total_diffusion_steps=self.total_diffusion_steps)
            X_np = X_generated.cpu().numpy()
            X_orig = self._inverse_transform_input(X_np)
            return [X_orig[i] for i in range(X_orig.shape[0])]

    def save(self, filename: str):
        """
        Saves the current instance state to disk.
        
        Parameters:
            filename (str): Path to the file where the state will be saved.
        """
        state = {
            'hyperparameters': {
                'latent_embedding_dimension': self.latent_embedding_dimension,
                'number_of_transformer_layers': self.number_of_transformer_layers,
                'transformer_attention_head_count': self.transformer_attention_head_count,
                'transformer_dropout': self.transformer_dropout,
                'time_embedding_dimension': self.time_embedding_dimension,
                'learning_rate': self.learning_rate,
                'maximum_epochs': self.maximum_epochs,
                'training_batch_size': self.training_batch_size,
                'total_diffusion_steps': self.total_diffusion_steps,
                'verbose': self.verbose,
                'important_feature_index': self.important_feature_index,
                'feature_reweight_factor': self.feature_reweight_factor,
                'warmup_epochs': self.warmup_epochs,
                'robust_epsilon': self.robust_epsilon
            },
            'number_of_rows_per_example': self.number_of_rows_per_example,
            'conditioning_num_rows': self.conditioning_num_rows,
            'input_feature_dimension': self.input_feature_dimension,
            'conditioning_feature_dimension': self.conditioning_feature_dimension,
            'model_state_dict': self.model.state_dict() if self.model is not None else None,
            'input_scaler': self.input_scaler,
            'cond_scaler': self.cond_scaler
        }
        torch.save(state, filename)

    @classmethod
    def load(cls, filename: str):
        """
        Loads an instance from disk.
        
        Parameters:
            filename (str): Path to the saved state file.
            
        Returns:
            TransformerConditionalDiffusionGenerator: The loaded instance.
        """
        state = torch.load(filename, map_location='cpu')
        hp = state['hyperparameters']
        obj = cls(
            latent_embedding_dimension=hp['latent_embedding_dimension'],
            number_of_transformer_layers=hp['number_of_transformer_layers'],
            transformer_attention_head_count=hp['transformer_attention_head_count'],
            transformer_dropout=hp['transformer_dropout'],
            time_embedding_dimension=hp['time_embedding_dimension'],
            learning_rate=hp['learning_rate'],
            maximum_epochs=hp['maximum_epochs'],
            training_batch_size=hp['training_batch_size'],
            total_diffusion_steps=hp['total_diffusion_steps'],
            verbose=hp['verbose'],
            important_feature_index=hp['important_feature_index'],
            feature_reweight_factor=hp['feature_reweight_factor'],
            warmup_epochs=hp['warmup_epochs'],
            robust_epsilon=hp['robust_epsilon']
        )
        obj.number_of_rows_per_example = state['number_of_rows_per_example']
        obj.conditioning_num_rows = state['conditioning_num_rows']
        obj.input_feature_dimension = state['input_feature_dimension']
        obj.conditioning_feature_dimension = state['conditioning_feature_dimension']
        obj.input_scaler = state['input_scaler']
        obj.cond_scaler = state['cond_scaler']
        if state['model_state_dict'] is not None:
            from_modified_model = HybridDiffusionTransformerModel  # Assumes this class is defined.
            obj.model = from_modified_model(
                number_of_rows_per_example=obj.number_of_rows_per_example,
                input_feature_dimension=obj.input_feature_dimension,
                conditioning_num_rows=obj.conditioning_num_rows,
                conditioning_feature_dimension=obj.conditioning_feature_dimension,
                latent_embedding_dimension=obj.latent_embedding_dimension,
                number_of_transformer_layers=obj.number_of_transformer_layers,
                transformer_attention_head_count=obj.transformer_attention_head_count,
                transformer_dropout=obj.transformer_dropout,
                time_embedding_dimension=obj.time_embedding_dimension,
                learning_rate=obj.learning_rate,
                verbose=obj.verbose,
                important_feature_index=obj.important_feature_index,
                feature_reweight_factor=obj.feature_reweight_factor,
                warmup_epochs=obj.warmup_epochs,
                sigma_min=0.1,
                sigma_max=1.0
            )
            obj.model.load_state_dict(state['model_state_dict'])
        return obj
