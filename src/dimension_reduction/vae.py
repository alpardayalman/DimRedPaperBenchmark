"""Variational Autoencoder (VAE) dimension reduction implementation."""

from typing import Optional, TYPE_CHECKING, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseDimensionReducer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VAEEncoder(nn.Module):
    """Encoder network for the Variational Autoencoder.
    
    This module encodes input data into a latent space representation
    by outputting both mean and log variance parameters.
    
    Attributes:
        input_dim: Input dimension.
        hidden_dims: List of hidden layer dimensions.
        latent_dim: Latent space dimension.
        dropout_rate: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout_rate: float = 0.1
    ) -> None:
        """Initialize the encoder.
        
        Args:
            input_dim: Input dimension.
            hidden_dims: List of hidden layer dimensions.
            latent_dim: Latent space dimension.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple of (mu, logvar) for the latent space.
        """
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for the Variational Autoencoder.
    
    This module reconstructs input data from latent space representations.
    
    Attributes:
        latent_dim: Latent space dimension.
        hidden_dims: List of hidden layer dimensions (reversed from encoder).
        output_dim: Output dimension.
        dropout_rate: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout_rate: float = 0.1
    ) -> None:
        """Initialize the decoder.
        
        Args:
            latent_dim: Latent space dimension.
            hidden_dims: List of hidden layer dimensions (reversed from encoder).
            output_dim: Output dimension.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build hidden layers (reversed order from encoder)
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            z: Latent space tensor.
            
        Returns:
            Reconstructed input tensor.
        """
        return self.decoder(z)


class VAE(nn.Module):
    """Variational Autoencoder (VAE) implementation.
    
    A VAE consists of an encoder that maps data to a latent space
    and a decoder that reconstructs data from the latent space.
    The latent space is regularized to follow a prior distribution.
    
    Attributes:
        encoder: Encoder network.
        decoder: Decoder network.
        latent_dim: Latent space dimension.
    """
    
    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        latent_dim: int
    ) -> None:
        """Initialize the VAE.
        
        Args:
            encoder: Encoder network.
            decoder: Decoder network.
            latent_dim: Latent space dimension.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from the latent space.
        
        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
            
        Returns:
            Sampled latent vectors.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple of (reconstructed_x, mu, logvar).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class VAEReducer(BaseDimensionReducer):
    """Variational Autoencoder (VAE) for dimension reduction.
    
    VAE is a deep learning-based dimension reduction technique that
    learns a compressed representation of data by training an autoencoder
    with a probabilistic latent space. It can capture complex non-linear
    relationships in the data.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        hidden_dims: List of hidden layer dimensions for encoder/decoder.
        learning_rate: Learning rate for optimization.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        beta: Weight for KL divergence term in loss.
        device: Device to run the model on.
        model: VAE model instance.
        optimizer: Optimizer instance.
        scaler: Data scaler for normalization.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        hidden_dims: Optional[list[int]] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        beta: float = 1.0,
        device: Optional[str] = None
    ) -> None:
        """Initialize VAE Reducer.
        
        Args:
            n_components: Number of components to reduce to.
            random_state: Random state for reproducibility.
            hidden_dims: List of hidden layer dimensions for encoder/decoder.
            learning_rate: Learning rate for optimization.
            batch_size: Batch size for training.
            epochs: Number of training epochs.
            beta: Weight for KL divergence term in loss.
            device: Device to run the model on.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.model: Optional[VAE] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scaler: Optional[Any] = None
        self.input_dim: Optional[int] = None
        
        # Set random seeds
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _build_model(self, input_dim: int) -> None:
        """Build the VAE model.
        
        Args:
            input_dim: Input dimension.
        """
        encoder = VAEEncoder(input_dim, self.hidden_dims, self.n_components)
        decoder = VAEDecoder(self.n_components, self.hidden_dims, input_dim)
        
        self.model = VAE(encoder, decoder, self.n_components).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            
        Returns:
            Total loss.
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(recon_x, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss
    
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "VAEReducer":
        """Fit VAE to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            self: Fitted VAE instance.
        """
        X = self._validate_data(X)
        
        # Store input dimension
        self.input_dim = X.shape[1]
        
        # Build model
        self._build_model(self.input_dim)
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_data in dataloader:
                x = batch_data[0]
                
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_x, mu, logvar = self.model(x)
                
                # Compute loss
                loss = self._vae_loss(recon_x, x, mu, logvar)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
        self.fitted = True
        return self
    
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data using the fitted VAE.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get latent representation
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encoder(X_tensor)
            return mu.cpu().numpy()
    
    def inverse_transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data back to original space.
        
        Args:
            X: Data in reduced space of shape (n_samples, n_components).
            
        Returns:
            Data in original space of shape (n_samples, n_features).
        """
        self._check_is_fitted()
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Reconstruct from latent space
        self.model.eval()
        with torch.no_grad():
            recon_x = self.model.decoder(X_tensor)
            return recon_x.cpu().numpy()
    
    def sample(self, n_samples: int) -> "NDArray[np.floating]":
        """Sample from the latent space.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            Generated samples in original space.
        """
        self._check_is_fitted()
        
        # Sample from standard normal distribution
        z = torch.randn(n_samples, self.n_components).to(self.device)
        
        # Decode samples
        self.model.eval()
        with torch.no_grad():
            samples = self.model.decoder(z)
            return samples.cpu().numpy()
    
    def get_latent_representation(self, X: "NDArray[np.floating]") -> Tuple["NDArray[np.floating]", "NDArray[np.floating]"]:
        """Get the full latent representation (mean and log variance).
        
        Args:
            X: Input data.
            
        Returns:
            Tuple of (mean, log_variance) in latent space.
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get latent representation
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encoder(X_tensor)
            return mu.cpu().numpy(), logvar.cpu().numpy()
