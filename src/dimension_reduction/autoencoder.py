"""Autoencoder-based dimension reduction implementation."""

from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseDimensionReducer
import copy

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Autoencoder(nn.Module):
    """Neural network autoencoder for dimension reduction.
    
    A simple feedforward autoencoder with configurable architecture.
    
    Attributes:
        encoder: The encoder network.
        decoder: The decoder network.
        bottleneck_dim: Dimension of the bottleneck layer.
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1
    ) -> None:
        """Initialize the autoencoder.
        
        Args:
            input_dim: Input dimension.
            bottleneck_dim: Dimension of the bottleneck layer.
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple of (encoded, decoded) tensors.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to bottleneck representation.
        
        Args:
            x: Input tensor.
            
        Returns:
            Encoded tensor.
        """
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck representation to output.
        
        Args:
            x: Bottleneck tensor.
            
        Returns:
            Decoded tensor.
        """
        return self.decoder(x)


class AutoencoderReducer(BaseDimensionReducer):
    """Autoencoder-based dimension reduction.
    
    Uses a neural network autoencoder to learn a compressed representation
    of the data. The autoencoder is trained to reconstruct the input data
    from a lower-dimensional bottleneck layer.
    
    Attributes:
        n_components: Number of components to reduce to.
        random_state: Random state for reproducibility.
        hidden_dims: List of hidden layer dimensions.
        learning_rate: Learning rate for optimization.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        dropout_rate: Dropout rate for regularization.
        device: Device to run the model on.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        hidden_dims: List[int] = [256, 128, 32],
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 300,
        dropout_rate: float = 0.0,
        device: Optional[str] = None
    ) -> None:
        """Initialize the autoencoder reducer.
        
        Args:
            n_components: Number of components to reduce to.
            random_state: Random state for reproducibility.
            hidden_dims: List of hidden layer dimensions.
            learning_rate: Learning rate for optimization.
            batch_size: Batch size for training.
            epochs: Number of training epochs.
            dropout_rate: Dropout rate for regularization.
            device: Device to run the model on.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
        
        self._model: Optional[Autoencoder] = None
        self._input_dim: Optional[int] = None
    
    def fit(self, X: "NDArray[np.floating]", y: Optional["NDArray"] = None) -> "AutoencoderReducer":
        """Fit the autoencoder to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored, present for API consistency.
            
        Returns:
            self: Fitted autoencoder reducer instance.
        """
        X = self._validate_data(X)
        self._input_dim = X.shape[1]
        best_loss = float("inf")
        
        # Create model
        self._model = Autoencoder(
            input_dim=self._input_dim,
            bottleneck_dim=self.n_components,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder learns identity mapping
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        
        # Training loop
        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                encoded, decoded = self._model(batch_x)
                #loss = criterion(decoded, batch_y)
                x_n = torch.nn.functional.normalize(batch_x, dim=1)
                xhat_n = torch.nn.functional.normalize(decoded, dim=1)
                cos_loss = 1.0 - (x_n * xhat_n).sum(dim=1).mean()
                loss = cos_loss  # or loss = cos_loss + λ * mse
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self._model.state_dict())
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        self._model.load_state_dict(best_state)
        self.fitted = True
        return self
    
    def transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data using the fitted autoencoder.
        
        Args:
            X: Data to transform of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        
        if self._model is None:
            raise ValueError("Model not fitted")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Encode
        self._model.eval()
        with torch.no_grad():
            encoded = self._model.encode(X_tensor)
        
        return encoded.cpu().numpy()
    
    def inverse_transform(self, X: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform data back to original space.
        
        Args:
            X: Data in reduced space of shape (n_samples, n_components).
            
        Returns:
            Data in original space of shape (n_samples, n_features).
        """
        self._check_is_fitted()
        
        if self._model is None:
            raise ValueError("Model not fitted")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Decode
        self._model.eval()
        with torch.no_grad():
            decoded = self._model.decode(X_tensor)
        
        return decoded.cpu().numpy()
    
    def get_model(self) -> Optional[Autoencoder]:
        """Get the trained autoencoder model.
        
        Returns:
            The trained autoencoder model or None if not fitted.
        """
        return self._model
    
    def get_reconstruction_error(self, X: "NDArray[np.floating]") -> float:
        """Calculate reconstruction error.
        
        Args:
            X: Input data.
            
        Returns:
            Mean squared reconstruction error.
        """
        self._check_is_fitted()
        X_reconstructed = self.inverse_transform(self.transform(X))
        return np.mean((X - X_reconstructed) ** 2)
