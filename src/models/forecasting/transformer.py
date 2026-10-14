import torch
import torch.nn as nn
import logging
import os
import copy
from typing import Tuple

logger = logging.getLogger(__name__)

class TimeSeriesTransformer(nn.Module):
    """
    Transformer architecture for time-series forecasting.
    """
    def __init__(self, input_dim: int, d_model: int = 64, n_heads: int = 4, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        # Project input to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output prediction head
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        x_proj = self.input_projection(x)
        
        # Encode
        encoded = self.transformer_encoder(x_proj)
        
        # Take the last sequence output to predict the future step (or use full sequence depending on task)
        # Assuming we predict a single future step based on the entire sequence:
        last_step_output = encoded[:, -1, :] 
        
        output = self.output_head(last_step_output)
        return output

class TransformerForecaster:
    """
    Wrapper class to manage training, GPU allocation, early stopping, and predictions.
    """
    def __init__(self, model_params: dict, learning_rate: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
        else:
            logger.error("CUDA is NOT available. Falling back to CPU is not recommended for this deep learning model!")
            
        self.model = TimeSeriesTransformer(**model_params).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_model_state = None
        
    def fit(self, train_loader, val_loader, epochs: int = 50, patience: int = 5, checkpoint_dir: str = "data/processed/checkpoints"):
        """
        Trains the Transformer with early stopping and best-model checkpointing to save GPU credits.
        """
        logger.info(f"Starting training on {self.device.type.upper()} for a maximum of {epochs} epochs...")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_transformer.pth")
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                
                loss = self.criterion(predictions.squeeze(), y_batch.squeeze())
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation Step
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_preds = self.model(X_val)
                    loss = self.criterion(val_preds.squeeze(), y_val.squeeze())
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            logger.debug(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_state, checkpoint_path)
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with val loss {best_val_loss:.5f}")
                break
                
        # Restore the best model before returning
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            
        logger.info(f"Training complete. Best checkpoint saved to {checkpoint_path}")

    def predict(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generates predictions using the trained best model.
        """
        self.model.eval()
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor)
        return preds.cpu()
