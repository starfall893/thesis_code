import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
import numpy as np
from .qlstm import QLSTM, quantile_loss

class QLSTMTrainer:
    """
    Trainer class for QLSTM models.
    
    This class handles the training process for both price and solar forecasting models.
    
    Args:
        model (QLSTM): The QLSTM model to train
        learning_rate (float): Learning rate for optimization
        device (str): Device to train on ('cuda' or 'cpu')
    """
    
    def __init__(self, model: QLSTM, learning_rate: float, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, dataloader: DataLoader, quantiles: list) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader (DataLoader): Training data loader
            quantiles (list): List of quantiles used for prediction
            
        Returns:
            Dict[str, float]: Dictionary of metrics including loss, MSE, and MAE
        """
        self.model.train()
        total_loss = 0
        total_mse = {q: 0 for q in quantiles}
        total_mae = {q: 0 for q in quantiles}
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            preds = self.model(batch_x)
            
            # Compute loss
            loss = quantile_loss(preds, batch_y, quantiles)
            
            # Compute MSE and MAE for each quantile
            for i, q in enumerate(quantiles):
                mse = torch.mean((preds[:, i, :] - batch_y) ** 2)
                mae = torch.mean(torch.abs(preds[:, i, :] - batch_y))
                total_mse[q] += mse.item()
                total_mae[q] += mae.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'mse': {q: total_mse[q] / num_batches for q in quantiles},
            'mae': {q: total_mae[q] / num_batches for q in quantiles}
        }
    
    def evaluate(self, dataloader: DataLoader, quantiles: list) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader (DataLoader): Evaluation data loader
            quantiles (list): List of quantiles used for prediction
            
        Returns:
            Dict[str, float]: Dictionary of metrics including loss, MSE, and MAE
        """
        self.model.eval()
        total_loss = 0
        total_mse = {q: 0 for q in quantiles}
        total_mae = {q: 0 for q in quantiles}
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                preds = self.model(batch_x)
                loss = quantile_loss(preds, batch_y, quantiles)
                
                # Compute MSE and MAE for each quantile
                for i, q in enumerate(quantiles):
                    mse = torch.mean((preds[:, i, :] - batch_y) ** 2)
                    mae = torch.mean(torch.abs(preds[:, i, :] - batch_y))
                    total_mse[q] += mse.item()
                    total_mae[q] += mae.item()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'mse': {q: total_mse[q] / num_batches for q in quantiles},
            'mae': {q: total_mae[q] / num_batches for q in quantiles}
        }
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              quantiles: list,
              epochs: int,
              early_stopping_patience: int = 5) -> Dict[str, Any]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            quantiles (list): List of quantiles used for prediction
            epochs (int): Maximum number of epochs
            early_stopping_patience (int): Number of epochs to wait for improvement
            
        Returns:
            Dict[str, Any]: Training history and best model state
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': {q: [] for q in quantiles},
            'val_mse': {q: [] for q in quantiles},
            'train_mae': {q: [] for q in quantiles},
            'val_mae': {q: [] for q in quantiles}
        }
        
        for epoch in range(epochs):
            # Train and evaluate
            train_metrics = self.train_epoch(train_loader, quantiles)
            val_metrics = self.evaluate(val_loader, quantiles)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            
            for q in quantiles:
                history['train_mse'][q].append(train_metrics['mse'][q])
                history['val_mse'][q].append(val_metrics['mse'][q])
                history['train_mae'][q].append(train_metrics['mae'][q])
                history['val_mae'][q].append(val_metrics['mae'][q])
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_metrics["loss"]:.4f}, Val Loss: {val_metrics["loss"]:.4f}')
            for q in quantiles:
                print(f'Quantile {q}:')
                print(f'  Train MSE: {train_metrics["mse"][q]:.4f}, Val MSE: {val_metrics["mse"][q]:.4f}')
                print(f'  Train MAE: {train_metrics["mae"][q]:.4f}, Val MAE: {val_metrics["mae"][q]:.4f}')
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Restore best model
        self.model.load_state_dict(best_state)
        
        return {
            'history': history,
            'best_state': best_state,
            'best_val_loss': best_val_loss,
            'model_params': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'horizon': self.model.horizon,
                'quantiles': quantiles,
                'num_cities': self.model.num_cities
            }
        } 