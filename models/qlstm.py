import torch
import torch.nn as nn
import numpy as np

class QLSTM(nn.Module):
    """
    Quantile LSTM model for time series forecasting.
    
    This model can operate in two modes:
    1. Quantile mode: Predicts multiple quantiles for uncertainty quantification
    2. DFL mode: Predicts single values for direct optimization
    
    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units in LSTM
        horizon (int): Forecast horizon
        quantiles (list): List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        num_cities (int): Number of cities to forecast for
        dropout (float): Dropout rate for regularization
        dfl_mode (bool): Whether to use DFL mode (single predictions)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 horizon: int, 
                 quantiles: list,
                 num_cities: int = 1,
                 dropout: float = 0.1,
                 dfl_mode: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_quantiles = len(quantiles)
        self.num_cities = num_cities
        self.dfl_mode = dfl_mode
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        
        # City-specific output layers
        if dfl_mode:
            # Single prediction per city
            self.city_layers = nn.ModuleList([
                nn.Linear(hidden_dim, horizon)
                for _ in range(num_cities)
            ])
        else:
            # Quantile predictions per city
            self.city_layers = nn.ModuleList([
                nn.Linear(hidden_dim, self.num_quantiles * horizon)
                for _ in range(num_cities)
            ])
        
    def forward(self, x: torch.Tensor, city_idx: int = 0) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            city_idx (int): Index of the city to forecast for
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_quantiles, horizon) in quantile mode
                        or (batch_size, horizon) in DFL mode
        """
        # LSTM forward pass
        h, _ = self.lstm(x)
        
        # Get last hidden state and project to predictions
        h = self.city_layers[city_idx](h[:, -1, :])
        
        if self.dfl_mode:
            return h  # (batch_size, horizon)
        else:
            return h.view(h.size(0), self.num_quantiles, self.horizon)
    
    def forward_all_cities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all cities.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_cities, num_quantiles, horizon) in quantile mode
                        or (batch_size, num_cities, horizon) in DFL mode
        """
        # LSTM forward pass
        h, _ = self.lstm(x)
        
        # Get last hidden state
        h = h[:, -1, :]
        
        # Project to predictions for each city
        city_preds = []
        for city_layer in self.city_layers:
            city_pred = city_layer(h)
            if self.dfl_mode:
                city_preds.append(city_pred)
            else:
                city_preds.append(city_pred.view(h.size(0), self.num_quantiles, self.horizon))
        
        # Stack city predictions
        return torch.stack(city_preds, dim=1)

def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """
    Compute the quantile loss (pinball loss) for multiple quantiles.
    
    Args:
        preds (torch.Tensor): Predictions of shape (batch_size, num_quantiles, horizon)
        target (torch.Tensor): Targets of shape (batch_size, horizon)
        quantiles (list): List of quantiles used for prediction
        
    Returns:
        torch.Tensor: Average quantile loss
    """
    loss = 0.0
    for i, q in enumerate(quantiles):
        error = target - preds[:, i, :]
        loss += torch.mean(torch.max((q-1) * error, q * error))
    return loss 