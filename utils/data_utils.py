import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Tuple, Dict, Any

class DataPreprocessor:
    """
    Data preprocessing utilities for time series forecasting.
    
    This class handles data preparation, scaling, and dataset creation
    for both price and solar forecasting.
    
    Args:
        input_len (int): Length of input sequence
        forecast_len (int): Length of forecast horizon
        batch_size (int): Batch size for training
    """
    
    def __init__(self, input_len: int, forecast_len: int, batch_size: int):
        self.input_len = input_len
        self.forecast_len = forecast_len
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_cols: list,
                    target_col: str) -> Tuple[DataLoader, StandardScaler]:
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_cols (list): List of feature columns
            target_col (str): Target column name
            
        Returns:
            Tuple[DataLoader, StandardScaler]: Data loader and fitted scaler
        """
        # Handle datetime columns if present
        if 'day' in df.columns and 'hour' in df.columns:
            # Convert day to numeric if it's a date
            if pd.api.types.is_datetime64_any_dtype(df['day']):
                df['day'] = df['day'].dt.dayofyear
            df['hour'] = df['hour'].astype(int)
            df = df.sort_values(['day', 'hour']).reset_index(drop=True)
            
        # Scale prices from $/MWh to $/kWh
        for col in ['rt_price', 'da_price']:
            if col in df.columns:
                df[col] = df[col] / 1000
                
        # Scale features
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.input_len - self.forecast_len + 1):
            X.append(df[feature_cols].iloc[i:i + self.input_len].values)
            y.append(df[target_col].iloc[i + self.input_len:i + self.input_len + self.forecast_len].values)
            
        X = np.stack(X)
        y = np.stack(y)
        
        # Create data loader
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        return loader, self.scaler
    
    def prepare_forecast_data(self, 
                            recent_data: pd.DataFrame,
                            feature_cols: list) -> np.ndarray:
        """
        Prepare data for forecasting.
        
        Args:
            recent_data (pd.DataFrame): Recent data for forecasting
            feature_cols (list): List of feature columns
            
        Returns:
            np.ndarray: Scaled input data for forecasting
        """
        # Scale features using the fitted scaler
        X = self.scaler.transform(recent_data[feature_cols].values)
        return X[None]  # Add batch dimension
    
    def inverse_transform(self, 
                         scaled_data: np.ndarray,
                         feature_cols: list) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            scaled_data (np.ndarray): Scaled data
            feature_cols (list): List of feature columns
            
        Returns:
            pd.DataFrame: Data in original scale
        """
        return pd.DataFrame(
            self.scaler.inverse_transform(scaled_data),
            columns=feature_cols
        ) 