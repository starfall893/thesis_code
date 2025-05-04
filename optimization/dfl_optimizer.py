import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from models.qlstm import QLSTM

class DFLOptimizer:
    """
    Decision-focused learning (DFL) optimizer using CVXPYLayer.
    
    This class implements end-to-end DFL by:
    1. Using QLSTM to predict single values for solar and RT prices
    2. Solving a differentiable LP using CVXPYLayer
    3. Computing regret-based loss for training
    
    Args:
        price_model (QLSTM): Price forecasting model
        solar_model (QLSTM): Solar forecasting model
        horizon (int): Optimization horizon
        battery_params (Dict[str, float]): Battery parameters
        ev_params (Dict[str, float]): Electric vehicle parameters
        learning_rate (float): Learning rate for optimization
    """
    
    def __init__(self,
                 price_model: QLSTM,
                 solar_model: QLSTM,
                 horizon: int,
                 battery_params: Dict[str, float],
                 ev_params: Dict[str, float],
                 learning_rate: float = 1e-3):
        self.price_model = price_model
        self.solar_model = solar_model
        self.horizon = horizon
        self.batt = battery_params
        self.ev = ev_params
        
        # Create differentiable LP layer
        self.lp_layer = self._create_lp_layer()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(price_model.parameters()) + list(solar_model.parameters()),
            lr=learning_rate
        )
    
    def _create_lp_layer(self) -> CvxpyLayer:
        """Create differentiable LP layer using CVXPYLayer."""
        T = self.horizon
        
        # Variables
        B = cp.Variable(T)  # Day-ahead bid
        G = cp.Variable(T)  # Real-time trading
        f = cp.Variable(T)  # Flexible load
        I = cp.Variable(T + 1)  # Inventory
        bc = cp.Variable(T, nonneg=True)  # Battery charging
        bd = cp.Variable(T, nonneg=True)  # Battery discharging
        ec = cp.Variable(T, nonneg=True)  # EV charging
        ed = cp.Variable(T, nonneg=True)  # EV discharging
        SoC_B = cp.Variable(T + 1)  # Battery state of charge
        SoC_E = cp.Variable(T + 1)  # EV state of charge
        
        # Parameters
        da_price = cp.Parameter(T)
        rt_price = cp.Parameter(T)
        solar = cp.Parameter(T)
        flexible_load = cp.Parameter(T)
        a_t = cp.Parameter(T)
        d_t = cp.Parameter(T)
        initial_soc_b = cp.Parameter()
        initial_soc_e = cp.Parameter()
        
        # Objective
        obj = (
            da_price @ B +
            rt_price @ (G - B) -
            self.batt['c_deg'] * cp.sum(bc + bd) -
            self.ev['c_deg'] * cp.sum(ec + ed)
        )
        
        # Constraints
        constraints = [
            I[0] == 0,
            SoC_B[0] == initial_soc_b * self.batt['E_max'],
            SoC_E[0] == initial_soc_e * self.ev['E_max']
        ]
        
        for t in range(T):
            # DSM inventory & flexible load
            constraints += [
                I[t+1] == I[t] + flexible_load[t] - f[t],
                f[t] >= 0,
                f[t] <= I[t] + flexible_load[t]
            ]
            
            # Battery constraints
            constraints += [
                SoC_B[t+1] == SoC_B[t] + self.batt['eta_ch'] * bc[t] - bd[t] / self.batt['eta_dis'],
                SoC_B[t+1] >= self.batt['SoC_min'] * self.batt['E_max'],
                SoC_B[t+1] <= self.batt['E_max'],
                bc[t] <= self.batt['P_max'],
                bd[t] <= self.batt['P_max']
            ]
            
            # EV constraints
            constraints += [
                SoC_E[t+1] == SoC_E[t] + self.ev['eta_ch'] * ec[t] - ed[t] / self.ev['eta_dis'],
                SoC_E[t+1] >= self.ev['E_min'],
                SoC_E[t+1] <= self.ev['E_max'],
                ec[t] <= self.ev['P_max'],
                ed[t] <= self.ev['P_max']
            ]
            
            # Power balance and trading
            Pexp = solar[t] + bd[t] + ed[t]
            Pimp = f[t] + bc[t] + ec[t] + (1 - a_t[t]) * d_t[t]
            
            constraints += [
                Pexp == Pimp + G[t],
                -Pimp <= B[t], B[t] <= Pexp,
                -Pimp <= G[t], G[t] <= Pexp
            ]
        
        constraints += [I[T] == 0]
        
        # Create CVXPYLayer
        return CvxpyLayer(
            cp.Problem(cp.Maximize(obj), constraints),
            parameters=[
                da_price, rt_price, solar, flexible_load,
                a_t, d_t, initial_soc_b, initial_soc_e
            ],
            variables=[B, G, f, I, bc, bd, ec, ed, SoC_B, SoC_E]
        )
    
    def compute_regret(self,
                      pred_solar: torch.Tensor,
                      pred_rt_price: torch.Tensor,
                      true_solar: torch.Tensor,
                      true_rt_price: torch.Tensor,
                      da_price: torch.Tensor,
                      flexible_load: torch.Tensor,
                      a_t: torch.Tensor,
                      d_t: torch.Tensor,
                      initial_soc_b: float,
                      initial_soc_e: float) -> torch.Tensor:
        """
        Compute regret-based loss.
        
        Args:
            pred_solar (torch.Tensor): Predicted solar generation
            pred_rt_price (torch.Tensor): Predicted RT prices
            true_solar (torch.Tensor): True solar generation
            true_rt_price (torch.Tensor): True RT prices
            da_price (torch.Tensor): Day-ahead prices
            flexible_load (torch.Tensor): Flexible load
            a_t (torch.Tensor): EV availability
            d_t (torch.Tensor): EV demand
            initial_soc_b (float): Initial battery SoC
            initial_soc_e (float): Initial EV SoC
            
        Returns:
            torch.Tensor: Regret loss
        """
        # Solve with predicted values
        pred_solution = self.lp_layer(
            da_price, pred_rt_price, pred_solar,
            flexible_load, a_t, d_t,
            torch.tensor(initial_soc_b), torch.tensor(initial_soc_e)
        )
        
        # Solve with true values (oracle)
        true_solution = self.lp_layer(
            da_price, true_rt_price, true_solar,
            flexible_load, a_t, d_t,
            torch.tensor(initial_soc_b), torch.tensor(initial_soc_e)
        )
        
        # Compute regret
        pred_profit = self._compute_profit(
            pred_solution, true_rt_price, true_solar,
            da_price, flexible_load, a_t, d_t
        )
        
        true_profit = self._compute_profit(
            true_solution, true_rt_price, true_solar,
            da_price, flexible_load, a_t, d_t
        )
        
        return true_profit - pred_profit
    
    def _compute_profit(self,
                       solution: Tuple[torch.Tensor, ...],
                       rt_price: torch.Tensor,
                       solar: torch.Tensor,
                       da_price: torch.Tensor,
                       flexible_load: torch.Tensor,
                       a_t: torch.Tensor,
                       d_t: torch.Tensor) -> torch.Tensor:
        """Compute profit for a given solution."""
        B, G, f, I, bc, bd, ec, ed, SoC_B, SoC_E = solution
        
        # Day-ahead revenue
        da_revenue = torch.sum(da_price * B)
        
        # Real-time profit
        rt_profit = torch.sum(rt_price * (G - B))
        
        # Degradation costs
        batt_deg = self.batt['c_deg'] * torch.sum(bc + bd)
        ev_deg = self.ev['c_deg'] * torch.sum(ec + ed)
        
        return da_revenue + rt_profit - batt_deg - ev_deg
    
    def optimize(self,
                test_data: pd.DataFrame,
                city_idx: int = 0) -> Dict[str, Any]:
        """
        Run optimization with DFL-enhanced forecasts.
        
        Args:
            test_data (pd.DataFrame): Test data
            city_idx (int): Index of the city to optimize for
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        # Generate forecasts
        with torch.no_grad():
            price_input = torch.tensor(
                test_data.tail(168)[['rt_price']].values[None],
                dtype=torch.float32
            )
            solar_input = torch.tensor(
                test_data.tail(168)[['solar_kwh']].values[None],
                dtype=torch.float32
            )
            
            pred_rt_price = self.price_model(price_input, city_idx)[0]
            pred_solar = self.solar_model(solar_input, city_idx)[0]
        
        # Get true values
        true_rt_price = torch.tensor(test_data['rt_price'].values[:24], dtype=torch.float32)
        true_solar = torch.tensor(test_data['solar_kwh'].values[:24], dtype=torch.float32)
        da_price = torch.tensor(test_data['da_price'].values[:24], dtype=torch.float32)
        flexible_load = torch.tensor(test_data['flexible_load'].values[:24], dtype=torch.float32)
        a_t = torch.tensor(test_data['a_t'].values[:24], dtype=torch.float32)
        d_t = torch.tensor(test_data['d_t'].values[:24], dtype=torch.float32)
        
        # Solve optimization
        solution = self.lp_layer(
            da_price, pred_rt_price, pred_solar,
            flexible_load, a_t, d_t,
            torch.tensor(0.5), torch.tensor(0.5)
        )
        
        # Compute profit
        profit = self._compute_profit(
            solution, true_rt_price, true_solar,
            da_price, flexible_load, a_t, d_t
        )
        
        return {
            'solution': solution,
            'profit': profit.item(),
            'forecasts': {
                'price': pred_rt_price.numpy(),
                'solar': pred_solar.numpy()
            }
        }
    
    def train_step(self,
                  train_data: pd.DataFrame,
                  city_idx: int = 0) -> float:
        """
        Perform one training step.
        
        Args:
            train_data (pd.DataFrame): Training data
            city_idx (int): Index of the city to train for
            
        Returns:
            float: Loss value
        """
        # Generate forecasts
        price_input = torch.tensor(
            train_data.tail(168)[['rt_price']].values[None],
            dtype=torch.float32
        )
        solar_input = torch.tensor(
            train_data.tail(168)[['solar_kwh']].values[None],
            dtype=torch.float32
        )
        
        pred_rt_price = self.price_model(price_input, city_idx)[0]
        pred_solar = self.solar_model(solar_input, city_idx)[0]
        
        # Get true values
        true_rt_price = torch.tensor(train_data['rt_price'].values[:24], dtype=torch.float32)
        true_solar = torch.tensor(train_data['solar_kwh'].values[:24], dtype=torch.float32)
        da_price = torch.tensor(train_data['da_price'].values[:24], dtype=torch.float32)
        flexible_load = torch.tensor(train_data['flexible_load'].values[:24], dtype=torch.float32)
        a_t = torch.tensor(train_data['a_t'].values[:24], dtype=torch.float32)
        d_t = torch.tensor(train_data['d_t'].values[:24], dtype=torch.float32)
        
        # Compute regret
        loss = self.compute_regret(
            pred_solar, pred_rt_price,
            true_solar, true_rt_price,
            da_price, flexible_load,
            a_t, d_t, 0.5, 0.5
        )
        
        # Update models
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 