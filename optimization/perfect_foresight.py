import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class PerfectForesightOptimizer:
    """
    Perfect foresight (deterministic) optimization solver.
    
    This class implements the deterministic optimization problem
    where all future values are known with certainty.
    
    Args:
        battery_params (Dict[str, float]): Battery parameters
        ev_params (Dict[str, float]): Electric vehicle parameters
    """
    
    def __init__(self, 
                 battery_params: Dict[str, float],
                 ev_params: Dict[str, float]):
        self.batt = battery_params
        self.ev = ev_params
        
    def solve_day(self, 
                 df_day: pd.DataFrame,
                 initial_soc_b: float,
                 initial_soc_e: float) -> Tuple[pd.DataFrame, float]:
        """
        Solve the deterministic optimization problem for one day.
        
        Args:
            df_day (pd.DataFrame): Day's data with known values
            initial_soc_b (float): Initial battery state of charge
            initial_soc_e (float): Initial EV state of charge
            
        Returns:
            Tuple[pd.DataFrame, float]: Results dataframe and objective value
        """
        T = len(df_day)
        
        # Decision variables
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
        
        # Initial conditions
        constraints = [
            SoC_B[0] == initial_soc_b * self.batt['E_max'],
            SoC_E[0] == initial_soc_e * self.ev['E_max'],
            I[0] == 0
        ]
        
        # Time step constraints
        for t in range(T):
            row = df_day.iloc[t]
            S = row.solar_kwh
            LF = row.flexible_load
            A = row.a_t
            D = row.d_t
            
            # DSM inventory
            constraints += [
                I[t+1] == I[t] + LF - f[t],
                f[t] >= 0,
                f[t] <= I[t] + LF
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
            if A == 1:
                constraints += [
                    SoC_E[t+1] == SoC_E[t] + self.ev['eta_ch'] * ec[t] - ed[t] / self.ev['eta_dis'],
                    SoC_E[t+1] >= self.ev['E_min'],
                    SoC_E[t+1] <= self.ev['E_max'],
                    ec[t] <= self.ev['P_max'],
                    ed[t] <= self.ev['P_max']
                ]
            else:
                constraints += [
                    ec[t] == 0,
                    ed[t] == 0,
                    SoC_E[t+1] == SoC_E[t] - D
                ]
            
            # Power balance
            Pexp = S + bd[t] + ed[t]
            Pimp = f[t] + bc[t] + ec[t] + (1 - A) * D
            
            constraints += [
                Pexp == Pimp + G[t],
                -Pimp <= B[t], B[t] <= Pexp,
                -Pimp <= G[t], G[t] <= Pexp
            ]
        
        constraints += [I[T] == 0]
        
        # Objective
        obj = (
            df_day['da_price'].values @ B + 
            df_day['rt_price'].values @ (G - B)
            - self.batt['c_deg'] * cp.sum(bc + bd)
            - self.ev['c_deg'] * cp.sum(ec + ed)
        )
        
        # Solve
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver=cp.GLPK, verbose=False)
        
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Day {df_day.day.iloc[0]} infeasible: {prob.status}")
        
        # Record results
        records = []
        for t in range(T):
            row = df_day.iloc[t]
            records.append({
                'day': int(row.day),
                'hour': int(row.hour),
                'da_price': row.da_price,
                'rt_price': row.rt_price,
                'solar': row.solar_kwh,
                'flex_load': row.flexible_load,
                'a_t': row.a_t,
                'd_t': row.d_t,
                'bid': B.value[t],
                'rt_trade': G.value[t],
                'flex_served': f.value[t],
                'inv': I.value[t],
                'b_ch': bc.value[t],
                'b_dis': bd.value[t],
                'e_ch': ec.value[t],
                'e_dis': ed.value[t],
                'soc_b': SoC_B.value[t],
                'soc_e': SoC_E.value[t]
            })
        
        return pd.DataFrame(records), prob.value 