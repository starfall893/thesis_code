import cvxpy as cp
import numpy as np
from typing import Dict, Any, List, Tuple

class StochasticOptimizer:
    """
    Stochastic optimization solver for energy trading.
    
    This class implements the two-stage stochastic optimization problem
    for day-ahead bidding and real-time trading, with support for
    multiple cities and load scenarios.
    
    Args:
        horizon (int): Optimization horizon
        battery_params (Dict[str, float]): Battery parameters
        ev_params (Dict[str, float]): Electric vehicle parameters
        num_cities (int): Number of cities to optimize for
        load_scenarios (List[str]): List of load scenarios (e.g., ['base', 'high', 'low'])
    """
    
    def __init__(self, 
                 horizon: int,
                 battery_params: Dict[str, float],
                 ev_params: Dict[str, float],
                 num_cities: int = 1,
                 load_scenarios: List[str] = ['base']):
        self.horizon = horizon
        self.batt = battery_params
        self.ev = ev_params
        self.num_cities = num_cities
        self.load_scenarios = load_scenarios
        
    def solve_day_ahead(self, 
                       scenarios: List[Dict[str, Any]],
                       initial_soc_b: float,
                       initial_soc_e: float,
                       city_idx: int = 0,
                       load_scenario: str = 'base') -> Dict[str, Any]:
        """
        Solve the day-ahead optimization problem.
        
        Args:
            scenarios (List[Dict[str, Any]]): List of scenarios
            initial_soc_b (float): Initial battery state of charge
            initial_soc_e (float): Initial EV state of charge
            city_idx (int): Index of the city to optimize for
            load_scenario (str): Load scenario to use
            
        Returns:
            Dict[str, Any]: Optimization results including bids and expected profit
        """
        T = self.horizon
        B = cp.Variable(T)  # Day-ahead bid
        
        # First-stage objective (day-ahead revenue)
        obj_first = 0
        for t in range(T):
            obj_first += scenarios[0]['da_price'][t] * B[t]
            
        # Second-stage objective (recourse)
        recourse = 0
        constraints = []
        
        # Store hourly variables for each scenario
        hourly_vars = []
        
        for sc in scenarios:
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
            constraints += [
                I[0] == 0,
                SoC_B[0] == initial_soc_b * self.batt['E_max'],
                SoC_E[0] == initial_soc_e * self.ev['E_max']
            ]
            
            hourly_vars.append({
                'G': G, 'f': f, 'I': I, 'bc': bc, 'bd': bd,
                'ec': ec, 'ed': ed, 'SoC_B': SoC_B, 'SoC_E': SoC_E
            })
            
            for t in range(T):
                # Load and solar constraints
                LF = sc['flexible_load'][t]
                A = sc['a_t'][t]
                D = sc['d_t'][t]
                S = sc['solar'][t]
                RT = sc['rt_price'][t]
                
                # DSM inventory & flexible load
                constraints += [
                    I[t+1] == I[t] + LF - f[t],
                    f[t] >= 0,
                    f[t] <= I[t] + LF,
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
                
                # Power balance and trading
                Pexp = S + bd[t] + ed[t]
                Pimp = f[t] + bc[t] + ec[t] + (1 - A) * D
                
                constraints += [
                    Pexp == Pimp + G[t],
                    -Pimp <= B[t], B[t] <= Pexp,
                    -Pimp <= G[t], G[t] <= Pexp
                ]
                
                # Add to recourse objective
                recourse += sc['weight'] * (
                    RT * (G[t] - B[t])
                    - self.batt['c_deg'] * (bc[t] + bd[t])
                    - self.ev['c_deg'] * (ec[t] + ed[t])
                )
            
            constraints += [I[T] == 0]
        
        # Solve the problem
        prob = cp.Problem(cp.Maximize(obj_first + recourse), constraints)
        prob.solve(solver=cp.GLPK_MI, verbose=False)
        
        # Extract results
        da_revenue = sum(scenarios[0]['da_price'][t] * B.value[t] for t in range(T))
        rt_revenue = prob.value - da_revenue
        
        # Extract hourly metrics for each scenario
        hourly_metrics = []
        for i, sc in enumerate(scenarios):
            vars = hourly_vars[i]
            for t in range(T):
                hourly_metrics.append({
                    'scenario_id': i,
                    'hour': t,
                    'da_price': sc['da_price'][t],
                    'rt_price': sc['rt_price'][t],
                    'solar': sc['solar'][t],
                    'flexible_load': sc['flexible_load'][t],
                    'a_t': sc['a_t'][t],
                    'd_t': sc['d_t'][t],
                    'da_bid': B.value[t],
                    'rt_trading': vars['G'].value[t],
                    'flexible_load_used': vars['f'].value[t],
                    'inventory': vars['I'].value[t],
                    'battery_charge': vars['bc'].value[t],
                    'battery_discharge': vars['bd'].value[t],
                    'battery_soc': vars['SoC_B'].value[t],
                    'ev_charge': vars['ec'].value[t],
                    'ev_discharge': vars['ed'].value[t],
                    'ev_soc': vars['SoC_E'].value[t],
                    'power_export': sc['solar'][t] + vars['bd'].value[t] + vars['ed'].value[t],
                    'power_import': vars['f'].value[t] + vars['bc'].value[t] + vars['ec'].value[t] + (1 - sc['a_t'][t]) * sc['d_t'][t],
                    'da_revenue': sc['da_price'][t] * B.value[t],
                    'rt_revenue': sc['rt_price'][t] * (vars['G'].value[t] - B.value[t]),
                    'battery_degradation': self.batt['c_deg'] * (vars['bc'].value[t] + vars['bd'].value[t]),
                    'ev_degradation': self.ev['c_deg'] * (vars['ec'].value[t] + vars['ed'].value[t])
                })
        
        return {
            "B": B.value,
            "profit": prob.value,
            "da_revenue": da_revenue,
            "rt_revenue": rt_revenue,
            "status": prob.status,
            "city_idx": city_idx,
            "load_scenario": load_scenario,
            "hourly_metrics": hourly_metrics
        }
    
    def evaluate_recourse(self,
                         B: np.ndarray,
                         scenario: Dict[str, Any],
                         initial_soc_b: float,
                         initial_soc_e: float,
                         city_idx: int = 0,
                         load_scenario: str = 'base') -> Dict[str, Any]:
        """
        Evaluate the recourse problem for a given scenario.
        
        Args:
            B (np.ndarray): Day-ahead bids
            scenario (Dict[str, Any]): Realized scenario
            initial_soc_b (float): Initial battery state of charge
            initial_soc_e (float): Initial EV state of charge
            city_idx (int): Index of the city to evaluate for
            load_scenario (str): Load scenario to use
            
        Returns:
            Dict[str, Any]: Recourse results including realized profit
        """
        T = self.horizon
        G = cp.Variable(T)
        f = cp.Variable(T)
        I = cp.Variable(T + 1)
        bc = cp.Variable(T, nonneg=True)
        bd = cp.Variable(T, nonneg=True)
        ec = cp.Variable(T, nonneg=True)
        ed = cp.Variable(T, nonneg=True)
        SoC_B = cp.Variable(T + 1)
        SoC_E = cp.Variable(T + 1)
        
        constraints = [
            I[0] == 0,
            SoC_B[0] == initial_soc_b * self.batt['E_max'],
            SoC_E[0] == initial_soc_e * self.ev['E_max']
        ]
        
        obj = 0
        for t in range(T):
            LF = scenario['flexible_load'][t]
            A = scenario['a_t'][t]
            D = scenario['d_t'][t]
            S = scenario['solar'][t]
            RT = scenario['rt_price'][t]
            
            constraints += [
                I[t+1] == I[t] + LF - f[t],
                f[t] >= 0,
                f[t] <= I[t] + LF,
                SoC_B[t+1] == SoC_B[t] + self.batt['eta_ch'] * bc[t] - bd[t] / self.batt['eta_dis'],
                SoC_B[t+1] >= self.batt['SoC_min'] * self.batt['E_max'],
                SoC_B[t+1] <= self.batt['E_max'],
                bc[t] <= self.batt['P_max'],
                bd[t] <= self.batt['P_max'],
            ]
            
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
            
            Pexp = S + bd[t] + ed[t]
            Pimp = f[t] + bc[t] + ec[t] + (1 - A) * D
            
            constraints += [
                Pexp == Pimp + G[t],
                -Pimp <= B[t], B[t] <= Pexp,
                -Pimp <= G[t], G[t] <= Pexp
            ]
            
            obj += RT * (G[t] - B[t]) \
                 - self.batt['c_deg'] * (bc[t] + bd[t]) \
                 - self.ev['c_deg'] * (ec[t] + ed[t])
        
        constraints += [I[T] == 0]
        da_revenue = sum(scenario['da_price'][t] * B[t] for t in range(T))
        
        prob = cp.Problem(cp.Maximize(da_revenue + obj), constraints)
        prob.solve(solver=cp.GLPK_MI, verbose=False)
        
        return {
            "realized_profit": prob.value,
            "da_revenue": da_revenue,
            "rt_profit": prob.value - da_revenue,
            "status": prob.status,
            "city_idx": city_idx,
            "load_scenario": load_scenario
        } 