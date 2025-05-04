from typing import List, Dict, Any, Tuple
import numpy as np
from itertools import product
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import wasserstein_distance

class ScenarioBuilder:
    """
    Builds scenarios for stochastic optimization from QLSTM forecasts.
    
    This class implements the Clustered Quantile Scenario Reduction (CQSR) algorithm:
    1. Generates scenarios from quantile forecasts
    2. Clusters similar scenarios using hierarchical clustering
    3. Selects representative scenarios to minimize Wasserstein-1 distance
    
    Args:
        forecast_horizon (int): Number of time steps to forecast
        quantiles (list): List of quantiles used for prediction
        n_clusters (int): Number of clusters for scenario reduction
    """
    
    def __init__(self, forecast_horizon: int, quantiles: list, n_clusters: int = 5):
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.n_clusters = n_clusters
        
    def _compute_wasserstein_distance(self, scenario1: np.ndarray, scenario2: np.ndarray) -> float:
        """
        Compute Wasserstein-1 distance between two scenarios.
        
        Args:
            scenario1 (np.ndarray): First scenario
            scenario2 (np.ndarray): Second scenario
            
        Returns:
            float: Wasserstein-1 distance
        """
        return wasserstein_distance(scenario1, scenario2)
    
    def _cluster_scenarios(self, scenarios: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """
        Cluster scenarios using hierarchical clustering.
        
        Args:
            scenarios (List[np.ndarray]): List of scenarios to cluster
            
        Returns:
            Tuple[np.ndarray, List[int]]: Cluster labels and distances
        """
        # Compute pairwise distances
        n_scenarios = len(scenarios)
        distances = np.zeros((n_scenarios, n_scenarios))
        
        for i in range(n_scenarios):
            for j in range(i+1, n_scenarios):
                dist = self._compute_wasserstein_distance(scenarios[i], scenarios[j])
                distances[i,j] = distances[j,i] = dist
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            linkage='complete'
        )
        
        labels = clustering.fit_predict(distances)
        return labels, distances
    
    def _select_representatives(self, 
                              scenarios: List[np.ndarray],
                              labels: np.ndarray,
                              distances: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Select representative scenarios from each cluster.
        
        Args:
            scenarios (List[np.ndarray]): List of scenarios
            labels (np.ndarray): Cluster labels
            distances (np.ndarray): Pairwise distances
            
        Returns:
            Tuple[List[np.ndarray], List[float]]: Representative scenarios and their weights
        """
        representatives = []
        weights = []
        
        for cluster in range(self.n_clusters):
            # Get scenarios in this cluster
            cluster_scenarios = [s for i, s in enumerate(scenarios) if labels[i] == cluster]
            
            if not cluster_scenarios:
                continue
                
            # Compute distances to cluster center
            cluster_distances = []
            for i, s1 in enumerate(cluster_scenarios):
                dist = sum(self._compute_wasserstein_distance(s1, s2) 
                          for j, s2 in enumerate(cluster_scenarios) if i != j)
                cluster_distances.append(dist)
            
            # Select scenario with minimum distance to others
            rep_idx = np.argmin(cluster_distances)
            representatives.append(cluster_scenarios[rep_idx])
            
            # Weight proportional to cluster size
            weight = len(cluster_scenarios) / len(scenarios)
            weights.append(weight)
        
        return representatives, weights
    
    def build_price_scenarios(self, price_forecasts: Dict[int, tuple]) -> List[Dict[str, Any]]:
        """
        Build scenarios from price forecasts using CQSR.
        
        Args:
            price_forecasts (Dict[int, tuple]): Dictionary mapping time steps to (quantiles, values)
            
        Returns:
            List[Dict[str, Any]]: List of reduced price scenarios
        """
        # Generate all possible scenarios
        scenarios = []
        for i in range(len(self.quantiles)):
            scenario = np.array([price_forecasts[t][1][i] for t in range(self.forecast_horizon)])
            scenarios.append(scenario)
        
        # Cluster scenarios
        labels, distances = self._cluster_scenarios(scenarios)
        
        # Select representatives
        representatives, weights = self._select_representatives(scenarios, labels, distances)
        
        # Convert to dictionary format
        reduced_scenarios = []
        for rep, weight in zip(representatives, weights):
            scenario = {
                'rt_price': rep.tolist(),
                'weight': weight
            }
            reduced_scenarios.append(scenario)
            
        return reduced_scenarios
    
    def build_solar_scenarios(self, solar_forecasts: Dict[int, tuple]) -> List[Dict[str, Any]]:
        """
        Build scenarios from solar forecasts using CQSR.
        
        Args:
            solar_forecasts (Dict[int, tuple]): Dictionary mapping time steps to (quantiles, values)
            
        Returns:
            List[Dict[str, Any]]: List of reduced solar scenarios
        """
        # Generate all possible scenarios
        scenarios = []
        for i in range(len(self.quantiles)):
            scenario = np.array([solar_forecasts[t][1][i] for t in range(self.forecast_horizon)])
            scenarios.append(scenario)
        
        # Cluster scenarios
        labels, distances = self._cluster_scenarios(scenarios)
        
        # Select representatives
        representatives, weights = self._select_representatives(scenarios, labels, distances)
        
        # Convert to dictionary format
        reduced_scenarios = []
        for rep, weight in zip(representatives, weights):
            scenario = {
                'solar': rep.tolist(),
                'weight': weight
            }
            reduced_scenarios.append(scenario)
            
        return reduced_scenarios
    
    def build_joint_scenarios(self, 
                            price_forecasts: Dict[int, tuple],
                            solar_forecasts: Dict[int, tuple]) -> List[Dict[str, Any]]:
        """
        Build joint scenarios combining price and solar forecasts using CQSR.
        
        Args:
            price_forecasts (Dict[int, tuple]): Dictionary mapping time steps to (quantiles, values)
            solar_forecasts (Dict[int, tuple]): Dictionary mapping time steps to (quantiles, values)
            
        Returns:
            List[Dict[str, Any]]: List of reduced joint scenarios
        """
        # Generate all possible joint scenarios
        scenarios = []
        for i_price in range(len(self.quantiles)):
            for i_solar in range(len(self.quantiles)):
                price_scenario = np.array([price_forecasts[t][1][i_price] for t in range(self.forecast_horizon)])
                solar_scenario = np.array([solar_forecasts[t][1][i_solar] for t in range(self.forecast_horizon)])
                joint_scenario = np.concatenate([price_scenario, solar_scenario])
                scenarios.append(joint_scenario)
        
        # Cluster scenarios
        labels, distances = self._cluster_scenarios(scenarios)
        
        # Select representatives
        representatives, weights = self._select_representatives(scenarios, labels, distances)
        
        # Convert to dictionary format
        reduced_scenarios = []
        for rep, weight in zip(representatives, weights):
            # Split back into price and solar
            price_values = rep[:self.forecast_horizon]
            solar_values = rep[self.forecast_horizon:]
            
            scenario = {
                'rt_price': price_values.tolist(),
                'solar': solar_values.tolist(),
                'weight': weight
            }
            reduced_scenarios.append(scenario)
            
        return reduced_scenarios 