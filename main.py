import os
import pandas as pd
import torch
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
from models.qlstm import QLSTM
from models.trainer import QLSTMTrainer
from optimization.scenario_builder import ScenarioBuilder
from optimization.stochastic_opt import StochasticOptimizer
from optimization.perfect_foresight import PerfectForesightOptimizer
from optimization.dfl_optimizer import DFLOptimizer
from utils.data_utils import DataPreprocessor
import argparse
import yaml

# Load config
with open('config.yaml', 'r') as _f:
    config = yaml.safe_load(_f)

# Override constants from config
INPUT_LEN      = config['INPUT_LEN']
FORECAST_LEN   = config['FORECAST_LEN']
HIDDEN_DIM     = config['HIDDEN_DIM']
EPOCHS         = config['EPOCHS']
LEARNING_RATE  = config['LEARNING_RATE']
BATCH_SIZE     = config['BATCH_SIZE']
QUANTILES      = config['QUANTILES']
CITIES         = config['CITIES']
SCENARIOS      = config['SCENARIOS']
REQUIRED_COLUMNS = ['rt_price', 'da_price', 'solar_kwh', 'flexible_load', 'a_t', 'd_t', 'day', 'hour']

# Battery and EV parameters from config
BATTERY_PARAMS = config['BATTERY_PARAMS']
EV_PARAMS      = config['EV_PARAMS']

DEFAULT_SOLVER = None  # will set after cvxpy import

def parse_args():
    parser = argparse.ArgumentParser(
        description="Energy-arbitrage pipeline: pf/train/stoch/dfl")
    parser.add_argument(
        '--mode', choices=['pf','train','stoch','dfl','all'], default='all',
        help="Which stage to run")
    return parser.parse_args()

def validate_data(df: pd.DataFrame, is_training: bool = False) -> None:
    """Validate that dataframe has required columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols and not is_training:  # For test data, all columns are required
        raise ValueError(f"Missing required columns: {missing_cols}")
    elif is_training and any(col not in df.columns for col in ['rt_price', 'solar_kwh', 'day', 'hour']):
        raise ValueError("Training data missing essential columns")

def load_and_preprocess_data(city: str, scenario: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess data for a given city and scenario."""
    # Load training data
    train_file = f'train_data/{city}_train_set.csv'
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    train_data = pd.read_csv(train_file)
    
    # Standardize column names
    train_data = train_data.rename(columns={
        'Solar_kWh': 'solar_kwh',
        'A_t': 'a_t',
        'D_t': 'd_t'
    })
    
    # Add missing columns to training data with default values
    if 'flexible_load' not in train_data.columns:
        train_data['flexible_load'] = 0.0
    if 'a_t' not in train_data.columns:
        train_data['a_t'] = 1.0
    if 'd_t' not in train_data.columns:
        train_data['d_t'] = 0.0
    
    # Validate training data
    validate_data(train_data, is_training=True)
        
    # Load test data
    if scenario:
        test_file = f'test_data/{city}_{scenario}_test_set.csv'
    else:
        test_file = f'test_data/{city}_base_test_set.csv'  # Default to base scenario
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    test_data = pd.read_csv(test_file)
    
    # Standardize column names in test data
    test_data = test_data.rename(columns={
        'Solar_kWh': 'solar_kwh',
        'A_t': 'a_t',
        'D_t': 'd_t'
    })
    
    # Validate test data
    validate_data(test_data)
    
    return train_data, test_data

def create_output_dirs(city: str, scenario: str = None):
    """Create output directories for a given city and scenario."""
    dirs = [
        f'model_runs/{city}',  # Always create model directory
    ]
    
    if scenario:  # Only create scenario-specific directories if scenario is provided
        dirs.extend([
            f'outputs/{city}/{scenario}',
            f'forecasts/{city}/{scenario}'
        ])
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def save_model(model: torch.nn.Module, path: str, model_type: str):
    """Save model state dict with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{model_type}_model_{timestamp}.pt'
    torch.save(model.state_dict(), os.path.join(path, filename))

def train_models(train_data: pd.DataFrame, device: str = 'cuda'):
    """Train price and solar forecasting models."""
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(INPUT_LEN, FORECAST_LEN, BATCH_SIZE)
    
    # Prepare price data
    price_loader, price_scaler = preprocessor.prepare_data(
        train_data.copy(),
        ['rt_price'],
        'rt_price'
    )
    
    # Prepare solar data
    solar_loader, solar_scaler = preprocessor.prepare_data(
        train_data.copy(),
        ['solar_kwh'],
        'solar_kwh'
    )
    
    # Initialize models
    price_model = QLSTM(1, HIDDEN_DIM, FORECAST_LEN, QUANTILES, num_cities=len(CITIES))
    solar_model = QLSTM(1, HIDDEN_DIM, FORECAST_LEN, QUANTILES, num_cities=len(CITIES))
    
    # Initialize trainers
    price_trainer = QLSTMTrainer(price_model, LEARNING_RATE, device)
    solar_trainer = QLSTMTrainer(solar_model, LEARNING_RATE, device)
    
    # Train models
    print("\nTraining price model...")
    price_history = price_trainer.train(price_loader, price_loader, QUANTILES, EPOCHS)
    
    print("\nTraining solar model...")
    solar_history = solar_trainer.train(solar_loader, solar_loader, QUANTILES, EPOCHS)
    
    return {
        'price_model': price_model,
        'solar_model': solar_model,
        'price_scaler': price_scaler,
        'solar_scaler': solar_scaler,
        'price_history': price_history,
        'solar_history': solar_history
    }

def run_perfect_foresight(test_data: pd.DataFrame) -> Dict[str, Any]:
    """Run perfect foresight (deterministic) optimization."""
    optimizer = PerfectForesightOptimizer(BATTERY_PARAMS, EV_PARAMS)
    
    all_results = []
    total_profit = 0.0
    
    for day in sorted(test_data['day'].unique()):
        df_day = test_data[test_data['day'] == day].reset_index(drop=True)
        
        # Solve optimization
        results, profit = optimizer.solve_day(df_day, 0.5, 0.5)
        all_results.append(results)
        total_profit += profit
    
    return {
        'results': pd.concat(all_results, ignore_index=True),
        'total_profit': total_profit
    }

def save_forecasts(forecasts: dict, day: int, model_type: str, city: str, scenario: str):
    """Save forecasts to file."""
    forecast_dir = f'forecasts/{city}/{scenario}/{model_type}'
    os.makedirs(forecast_dir, exist_ok=True)
    
    # Save price forecasts
    price_df = pd.DataFrame({
        'hour': range(FORECAST_LEN),
        'q10': forecasts['price'][:, 0],
        'q50': forecasts['price'][:, 1],
        'q90': forecasts['price'][:, 2]
    })
    price_df.to_csv(f'{forecast_dir}/price_day_{day}.csv', index=False)
    
    # Save solar forecasts
    solar_df = pd.DataFrame({
        'hour': range(FORECAST_LEN),
        'q10': forecasts['solar'][:, 0],
        'q50': forecasts['solar'][:, 1],
        'q90': forecasts['solar'][:, 2]
    })
    solar_df.to_csv(f'{forecast_dir}/solar_day_{day}.csv', index=False)

def save_scenarios(scenarios: list, day: int, model_type: str, city: str, scenario: str):
    """Save scenarios to file."""
    scenario_dir = f'forecasts/{city}/{scenario}/{model_type}/scenarios'
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Convert scenarios to DataFrame
    scenario_data = []
    for i, scenario in enumerate(scenarios):
        for t in range(FORECAST_LEN):
            scenario_data.append({
                'scenario_id': i,
                'hour': t,
                'rt_price': scenario['rt_price'][t],
                'solar': scenario['solar'][t],
                'weight': scenario['weight']
            })
    
    scenario_df = pd.DataFrame(scenario_data)
    scenario_df.to_csv(f'{scenario_dir}/day_{day}.csv', index=False)

def run_stochastic_optimization(models: dict, test_data: pd.DataFrame, city: str, scenario: str, device: str = 'cuda'):
    """Run stochastic optimization with trained models."""
    # Initialize components
    scenario_builder = ScenarioBuilder(FORECAST_LEN, QUANTILES)
    optimizer = StochasticOptimizer(FORECAST_LEN, BATTERY_PARAMS, EV_PARAMS)
    
    all_results = []
    total_profit = 0.0
    
    for day in sorted(test_data['day'].unique()):
        df_day = test_data[test_data['day'] == day].reset_index(drop=True)
        
        # Prepare recent data for forecasting
        recent_data = test_data[test_data['day'] < day].tail(INPUT_LEN)
        
        # Generate forecasts
        price_input = torch.tensor(
            models['price_scaler'].transform(recent_data[['rt_price']].values)[None],
            dtype=torch.float32
        ).to(device)
        
        solar_input = torch.tensor(
            models['solar_scaler'].transform(recent_data[['solar_kwh']].values)[None],
            dtype=torch.float32
        ).to(device)
        
        with torch.no_grad():
            price_forecasts = models['price_model'](price_input).cpu().numpy()[0]
            solar_forecasts = models['solar_model'](solar_input).cpu().numpy()[0]
        
        # Save forecasts
        save_forecasts({
            'price': price_forecasts,
            'solar': solar_forecasts
        }, day, 'stochastic', city, scenario)
        
        # Build scenarios
        price_scenarios = {
            t: (QUANTILES, price_forecasts[:, t])
            for t in range(FORECAST_LEN)
        }
        
        solar_scenarios = {
            t: (QUANTILES, solar_forecasts[:, t])
            for t in range(FORECAST_LEN)
        }
        
        scenarios = scenario_builder.build_joint_scenarios(price_scenarios, solar_scenarios)
        
        # Save scenarios
        save_scenarios(scenarios, day, 'stochastic', city, scenario)
        
        # Solve optimization
        result = optimizer.solve_day_ahead(scenarios, 0.5, 0.5)
        
        # Evaluate recourse
        realized_scenario = {
            'rt_price': df_day['rt_price'].values,
            'da_price': df_day['da_price'].values,
            'solar': df_day['solar_kwh'].values,
            'flexible_load': df_day['flexible_load'].values,
            'a_t': df_day['a_t'].values,
            'd_t': df_day['d_t'].values
        }
        
        recourse_result = optimizer.evaluate_recourse(
            result['B'],
            realized_scenario,
            0.5,
            0.5
        )
        
        all_results.append({
            'day': day,
            'optimization_result': result,
            'recourse_result': recourse_result
        })
        
        total_profit += recourse_result['realized_profit']
    
    return {
        'results': all_results,
        'total_profit': total_profit
    }

def run_dfl_optimization(models: dict, train_data: pd.DataFrame, test_data: pd.DataFrame, city: str, scenario: str, device: str = 'cuda'):
    """Run DFL-enhanced optimization."""
    # Initialize components
    optimizer = StochasticOptimizer(FORECAST_LEN, BATTERY_PARAMS, EV_PARAMS)
    dfl_optimizer = DFLOptimizer(
        models['price_model'],
        models['solar_model'],
        FORECAST_LEN,  # Add horizon parameter
        BATTERY_PARAMS,
        EV_PARAMS,
        learning_rate=LEARNING_RATE
    )
    
    # Fine-tune model with DFL
    dfl_history = dfl_optimizer.fine_tune(train_data, test_data, QUANTILES)
    
    # Run optimization with fine-tuned model
    all_results = []
    total_profit = 0.0
    
    for day in sorted(test_data['day'].unique()):
        df_day = test_data[test_data['day'] == day].reset_index(drop=True)
        
        # Run optimization
        result = dfl_optimizer.optimize(df_day)
        
        # Save forecasts and scenarios from DFL optimizer
        save_forecasts(result['forecasts'], day, 'dfl', city, scenario)
        
        all_results.append({
            'day': day,
            'optimization_result': result['optimization_result'],
            'recourse_result': result['recourse_result']
        })
        
        total_profit += result['recourse_result']['realized_profit']
    
    return {
        'results': all_results,
        'total_profit': total_profit,
        'dfl_history': dfl_history
    }

def do_perfect_foresight():
    # 1. Run Perfect Foresight for all cities first
    print("\n=== Running Perfect Foresight for All Cities ===")
    for city in CITIES:
        print(f"\nProcessing {city}...")
        for scenario in SCENARIOS:
            print(f"\nRunning {scenario} scenario...")
            
            # Create output directories
            output_dirs = create_output_dirs(city, scenario)
            
            # Load test data
            _, test_data = load_and_preprocess_data(city, scenario)
            
            # Run perfect foresight
            print("Running Perfect Foresight (Deterministic)")
            pf_results = run_perfect_foresight(test_data)
            print(f"Total Profit: ${pf_results['total_profit']:.2f}")
            
            # Save results
            base_path = f'outputs/{city}/{scenario}'
            pf_results['results'].to_csv(
                f'{base_path}/perfect_foresight_results.csv', 
                index=False
            )

def do_training(device):
    # 2. Train models for all cities
    print("\n=== Training Models for All Cities ===")
    city_models = {}
    for city in CITIES:
        print(f"\nTraining models for {city}...")
        model_dir = f'model_runs/{city}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Load training data and train models
        train_data, _ = load_and_preprocess_data(city)
        models = train_models(train_data, device)
        
        # Save trained models and training history
        save_model(models['price_model'], model_dir, 'price')
        save_model(models['solar_model'], model_dir, 'solar')
        
        # Save training history
        price_history = models['price_history']['history']
        solar_history = models['solar_history']['history']
        
        # Convert nested dictionaries to flat structure for saving
        price_history_flat = {
            'epoch': list(range(len(price_history['train_loss']))),
            'train_loss': price_history['train_loss'],
            'val_loss': price_history['val_loss']
        }
        
        for q in QUANTILES:
            price_history_flat[f'train_mse_{q}'] = price_history['train_mse'][q]
            price_history_flat[f'val_mse_{q}'] = price_history['val_mse'][q]
            price_history_flat[f'train_mae_{q}'] = price_history['train_mae'][q]
            price_history_flat[f'val_mae_{q}'] = price_history['val_mae'][q]
        
        pd.DataFrame(price_history_flat).to_csv(
            f'{model_dir}/price_training_history.csv',
            index=False
        )
        
        solar_history_flat = {
            'epoch': list(range(len(solar_history['train_loss']))),
            'train_loss': solar_history['train_loss'],
            'val_loss': solar_history['val_loss']
        }
        
        for q in QUANTILES:
            solar_history_flat[f'train_mse_{q}'] = solar_history['train_mse'][q]
            solar_history_flat[f'val_mse_{q}'] = solar_history['val_mse'][q]
            solar_history_flat[f'train_mae_{q}'] = solar_history['train_mae'][q]
            solar_history_flat[f'val_mae_{q}'] = solar_history['val_mae'][q]
        
        pd.DataFrame(solar_history_flat).to_csv(
            f'{model_dir}/solar_training_history.csv',
            index=False
        )
        
        # Save model parameters
        with open(f'{model_dir}/model_params.json', 'w') as f:
            json.dump({
                'price_model': models['price_history']['model_params'],
                'solar_model': models['solar_history']['model_params']
            }, f, indent=2)
        
        city_models[city] = models
    return city_models

def do_stochastic(device, city_models):
    # 3. Run Stochastic Optimization for all cities
    print("\n=== Running Stochastic Optimization for All Cities ===")
    for city in CITIES:
        print(f"\nProcessing {city}...")
        for scenario in SCENARIOS:
            print(f"\nRunning {scenario} scenario...")
            
            # Create output directories
            output_dirs = create_output_dirs(city, scenario)
            
            # Load test data
            _, test_data = load_and_preprocess_data(city, scenario)
            
            # Run stochastic optimization
            print("Running Stochastic Optimization")
            stoch_results = run_stochastic_optimization(
                city_models[city],
                test_data, 
                city, 
                scenario, 
                device
            )
            print(f"Total Profit: ${stoch_results['total_profit']:.2f}")
            
            # Save results
            base_path = f'outputs/{city}/{scenario}'
            
            # Save summary results
            stoch_df = pd.DataFrame([
                {
                    'day': r['day'],
                    'expected_profit': r['optimization_result']['profit'],
                    'realized_profit': r['recourse_result']['realized_profit'],
                    'da_revenue': r['recourse_result']['da_revenue'],
                    'rt_profit': r['recourse_result']['rt_profit']
                }
                for r in stoch_results['results']
            ])
            stoch_df.to_csv(f'{base_path}/stochastic_results.csv', index=False)
            
            # Save hourly metrics
            hourly_metrics = []
            for r in stoch_results['results']:
                hourly_metrics.extend(r['optimization_result']['hourly_metrics'])
            pd.DataFrame(hourly_metrics).to_csv(
                f'{base_path}/stochastic_hourly_metrics.csv',
                index=False
            )

def do_dfl(device, city_models):
    # 4. Run DFL Optimization for all cities
    print("\n=== Running DFL Optimization for All Cities ===")
    for city in CITIES:
        print(f"\nProcessing {city}...")
        for scenario in SCENARIOS:
            print(f"\nRunning {scenario} scenario...")
            
            # Create output directories
            output_dirs = create_output_dirs(city, scenario)
            
            # Load data
            train_data, test_data = load_and_preprocess_data(city, scenario)
            
            # Run DFL optimization
            print("Running DFL-enhanced Optimization")
            dfl_results = run_dfl_optimization(
                city_models[city],
                train_data, 
                test_data, 
                city, 
                scenario, 
                device
            )
            print(f"Total Profit: ${dfl_results['total_profit']:.2f}")
            
            # Save results
            base_path = f'outputs/{city}/{scenario}'
            
            # Save summary results
            dfl_df = pd.DataFrame([
                {
                    'day': r['day'],
                    'expected_profit': r['optimization_result']['profit'],
                    'realized_profit': r['recourse_result']['realized_profit'],
                    'da_revenue': r['recourse_result']['da_revenue'],
                    'rt_profit': r['recourse_result']['rt_profit']
                }
                for r in dfl_results['results']
            ])
            dfl_df.to_csv(f'{base_path}/dfl_results.csv', index=False)
            
            # Save hourly metrics
            hourly_metrics = []
            for r in dfl_results['results']:
                hourly_metrics.extend(r['optimization_result']['hourly_metrics'])
            pd.DataFrame(hourly_metrics).to_csv(
                f'{base_path}/dfl_hourly_metrics.csv',
                index=False
            )
            
            # Save DFL training history
            pd.DataFrame(dfl_results['dfl_history']).to_csv(
                f'{base_path}/dfl_training_history.csv',
                index=False
            )

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Pre-create all needed directories
    base_dirs = ['model_runs','outputs','forecasts'] + \
        [f'outputs/{c}/{s}' for c in CITIES for s in SCENARIOS] + \
        [f'forecasts/{c}/{s}' for c in CITIES for s in SCENARIOS]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)
    
    # Run stages per args.mode, propagating model dict
    city_models = {}
    if args.mode in ('pf','all'):
        do_perfect_foresight()
    if args.mode in ('train','all'):
        city_models = do_training(device)
    if args.mode in ('stoch','all'):
        if not city_models:
            raise RuntimeError("Need to train models first; run with --mode train or all")
        do_stochastic(device, city_models)
    if args.mode in ('dfl','all'):
        if not city_models:
            raise RuntimeError("Need to train models first; run with --mode train or all")
        do_dfl(device, city_models)

if __name__ == "__main__":
    main() 