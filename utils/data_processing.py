"""
Data processing utilities for ENSO RL project.
"""
import xarray as xr
import numpy as np
from config.env_config import EnvConfig


def load_observational_data(data_path: str, train_start: str, train_end: str):
    """
    Load observational SST data from XRO dataset.
    
    Args:
        data_path (str): Path to XRO_indices_oras5.nc file
        train_start (str): Start date (YYYY-MM format)
        train_end (str): End date (YYYY-MM format)
        
    Returns:
        tuple: (full_dataset, train_dataset, var_names, bounds)
    """
    # Load full dataset
    obs_ds = xr.open_dataset(data_path)
    
    # Select training period
    train_ds = obs_ds.sel(time=slice(train_start, train_end))
    
    # Extract variable names and bounds
    var_names = list(train_ds.data_vars)
    bounds = {}
    for var in var_names:
        bounds[var] = (float(train_ds[var].data.min()), float(train_ds[var].data.max()))
    
    return obs_ds, train_ds, var_names, bounds


def prepare_xro_parameters(model, train_ds, var_names, bounds):
    """
    Prepare parameters for XRO physics function.
    
    Args:
        model: Fitted XRO model
        train_ds: Training dataset (xarray)
        var_names (list): Variable names
        bounds (dict): Bounds for each variable
        
    Returns:
        dict: Parameters dictionary with model, fit_ds, noise_cov, etc.
    """
    
    # Fit the model
    config = EnvConfig()
    fitted_params = model.fit_matrix(
        train_ds,
        maskb=["IOD"],
        maskNT=["T2", "TH"]
    )
    
    # Calculate noise covariance
    residuals = fitted_params['Y'] - fitted_params['Yfit']
    noise_cov = np.cov(residuals.values, rowvar=True)
    
    # Create params dictionary
    params = {
        'model': model,
        'fit_ds': fitted_params,
        'noise_cov': noise_cov,
        'var_names': var_names,
        'bounds': bounds,
        'dt': 1.0 / 12.0,
        'action_scale': config.action_scale,
        'reward_config': config.reward_config,
    }

    return params


def get_data(train_ds, var_names, target_year, target_month):
    """
    Returns the variables of given month and year from training dataset.
    
    Args:
        train_ds: Training dataset (xarray)
        var_names (list): Variable names
        target_year (int): Year in YYYY format
        target_month (int): Month (1-12)
        
    Returns:
        list: Values for the specified month/year
    """
    year_month_data = train_ds.sel(time=f'{target_year}-{target_month:02d}')
    
    year_month_var_values = []
    for var in var_names:
        year_month_var_values.append(float(year_month_data[var].values.item()))
    
    return year_month_var_values
