"""
Data processing utilities for ENSO RL project.
"""
import xarray as xr
import numpy as np


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


def calculate_state_bounds(train_ds, var_names, bounds_scale=1.0):
    """Return state bounds derived from observations and widened exactly once.

    Bounds are always recomputed from ``train_ds`` so callers cannot accidentally
    pass an already-widened dictionary and compound ``bounds_scale``.
    """
    if bounds_scale < 1.0:
        raise ValueError("bounds_scale must be >= 1.0 (1.0 = observed envelope)")

    observed_bounds = {
        var: (float(train_ds[var].data.min()), float(train_ds[var].data.max()))
        for var in var_names
    }
    if bounds_scale == 1.0:
        return observed_bounds

    return {
        var: ((lo + hi) / 2 - bounds_scale * (hi - lo) / 2,
              (lo + hi) / 2 + bounds_scale * (hi - lo) / 2)
        for var, (lo, hi) in observed_bounds.items()
    }


def prepare_xro_parameters(model, train_ds, var_names, *, config):
    """
    Prepare parameters for XRO physics function.
    
    Args:
        model: Fitted XRO model
        train_ds: Training dataset (xarray)
        var_names (list): Variable names
        config (EnvConfig): Configuration supplying action_scale, reward_config,
            clip_mode and bounds_scale. It is required so caller overrides cannot
            be silently replaced by a fresh default configuration.
        
    Returns:
        dict: Parameters dictionary with model, fit_ds, noise_cov, etc.
    """
    
    # Fit the model
    fitted_params = model.fit_matrix(
        train_ds,
        maskb=["IOD"],
        maskNT=["T2", "TH"]
    )
    
    # Calculate noise covariance
    residuals = fitted_params['Y'] - fitted_params['Yfit']
    noise_cov = np.cov(residuals.values, rowvar=True)
    
    # Precompute the 12 calendar-month rolls of fit_ds once. xro_step needs the
    # roll aligned to step % 12 every month; there are only 12 distinct rolls, so
    # caching them here avoids re-rolling the xarray Dataset on every env step
    # (a large per-step cost across all rollout-based analyses). The cached rolls
    # are identical to the per-step roll and are used read-only by model.simulate.
    fit_ds_rolled = [
        fitted_params.roll(cycle=-m, roll_coords=False) for m in range(12)
    ]

    # Derive the observed envelope here, then widen it exactly once. Keeping bound
    # ownership in this function prevents compounding when constructing variants.
    bounds = calculate_state_bounds(train_ds, var_names, config.bounds_scale)

    # Create params dictionary
    params = {
        'model': model,
        'fit_ds': fitted_params,
        'fit_ds_rolled': fit_ds_rolled,
        'noise_cov': noise_cov,
        'var_names': var_names,
        'bounds': bounds,
        'clip_mode': config.clip_mode,
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
