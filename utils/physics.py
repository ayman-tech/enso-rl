"""
Physics simulation functions for XRO climate model.
"""
import numpy as np
import xarray as xr


def xro_step(state, params, action, rng, step_idx, xro_debug=False):
    """
    Adds Action (Tuned) to the state and advances the XRO model by one month 
    using model.simulate().
    
    Args:
        state (np.ndarray): Current state (10 variables)
        params (dict): Parameters dict containing model, fit_ds, var_names, bounds, action_scale
        action (np.ndarray): Control action (9D)
        rng: Random number generator
        step_idx (int): Step index to determine current month
        xro_debug (bool): If True, return debug info
        
    Returns:
        next_state (np.ndarray): Next state or tuple of (control_actions, updated_state, next_state) if debug
    """
    model = params['model']
    current_month_idx = step_idx % 12  # current month index

    # Roll parameters to align with the current calendar month.
    # There are only 12 distinct rolls (one per month); rolling the xarray
    # Dataset every step is a major per-step cost, so prefer the precomputed
    # cache built in prepare_xro_parameters. The roll is read-only downstream
    # (only passed to model.simulate), so reusing cached views is equivalent
    # to rolling fresh each step. Fall back to an on-the-fly roll if a caller
    # built params without the cache.
    rolled = params.get('fit_ds_rolled')
    if rolled is not None:
        fit_ds = rolled[current_month_idx]
    else:
        fit_ds = params['fit_ds'].roll(
            cycle=-current_month_idx,
            roll_coords=False
        )

    var_names = params['var_names']
    bounds = params.get('bounds', {})
    action_scale_matrix = params.get('action_scale', None)
    # Monthly action scale: pick the column for the current month
    action_scale = np.array([row[current_month_idx] for row in action_scale_matrix])
    # 1. Apply Control Action
    control_actions = np.zeros_like(state)
    control_actions = action * action_scale
    control_actions = np.insert(control_actions, 0, 0)  # add 0 at beginning for nino action = 0

    # Update State & Safety clip
    updated_state = state + control_actions
    for i, var_name in enumerate(var_names):
        if var_name in bounds:
            min_val, max_val = bounds[var_name]
            updated_state[i] = np.clip(updated_state[i], min_val, max_val)

    # 2. Prepare Input for XRO.simulate
    data_dict = {
        name: float(val)
        for name, val in zip(var_names, updated_state)
    }
    input_ds = xr.Dataset(data_dict)

    # 3. Compute Dynamics using model.simulate
    #
    # XRO.gen_noise seeds and draws from the GLOBAL NumPy RNG (np.random.seed +
    # np.random.normal) on every call. That global RNG is also what PPO's mini-batch
    # shuffle uses, so an unguarded simulate would (a) clobber the shuffle ordering
    # (entangling the physics-noise axis with the shuffle axis) and (b) let the eval
    # environment perturb training. Snapshot and restore the global RNG state around
    # the call to isolate XRO's noise. Physics noise stays reproducible because the
    # per-step seed below comes from `rng` (the env's physics generator, axis #5).
    seed = int(rng.integers(0, 1_000_000))
    np_state = np.random.get_state()
    try:
        prediction_ds = model.simulate(
            fit_ds=fit_ds,
            X0_ds=input_ds,
            nyear=1,
            seed=seed
        )

        # 4. Extract the next state (t+1)
        next_state_ds = prediction_ds.isel(time=0)

        # Handle member dimension if it exists
        if 'member' in next_state_ds.coords or 'member' in next_state_ds.dims:
            next_state_ds = next_state_ds.isel(member=0)

        # Ensure next_state is always a 1D array of scalars
        next_state = np.array(
            [float(next_state_ds[name]) for name in var_names]
        )

    except Exception as e:
        print(f"Warning: Simulation failed at step {step_idx}. Resetting state. Error: {e}")
        return np.zeros_like(state)
    finally:
        # Always restore the global NumPy RNG, even if simulate raised partway
        # through gen_noise (which advances it).
        np.random.set_state(np_state)

    # 5. Safety Clipping and NaN check
    for i, var_name in enumerate(var_names):
        if var_name in bounds:
            min_val, max_val = bounds[var_name]
            next_state[i] = np.clip(next_state[i], min_val, max_val)

    # If we still have NaNs, replace with zeros
    if np.isnan(next_state).any():
        return np.zeros_like(state)

    if xro_debug:
        return control_actions, updated_state, next_state

    return next_state
