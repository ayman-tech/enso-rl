"""
Utility functions for ENSO RL project.
"""
import time
import warnings


def suppress_warnings():
    """Suppress common warnings from XRO package."""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.filterwarnings("ignore")


def timer(fn):
    """
    Decorator to measure function execution time.
    
    Args:
        fn: Function to time
        
    Returns:
        Wrapped function that prints execution time
    """
    def wrapper(*args, debug_mode=False, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        diff = end - start
        
        if debug_mode:
            if diff < 60:
                print(f"{fn.__name__}() took {diff:.2f} seconds")
            else:
                print(f"{fn.__name__}() took {diff//60} min {diff%60:.2f} sec")
        
        return result
    return wrapper
