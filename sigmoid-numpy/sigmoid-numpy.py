import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    
    x = np.asarray(x)
    result = 1/(1+np.exp(-x))
    if result.ndim == 0:
        return float(result)
    return result.astype(float)