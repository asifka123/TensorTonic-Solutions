import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    loss = np.maximum(0,margin - y_true*y_score)

    if reduction == 'mean':
        return float(np.mean(loss))
    elif reduction == 'sum':
        return float(np.sum(loss))
    else:
        raise ValueError ('reduction must be mean or sum')