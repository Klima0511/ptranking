
import torch

def mseloss_objective(yhat, y, sample_weight=None):
    gradient = (y - yhat)
    hessian = torch.ones_like(yhat)

    return gradient, hessian