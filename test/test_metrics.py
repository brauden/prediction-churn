import numpy as np
import torch
from src import metrics

def test_super_init():
    a = metrics.ChurnMetric()
    b = metrics.ChurnMetric(tensor_type="numpy")
    c = metrics.ChurnMetric(tensor_type="torch")

def test_churn():
    a = metrics.Churn()
    b = metrics.Churn(tensor_type="numpy", output_mode="proportion")
    c = metrics.Churn(tensor_type="torch", output_mode="count")
        
    v1 = np.array([1.,0.,1.,1.,0.])
    v2 = np.array([1.,0.,1.,1.,1.])
    v3 = np.array([0.,0.,0.,0.,0.])
    v4 = np.array([[1.],[0.],[1.],[1.],[0.]])
    v5 = np.array([[0.],[0.],[0.],[1.],[0.]])
    v6 = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    v7 = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
    ])
    v8 = np.array([
        [0.1, 0.3, 0.2, 0.4, 0],
        [0.1, 0., 0.2, 0., 9],
        [0.8, 0.2, 0.2, 0.75, 0],
    ])
    v1t = torch.Tensor(v1)
    v2t = torch.Tensor(v2)
    v3t = torch.Tensor(v3)
    v4t = torch.Tensor(v4)
    v5t = torch.Tensor(v5)
    v6t = torch.Tensor(v6)
    v7t = torch.Tensor(v7)
    v8t = torch.Tensor(v8)
    
    assert b(v1, v2) == 1/5
    assert b(v1, v3) == 3/5
    assert b(v4, v5) == 2/5
    assert b(v6, v7) == 1/3
    assert b(v8, v7) == 1/3
    assert c(v1t, v2t) == 1
    assert c(v1t, v3t) == 3
    assert c(v4t, v5t) == 2
    assert c(v6t, v7t) == 1
    assert c(v6t, v6t) == 0
    assert c(v8t, v7t) == 1
    