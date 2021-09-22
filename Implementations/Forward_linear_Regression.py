import numpy as np
from numpy import ndarray

B = []
def forward_linear_regression(x_batch,y_batch,weights):
    assert x_batch.shape[0] == y_batch.shape[0]
    
    assert x_batch.shape[1] == weights.shape[0]
    assert B.shape[1] == 1

    # Forward pass operation
    output = np.dot(x_batch,weights)
    prediction = output + B[0]
    
   
    loss = np.mean(prediction - y_batch**2)

    Forward_info = {}

    Forward_info['x'] = x_batch
    Forward_info['y'] = y_batch
    Forward_info['Output'] = output
    Forward_info['Prediction'] = prediction

    return loss,Forward_info



