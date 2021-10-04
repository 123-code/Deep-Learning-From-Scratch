import numpy as np
from numpy import ndarray

B = []
weights = []
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

# calculating model gradients.
def loss_gradients(Forward_info,weights):

    # computing partial derivatives
    # derivative with respect to y 

    d_y = 2*(Forward_info['y'] - Forward_info['prediction'])
    # bias derivative, a matrix of ones.
    d_o = np.ones_like(Forward_info['Output'])
    d_x = np.transpose(Forward_info('x'))
    d_b = np.ones.like(Forward_info['B'])
    dldn = d_y * d_o
    dLdW = np.dot(d_b,dldn)


    Loss_Gradients = {}
    Loss_Gradients['W'] = dLdW
    Loss_Gradients['B'] = dldn

    return loss_gradients

loss_grads = loss_gradients(Forward_info,weights)

for key in weights.keys():
    # applying weight update rule.
    
    weights[key] -= learning_rate * loss_grads[key]


def predict(X:ndarray,weights: Dict[str:ndarray]):
    # generating LR predictions.
    W_sum = np.dot(X,weights['x'])
    Prediction = W_sum + B[0]
    return prediction
    
print(predict(Forward_info['x'],weights))

# to compare predicted values with actual training data.
def mae(preds:ndarray,actuals:ndarray):
    return np.mean(np.abs(preds-actuals))
    

    






