import numpy as np

class LinearRegression:
    def __init__(self,lr=0.001,n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # training function

    def fit(self,X,y):
        n_samples,n_features = X.shape

        #initializing weights randomly
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X,self.weights) + self.bias

            derivw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            derivb = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr * derivw
            self.bias -= self.lr * derivb





        


    # predicting function

    def predict(self,X,y):
        pass



