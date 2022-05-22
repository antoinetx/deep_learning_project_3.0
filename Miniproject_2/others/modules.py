import math
from turtle import forward


class ReLU (object):
    def forward(self, input):
        self.x = max(0, input)
        return self.x
    def backward(self, gradwrtoutput):
        self.output = (self.x > 0)
        self.output = self.output *  gradwrtoutput
        return self.output
    
class Sigmoid (object):
    def forward(self, input):
        self.x = 1/(1 + math.e**(-input))
        return self.x
    def backward(self, gradwrtoutput):
        self.output = (self.x*(1-self.x))*gradwrtoutput
        return self.output
    
class MSE (object):
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        self.mse = (1/self.predictions.shape[1])*sum((self.predictions - self.targets)**2)
        return self.mse
    def backward(self, gradwrtoutput):
        self.output = (2/self.predictions.shape[1])*sum((self.predictions - self.targets))
        return self.output
    
""" class SGD (object):
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        self.mse = (1/self.predictions.shape[1])*sum((self.predictions - self.targets)**2)
        return self.mse
    def backward(self, gradwrtoutput):
        self.output = (2/self.predictions.shape[1])*sum((self.predictions - self.targets))
        return self.output """
    
class Sequential (object):
    def __init__(self, *input):
        self.layers = input
        
    def forward(self):
        for i in self.layers.shape[1]:
            x = i.forward(x)
        return x
        