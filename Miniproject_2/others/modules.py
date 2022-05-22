import math
from turtle import forward


class ReLU (object):
    def forward(self, input):
        self.relu = input > 0
        self.x = self.relu*input
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
        self.mse = (((self.predictions - self.targets)**2).mean())
        return self.mse
    def backward(self, gradwrtoutput):
        self.output = 2*((self.predictions - self.targets).mean())
        self.output = self.output*gradwrtoutput
        return self.output
    
class Sequential (object):
    def __init__(self, *input):
        self.layers = input
        
    def forward(self, input):
        x = input
        for lay in self.layers:
            x = lay.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        out = gradwrtoutput
        for lay in self.layers:
            out = lay.backward(out)
        return out
            
            
            
            
class SGD (object):
    def __init__():
        pass
    def zero_grad():
        pass
    def step():
        pass