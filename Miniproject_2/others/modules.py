import math

import torch

from torch import empty , cat , arange
from torch.nn.functional import fold , unfold



class ReLU (object):
    def forward(self, input):
        self.relu = input > 0
        self.x = self.relu*input
        return self.x
    def backward(self, gradwrtoutput):
        self.output = (self.x > 0)
        self.output = self.output *  gradwrtoutput
        return self.output
    def param(self):
        return None
    
    
class Sigmoid (object):
    def forward(self, input):
        self.x = 1/(1 + math.e**(-input))
        return self.x
    def backward(self, gradwrtoutput):
        self.output = (self.x*(1-self.x))*gradwrtoutput
        return self.output
    
    def param(self):
        return None
        
    
    
class Upsample(object):
    def __init__(self, factor_size):
        self.kernel = torch.ones(factor_size,factor_size)
        self.factor_size = factor_size
        #print(self.kernel.shape)
        
    def forward(self,x):
        print('x', x.shape)
        self.b, self.channels, self.s1, self.s2 = x.shape
        print('kernel',self.kernel.shape)
        self.s3, self.s4 = self.factor_size,self.factor_size
        x = x.reshape(self.b, self.channels, self.s1, 1, self.s2, 1)
        self.kernel = self.kernel.reshape(1, self.s3, 1, self.s4)
        return (x * self.kernel).reshape(self.b, self.channels, self.s1 * self.s3,self.s2 * self.s4) 
    
    def backward(self,gradwrtoutput):
        dL_dS = gradwrtoutput
        dS_dX = self.kernel # [1, K, 1, K]
        
        #backward x
        dL_dS_reshape = dL_dS.reshape(self.b, self.channels, self.s1, self.s3, self.s2, self.s4) #[B, C, SI, K, SI, K]
        #print('dL_dS_reshape',dL_dS_reshape.shape)
        
        dL_dX = dL_dS_reshape /(dS_dX)
        
        #print(dL_dX.shape)
        dL_dX = dL_dX.permute(0,1,2,4,3,5)
        #print(dL_dX.shape)
        
        dL_dX_red = dL_dX[:,:,:,:,0,0]
        #print(dL_dX_red.shape)
    
        #print(dL_dX.reshape(self.b, self.channels, self.s1, self.s2).shape)
        
        return dL_dX_red
    
    def param(self):
        return None
    

    
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
    
class Convolution(object):
    def __init__(self, channels_input, channels_output, kernel_size, stride):
        
        self.weight = torch.empty(channels_output, channels_input, kernel_size, kernel_size).normal_()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels_output = channels_output
        self.channels_input = channels_input
        
        self.grad = torch.empty(channels_output, channels_input, kernel_size, kernel_size)

        #print('weight',self.weight.shape)
         
        
    def forward(self, imgs):
        
        _,_,H,W = imgs.shape
        #print('h w ',H,W)
        self.H = H
        self.W = W
        self.Hout = int((H - self.kernel_size)/self.stride + 1)
        self.Wout = int((H - self.kernel_size)/self.stride + 1)
        #print('Hout Wout', self.Hout, self.Wout)
        self.x = imgs
        #print('x', self.x.shape)
        self.x_unfolded = unfold(self.x, kernel_size = (self.kernel_size, self.kernel_size), stride = self.stride)
        #print('x unfold',self.x_unfolded.shape)
        #print('w shape', self.weight.view(self.channels_output, -1).shape)
        self.y = self.x_unfolded.transpose(1, 2).matmul(self.weight.view(self.channels_output, -1).t()).transpose(1, 2)
        #print('y',self.y.shape)
        self.y = fold(self.y, (int(self.Hout), int(self.Wout)),(1,1), stride = 1)
        #self.y = self.y.view(1,10,15,15)
        return self.y  #, self_x, self_weight
    

    def backward(self,gradwrtoutput):
        dL_dS = gradwrtoutput # [B, O, SO, SO]
        dS_dX = self.weight   # weight.shape [O, I, K, K]
        #print('dL_dS', dL_dS.shape)
        #print('dS_dX', dS_dX.shape)
        
        #define the size IxKxK
        inKerKer_size = self.channels_input*self.kernel_size*self.kernel_size
        #print('inKerKer_size',inKerKer_size)
        
        dL_dS_reshape = dL_dS.reshape(1,self.channels_output,-1) # [B, O, (SOxSO)]
        #print('dL_dS_reshape',dL_dS_reshape.shape)
        dS_dX_reshape = dS_dX.reshape(self.channels_output, -1).transpose(0,1)   # [O, (IxKxK)]^T
        #print('dS_dX_reshape',dS_dX_reshape.shape)
        
        
        #backward input
        dL_dX_reshape = dS_dX_reshape @ dL_dS_reshape # [B, (IxKxK), (SOxSO)]
        #print('dL_dX_reshape',dL_dX_reshape.shape)
        dL_dX = fold(dL_dX_reshape, kernel_size = (self.kernel_size, self.kernel_size), stride = self.stride, output_size = (self.W, self.H)) # [B, I, SI, SI]
        #print('dL_dX',dL_dX.shape)
        
        
        #backward weight
        dL_dS_reshape2 = dL_dS.reshape(self.channels_output, -1) # [O, (BxSOxSO)]
        dS_dW = self.x_unfolded # [B, (IxKxK), (SOxSO)]
        dS_dW_reshape = dS_dW.reshape(-1, inKerKer_size) # [(BxSOxSO), (IxKxK))] 
        dL_dW_reshape = dL_dS_reshape2 @ dS_dW_reshape # [O, (IxKxK)]
        dL_dW = dL_dW_reshape.view(self.channels_output,  self.channels_input, self.kernel_size, self.kernel_size) # [O, I, K, K] 
        
        self.grad = dL_dW
        #print('backward weight', self.grad.shape )
        #print('original weight', self.weight.shape )
        #backward bias
       
        
        return dL_dX  
    
    def param(self):
        #print('hello convolution')
        return self.weight, self.grad 
    
    
"""   
class ConvolutionTransposed(object):
    def __init__(self, channels_input, channels_output, kernel_size, stride):
        self.weight = torch.empty(channels_output, channels_input, kernel_size, kernel_size).normal_()
        
        self.channel_input = channels_input
        #print('input channels', self.channel_input)
        self.kernel_size = kernel_size
        self.stride = stride
        
        print('weight',self.weight.shape)
        
        
    def forward(self, imgs):
        #print('forward')
        _,_,H,W = imgs.shape
        H_out = (H - 1)*self.stride + self.kernel_size 
        W_out = (W - 1)*self.stride + self.kernel_size 
        
        #print('Hout Wout', H_out, W_out)
        
        self.x = imgs.permute(1, 2, 3, 0).reshape(self.channel_input, -1)
        #print('x',self.x.shape)
        self.y = (self.weight.reshape(self.channel_input, -1)).t().matmul(self.x)
        #print('y',self.y.shape)
        self.y = self.y.reshape(self.y.shape[0], -1, imgs.shape[0])
        #print('y2',self.y.shape)
        self.y = self.y.permute(2, 0, 1)
        
        #print(self.y.shape)
        self.y = fold( self.y, (H_out, W_out), kernel_size=(self.kernel_size,self.kernel_size), stride=self.stride)
        
        return self.y
"""
    
class Sequential (object):
    def __init__(self, *input):
        self.layers = input
        
    def forward(self, input):
        print('Sequential forward')
        x = input
        for lay in self.layers:
            print(lay)
            print('img', x.shape)
            x = lay.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        print('Sequential backward')
        out = gradwrtoutput
        for lay in reversed(self.layers):
            #print(lay)
            out = lay.backward(out)
        return out
    

class SGD (object):
    def __init__(self,layers, lr):
        self.lr = lr
        self.layers = layers
        
    def zero_grad(self):
        #print('zero grad')
        #print(self.layers.layers)
        
        for lay in self.layers.layers:
            #print(lay)
            values = lay.param()
            if lay.param() is not None:
                lay.grad = lay.grad.zero_()
          
    
    def step(self):
        
        for lay in self.layers.layers:
            print(lay)
            values = lay.param()
            if lay.param() is not None:
                #print('step', lay.weight)
                #print('step lr', self.lr*lay.grad )
                lay.weight = lay.weight + self.lr*lay.grad
                #print('step', lay.weight)
                
                