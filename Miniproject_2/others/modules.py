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
        self.grad = self.output *  gradwrtoutput
        return self.grad
    def param(self):
        return None
    
    
class Sigmoid (object):
    def forward(self, input):
        self.x = 1/(1 + math.e**(-input))
        return self.x
    def backward(self, gradwrtoutput):
        self.grad = (self.x*(1-self.x))*gradwrtoutput
        return self.grad
    
    def param(self):
        return None
        
    
    
class Upsample(object):
    def __init__(self, factor_size):
        self.kernel = torch.ones(factor_size,factor_size)
        self.factor_size = factor_size
        #print(self.kernel.shape)
        
    def forward(self,x):
        #print('x', x.shape)
        self.b, self.channels, self.s1, self.s2 = x.shape
        #print('kernel',self.kernel.shape)
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
        self.grad = dL_dX_red
    
        #print(dL_dX.reshape(self.b, self.channels, self.s1, self.s2).shape)
        
        return self.grad
    
    def param(self):
        return None
    

    
class MSE (object):
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        self.mse = (((self.predictions - self.targets)**2).mean())
        return self.mse
    def backward(self): #, gradwrtoutput):
        self.output = 2*((self.predictions - self.targets))#.mean())
        #self.output = self.output
        return self.output
    
class Convolution(object):
    def __init__(self, channels_input, channels_output, kernel_size, dilation, padding, stride):
        self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = torch.empty(channels_output, channels_input, kernel_size, kernel_size).normal_()
        self.bias = torch.empty(channels_output).normal_()
        self.dilation = 0
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels_output = channels_output
        self.channels_input = channels_input
        
        self.grad = torch.empty(channels_output, channels_input, kernel_size, kernel_size)
        self.grad_bias = torch.empty(channels_output)

        #print('weight',self.weight.shape)
         
        
    def forward(self, imgs):
        
        _,_,H,W = imgs.shape
        #print('h w ',H,W)
        self.H = H
        self.W = W
        self.Hout = int((H - self.kernel_size + self.padding*2)/self.stride + 1)
        self.Wout = int((H - self.kernel_size + self.padding*2)/self.stride + 1)
        #print('Hout Wout', self.Hout, self.Wout)
        self.x = imgs
        #print('x', self.x.shape)
        self.x_unfolded = unfold(self.x, kernel_size = (self.kernel_size, self.kernel_size),padding = self.padding, stride = self.stride)
        #print('x unfold',self.x_unfolded.shape)
        #print('w shape', self.weight.view(self.channels_output, -1).shape)
        self.y = self.x_unfolded.transpose(1, 2).matmul(self.weight.view(self.channels_output, -1).t()).transpose(1, 2)
        #print('y',self.y.shape)
        self.y = fold(self.y, (int(self.Hout), int(self.Wout)),(1,1), stride = 1)
        #self.y = self.y.view(1,10,15,15)
        return self.y + self.bias.reshape(1,-1,1,1)
    

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
        dL_dX = fold(dL_dX_reshape, kernel_size = (self.kernel_size, self.kernel_size), padding = self.padding, stride = self.stride, output_size = (self.W, self.H)) # [B, I, SI, SI]
        #print('dL_dX',dL_dX.shape)
        
        
        #backward weight
        dL_dS_reshape2 = dL_dS.reshape(self.channels_output, -1).to(self.device) # [O, (BxSOxSO)]
        dS_dW = self.x_unfolded # [B, (IxKxK), (SOxSO)]
        dS_dW_reshape = dS_dW.reshape(-1, inKerKer_size) # [(BxSOxSO), (IxKxK))] 
        dL_dW_reshape = dL_dS_reshape2 @ dS_dW_reshape # [O, (IxKxK)]
        dL_dW = dL_dW_reshape.view(self.channels_output,  self.channels_input, self.kernel_size, self.kernel_size) # [O, I, K, K] 
        
        self.grad = dL_dW
        #print('backward weight', self.grad.shape )
        #print('original weight', self.weight.shape )
        #backward bias
        
        #backward bias
        dS_dB_reshape = 1 + torch.empty(self.Hout*self.Wout ).normal_() # [BxSOxSO]
        dL_dB = dL_dS_reshape2 @ dS_dB_reshape # [O, (BxSOxSO)] [BxSOxSO]
        
        self.grad_bias = dL_dB
        #print('dL_dB', self.grad_bias.shape)
        
        
        return dL_dX  
    
    def param(self):
        #print('hello convolution')
        return self.weight, self.grad, self.bias, self.grad_bias
    
    
class ConvolutionTransposed(object):
    def __init__(self,channels_input, channels_output, kernel_size, dilation, padding, stride):
        self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.weight = torch.empty(channels_output, channels_input, kernel_size, kernel_size).normal_()
        self.grad = torch.empty(channels_output, channels_input, kernel_size, kernel_size)
        
        self.bias = torch.empty(channels_output)
        self.grad_bias = torch.empty(channels_output)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels_output = channels_output
        self.channels_input = channels_input
        
        self.padding = padding
        self.dilation = 0
        
    def forward(self,x):
        
        B,I,SI,SI = x.shape
        SO = (SI -1)*self.stride + self.kernel_size - 2*self.padding
        self.H = SI
        self.W = SI
        self.x = x
        self.x_reshape = self.x.reshape(B,I,-1) # [B,I,SI,SI]
        self.weight_reshape = self.weight.permute(1,0,2,3).reshape(I,-1) # [I,(OxKxK)]
        
        self.y_reshape = self.weight_reshape.T @ self.x_reshape # [B, OxKxK, SIxSI]
        #print('y', self.y_reshape.shape)
        
        self.y = fold(self.y_reshape, kernel_size =(self.kernel_size,self.kernel_size),padding = self.padding, stride = self.stride, output_size=(SO,SO))
        

        
        
        return self.y
    
    def backward(self,gradwrtoutput):
        
        dL_dS = gradwrtoutput # [B, O, SO, SO]
        #print('dL_dS', dL_dS.shape)
        dS_dX = self.weight # [O,I,K,K]
        
        #backward Input
        dL_dS_unfold = unfold(dL_dS, kernel_size =(self.kernel_size,self.kernel_size), padding = self.padding, stride = self.stride)#[B, OxKxK, SIxSI]
        #print('unfold', dL_dS_unfold.shape )
       
        dS_dX_reshape = dS_dX.reshape(self.channels_input,-1) # [I,OxKxK]
        #print('dS_dX_reshape', dS_dX_reshape.shape)
        
        dL_dX_reshape = dS_dX_reshape @ dL_dS_unfold # [B,I,SIxSI]
        dL_dX =  dL_dX_reshape.view(1,self.channels_input,self.H, self.W)
        
        #backward weight
        #print('dL_dS_unfold', dL_dS_unfold.shape)
        dL_dS_unfold2 = dL_dS_unfold.permute(1,0,2).reshape(self.channels_output*self.kernel_size*self.kernel_size,-1) #[OxKxK, BxSIxSI]
        #print('unfold 2', dL_dS_unfold2.shape )
        dS_dW_reshape = self.x.reshape(self.channels_input, -1 ) #[I, BxSIxSI]
        
        dL_dW_reshape =  dS_dW_reshape @ dL_dS_unfold2.T # [I, OxKxK]
        
        dL_dW = dL_dW_reshape.reshape(self.channels_input, self.channels_output, self.kernel_size, self.kernel_size ).permute(1,0,2,3)
        
        self.grad = dL_dW
        #print('self.grad',self.grad.shape)
        
        return dL_dX
    
    def param(self):
        return self.weight, self.grad, self.bias, self.grad_bias 
        
    
class Sequential (object):
    def __init__(self, *input):
        self.layers = input
        
    def forward(self, input):
        #print('Sequential forward')
        x = input
        for lay in self.layers:
            #print(lay)
            #print('img', x.shape)
            x = lay.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        #print('Sequential backward')
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
                lay.grad_bias = lay.grad_bias.zero_()
          
    
    def step(self):
        
        for lay in self.layers.layers:
            #print(lay)
            values = lay.param()
            if lay.param() is not None:
                #print('step', lay.weight.shape)
                #print('step lr', lay.grad.shape )
                lay.weight = lay.weight - self.lr*lay.grad
                lay.bias = lay.bias - self.lr*lay.grad_bias
                #print('step', lay.weight)
                
"""
class Adam(object)
    def __init__(self,layers, lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.lr = lr
        self.layers = layers
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
    def zero_grad(self):
        #print('zero grad')
        #print(self.layers.layers)
        
        for lay in self.layers.layers:
            #print(lay)
            values = lay.param()
            if lay.param() is not None:
                lay.grad = lay.grad.zero_()
                lay.grad_bias = lay.grad_bias.zero_()
          
    
    def step(self):
        
        for lay in self.layers.layers:
            #print(lay)
            values = lay.param()
            if lay.param() is not None:
                #print('step', lay.weight.shape)
                #print('step lr', lay.grad.shape )
                vdw = self.beta_1*vdw + (1 - self.beta_1)*lay.grad
                sdw = self.beta_2*sdw + (1 - self.beta_2)*((lay.grad)**2)
                
                lay.weight = lay.weight - self.epsilon*vdw/(sdw +.self.epsilon).sqrt()
                
                lay.bias = lay.bias - self.lr*lay.grad_bias
                #print('step', lay.weight)
                
"""
                