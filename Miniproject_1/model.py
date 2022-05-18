### For mini - project 1
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from others.autoencoder import Net

class Model () :
    def __init__( self ) :
    ## instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU
        self.mini_batch_size = 1000
        self.autoenc = Net().to(self.device) # init autoencoder
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.autoenc.parameters(), lr = 1e-2)
        
        pass

    def load_pretrained_model(self):
    ## This loads the parameters saved in bestmodel .pth into the model 
    
        best_model = torch.load('bestmodel.pth')
        self.autoenc.load_state_dict(best_model)
        
        pass

    def train(self , train_input , train_target)  :
    #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images.
    #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        
        train_input, train_target = train_input.to(self.device), train_target.to(self.device)
        
        nb_epochs = 40
        eta = 1e-1

        for e in range(nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.autoenc(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                acc_loss = acc_loss + loss.item()

                self.autoenc.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for p in self.autoenc.parameters():
                        p -= eta * p.grad

            print(e, acc_loss)
        pass

    def predict(self , test_input ) :
    #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
    #: returns a tensor of the size (N1 , C, H, W)
        output = self.autoenc(test_input)
        
        return output
