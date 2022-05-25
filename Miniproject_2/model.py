### For mini - project 1
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import others.modules as m

class Model () :
    def __init__( self ) -> None :
    ## instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU
        self.mini_batch_size = 1000
        Relu = m.ReLU()
        self.sequence = m.Sequential(""" m.Conv(stride = 2), Relu, m.Conv(stride = 2), Relu, Upsampling, Relu, Upsampling, Sigmoid """)
        self.autoenc = self.sequence.to(self.device)
        self.criterion = m.MSE()
        #self.optimizer = m.SGD(self.autoenc.parameters(), lr = 1e-2)
        pass

    def load_pretrained_model(self) -> None :
    ## This loads the parameters saved in bestmodel .pth into the model 
        best_model = torch.load('bestmodel.pth')
        self.sequence.load_state_dict(best_model)
        pass

    def train(self , train_input , train_target) -> None :
        
        train_input, train_target = train_input.to(self.device), train_target.to(self.device)
        
        nb_epochs = 100
        eta = 1e-1

        for e in range(nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.sequence(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                acc_loss = acc_loss + loss.item()

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
            print(e, acc_loss)
        pass

    def predict(self , test_input ) -> torch.Tensor :
    
        test_input = test_input.to(self.device)
        output = self.autoenc(test_input)
        output = output.cpu()
        
        return output
