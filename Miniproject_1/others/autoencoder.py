from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
            def __init__(self):
                super().__init__()
               
                """
                self.conv1 = nn.Conv2d(3,32,kernel_size=2,stride=1)
                self.conv2 = nn.Conv2d(32,32,kernel_size=2,stride=1)
                self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1)
                self.convT1 = nn.ConvTranspose2d(32,32,kernel_size=3,stride=1)
                self.convT2 = nn.ConvTranspose2d(32,32,kernel_size=3,stride=1)
                self.convT3 = nn.ConvTranspose2d(32,3,kernel_size=3,stride=1)
                """
                
                
                #self.conv0 = nn.Conv2d(3,3,kernel_size=1)
                self.conv1 = nn.Conv2d(3,10,kernel_size=5, padding = (5-1)//2)
                self.bn1 = nn.BatchNorm2d(10)
                self.skip_connections = True
                self.conv2 = nn.Conv2d(10,10,kernel_size=5, padding = (5-1)//2)
                self.convT1 = nn.ConvTranspose2d(10,3,kernel_size=5, padding = (5-1)//2)
                self.batch_norm = True
                
                
            def forward(self, x):
                
                # skip conection 
                # res net
                
                """
                #print("start", x.shape)
                x = F.leaky_relu(self.conv1(x))
                #print("first_check", x.shape)
                x = F.max_pool2d(x, 3)
                #print("max_pool_1", x.shape)
                x = F.leaky_relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                #print("max_pool_2", x.shape)
                x = F.leaky_relu(self.conv3(x))
                #print("end_conv", x.shape)
                x = F.leaky_relu(self.convT1(x))
                x = F.upsample(x, size=None, scale_factor=2)
                #print("Up_sample_1", x.shape)
                x = F.leaky_relu(self.convT2(x))
                x = F.upsample(x, size=None, scale_factor=3)
                #print("Up_sample_2", x.shape)
                x = F.sigmoid(self.convT3(x))
                #print("end", x.shape)
                #print("conv_T_2", x.shape)
                
                
                print("1", x.shape)
                x = F.relu(self.conv1(x))
                print("conv_1", x.shape)
                x = F.max_pool2d(x, 3)
                print("pool_1", x.shape)
                x = F.upsample(x, size=None, scale_factor=2)
                print("upsample_1", x.shape)
                """
                

                
                y1 = F.leaky_relu(self.conv1(x))
                y = F.leaky_relu(self.conv2(y1))
                if self.batch_norm: y = self.bn1(y)
                if self.skip_connections: y = y + y1
                y = F.sigmoid(self.convT1(y))
                
                return y