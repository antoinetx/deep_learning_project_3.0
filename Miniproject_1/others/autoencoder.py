from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
            def __init__(self):
                super().__init__()
               
                self.conv1 = nn.Conv2d(3,32,kernel_size=2,stride=1)
                self.conv2 = nn.Conv2d(32,32,kernel_size=2,stride=1)
                self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1)
                self.convT1 = nn.ConvTranspose2d(32,32,kernel_size=3,stride=1)
                self.convT2 = nn.ConvTranspose2d(32,32,kernel_size=3,stride=1)
                self.convT3 = nn.ConvTranspose2d(32,3,kernel_size=3,stride=1)

                """
                self.conv1 = nn.Conv2d(3,32,kernel_size=2,stride=2)
                self.conv3 = nn.Conv2d(32,32,kernel_size=2,stride=2)
                self.convT1 = nn.ConvTranspose2d(32,32,kernel_size=2,stride=2)
                self.convT3 = nn.ConvTranspose2d(64,3,kernel_size=2,stride=2)
                """ 
            def forward(self, x):
                
                #print("max_pool", x.shape)
                #print("1", x.shape)
                #print("2", x.shape)
                #print("conv2", x.shape)
                #print("conv_T_1", x.shape)
                #print("up_sample", x.shape)
                
                
                # skip conection 
                # res net
                
                
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
                
                """
                print("1", x.shape)
                x = F.relu(self.conv1(x))
                print("conv_1", x.shape)
                x = F.max_pool2d(x, 3)
                print("pool_1", x.shape)
                x = F.upsample(x, size=None, scale_factor=2)
                print("upsample_1", x.shape)
                """
                """
                x1 = F.leaky_relu(self.conv1(x))
                x = F.leaky_relu(self.conv3(x1))
                x = F.leaky_relu(self.convT1(x))
                x = torch.cat((x,x1),1)
                x = F.leaky_relu(self.convT3(x))
                x = F.leaky_relu(x)
                """
                return x