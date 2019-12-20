import torch.nn as nn
import torch

# Encoder module
class AWESOME_NET(nn.Module):
    def __init__(self):
        super(AWESOME_NET,self).__init__()
        ## Convolution Layer 1
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        ## Convolution Layer 2 
        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        ## Convolution Layer 3
        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        ## Convolution Layer 4
        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,32,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)


        # Decoder

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(32,64,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
       
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
       
        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(64,64,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        out = self.conv3(self.reflecPad3(self.relu2(self.conv2(self.reflecPad1(self.conv1(x))))))
        pool = self.relu3(out)
        out,pool_idx = self.maxPool(pool)
        encoder_out = self.relu4(self.conv4(self.reflecPad4(out)))
        
        # decoder
        out = self.reflecPad5(encoder_out) 
        out = self.unpool(self.relu5(self.conv5(out)))
        out = self.relu6(self.conv6(self.reflecPad6(out)))
        out = self.conv7(self.reflecPad7(out))
        
        return encoder_out,out
