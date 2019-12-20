import torch.nn as nn
import torch

# Encoder module
class AWESOME_NET3(nn.Module):
    def __init__(self):
        super(AWESOME_NET3,self).__init__()
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
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        ## Convolution Layer extra 1
        self.reflecPad_e1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_e1 = nn.Conv2d(128,64,3,1,0)
        self.relu_e1 = nn.ReLU(inplace=True)
        ## Convolution Layer extra 2
        self.reflecPad_e2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_e2 = nn.Conv2d(64,32,3,1,0)
        self.relu_e2 = nn.ReLU(inplace=True)


        # Decoder

        ## Convolution Layer extra 3
        self.reflecPad_e3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_e3 = nn.Conv2d(32,64,3,1,0)
        self.relu_e3 = nn.ReLU(inplace=True)

        ## Convolution Layer extra 4
        self.reflecPad_e4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_e4 = nn.Conv2d(64,128,3,1,0)
        self.relu_e4 = nn.ReLU(inplace=True)


        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,64,3,1,0)
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
        out = self.relu4(self.conv4(self.reflecPad4(out)))

        out,pool_idx = self.maxPool(out) 
        # extra layer encoder
        encoder_out = (self.reflecPad_e2(self.relu_e2(self.conv_e2(self.relu_e1(self.conv_e1(self.reflecPad_e1(out))))))) 
        encoder_out,pool_idx = self.maxPool(encoder_out)
        # decoder
                
        # extra layer encoder
        out = (self.reflecPad_e4(self.relu_e4(self.conv_e4(self.relu_e3(self.conv_e3(self.reflecPad_e3(encoder_out)))))))
        out = self.unpool(out) 
        out = self.reflecPad5(out) 
        out = self.relu5(self.conv5(out))
        out = self.unpool(out) 
        out = self.relu6(self.conv6(self.reflecPad6(out)))
        out = self.unpool(out) 
        out = self.conv7(self.reflecPad7(out))
        
        return encoder_out, out
