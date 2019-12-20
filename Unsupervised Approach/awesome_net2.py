import torch.nn as nn
import torch

class AWESOME_NET2(nn.Module):
    def __init__(self):
        super(AWESOME_NET2,self).__init__()

        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)


        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)


        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)


        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)


        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)


        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)


        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)


        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)


        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)


        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,512,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)


        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(512,512,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)


        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(512,512,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)


        self.maxPool4 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)


        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(512,512,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)



        # decoder
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)


        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        self.relu16 = nn.ReLU(inplace=True)


        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(512,512,3,1,0)
        self.relu17 = nn.ReLU(inplace=True)


        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        self.relu18 = nn.ReLU(inplace=True)


        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        self.relu19 = nn.ReLU(inplace=True)


        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)


        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        self.relu20 = nn.ReLU(inplace=True)


        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(256,256,3,1,0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)


        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        self.relu24 = nn.ReLU(inplace=True)

        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(128,64,3,1,0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        self.relu26 = nn.ReLU(inplace=True)

        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        out = self.reflecPad3(self.relu2(self.conv2(self.reflecPad1(self.conv1(x)))))
        out = self.relu3(self.conv3(out))
        out,pool_idx = self.maxPool(out)
        out = self.reflecPad5(self.relu4(self.conv4(self.reflecPad4(out))))
        out = self.relu5(self.conv5(out))
        out,pool_idx2 = self.maxPool2(out)
        out = self.conv7(self.reflecPad7(self.relu6(self.conv6(self.reflecPad6(out)))))
        out = self.reflecPad9(self.relu8(self.conv8(self.reflecPad8(self.relu7(out)))))
        out = self.relu9(self.conv9(out))
        out,pool_idx3 = self.maxPool3(out)
        out = self.conv11(self.reflecPad11(self.relu10(self.conv10(self.reflecPad10(out)))))
        out = self.relu12(self.conv12(self.reflecPad12(self.relu11(out))))
        out = self.relu13(self.conv13(self.reflecPad13(out)))
        out,pool_idx4 = self.maxPool4(out)
        out = self.relu14(self.conv14(self.reflecPad14(out)))
        
        #decoder
        out = self.reflecPad16(self.unpool(self.relu15(self.conv15(self.reflecPad15(out)))))
        out = self.reflecPad18(self.relu17(self.conv17(self.reflecPad17(self.relu16(self.conv16(out))))))
        out = self.relu19(self.conv19(self.reflecPad19(self.relu18(self.conv18(out)))))
        out = self.reflecPad21(self.relu20(self.conv20(self.reflecPad20(self.unpool2(out))))) 
        out = self.relu22(self.conv22(self.reflecPad22(self.relu21(self.conv21(out)))))
        out = self.reflecPad24(self.unpool3(self.relu23(self.conv23(self.reflecPad23(out)))))
        out = self.reflecPad25(self.relu24(self.conv24(out)))
        out = self.reflecPad26(self.unpool4(self.relu25(self.conv25(out))))
        out = self.conv27(self.reflecPad27(self.relu26(self.conv26(out))))

        return out


class decoder5(nn.Module):
    def __init__(self,d):
        super(decoder5,self).__init__()

        

    def forward(self,x):
        # decoder


        return out
