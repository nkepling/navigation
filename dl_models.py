import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import random
import torch.nn.functional as F


class ValueIterationModel(torch.nn.Module):
    """Encoder-Decoder model for Value Iteration
    """
    def __init__(self):
        super(ValueIterationModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1) 
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x


class ValueIterationModelWithPooling(torch.nn.Module):
    """Encoder-Decoder model for Value Iteration
    """
    def __init__(self):
        super(ValueIterationModelWithPooling, self).__init__()
        # Encoder (Downsampling)
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downsample to (64, 5, 5)

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        # Use pooling only once to avoid excessive downsampling
        # self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Avoid this second downsample
        

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = torch.nn.BatchNorm2d(1)

        self.final_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
        
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        # No pooling here to avoid reducing the spatial dimensions too much
        
        # Decoder
        x = F.relu(self.bn3(self.deconv1(x)))
        x = F.relu(self.bn4(self.deconv2(x)))

        # Final convolution to adjust to (1, 10, 10)
        x = self.final_conv(x)

        # Sigmoid activation to keep output between 0 and 1
        x = torch.sigmoid(x)
        return x

class DeeperValueIterationModel(torch.nn.Module):
    def __init__(self):
        super(DeeperValueIterationModel, self).__init__()
        
        # Encoder (Downsampling) Layers
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        
        # Decoder (Upsampling) Layers
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        
        # Sigmoid activation to keep output between 0 and 1
        x = torch.sigmoid(x)
        return x
    

###### UNet Model ######
# I refrence this implementation from the following link: https://github.com/milesial/Pytorch-UNet/tree/master


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)


    

class Up(nn.Module):
    def __init__(self, in_channels,out_channels) -> None:
        super(Up,self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3)

    def forward(self, x):
        return self.up(x)
    
class Down(nn.Module):
    def __init__(self) -> None:
        super(Down,self).__init__()
        self.down = nn.MaxPool2d(kernel_size=3,stride=1)

    def forward(self, x):
        return self.down(x)





class UNet(torch.nn.Module):
    """UNet model for Value Function Approximation
    """
    def __init__(self) -> None:
        super(UNet,self).__init__()

        # down sample and up sample layers
        self.down = Down()
        self.up1 = Up(512,256)
        self.up2 = Up(256,128)
        self.up3 = Up(128,64)
        self.up4 = Up(64,32)

        # Double Convolution Layers

        self.layer_1 = DoubleConv(2,32)
        self.layer_2 = DoubleConv(32,64)
        self.layer_3 = DoubleConv(64,128)
        self.layer_4 = DoubleConv(128,256)
        self.layer_5 = DoubleConv(256,512)

        self.layer_6 = DoubleConv(512,256)
        self.layer_7 = DoubleConv(256,128)
        self.layer_8 = DoubleConv(128,64)
        self.layer_9 = DoubleConv(64,32)

        # Output Layer
        self.out = nn.Conv2d(32,1,kernel_size=1)
    
    def forward(self, x):

        ### Left Side of the UNet
        layer_1_out = self.layer_1(x) # 2, 10, 10 -> 32, 10, 10

        layer_2_out = self.layer_2(self.down(layer_1_out)) # 32, 10, 10 -> 64, 8, 8

        layer_3_out = self.layer_3(self.down(layer_2_out)) # 64, 8, 8 -> 128, 6, 6

        layer_4_out = self.layer_4(self.down(layer_3_out)) # 128, 6, 6 -> 256, 4, 4

        layer_5_out = self.layer_5(self.down(layer_4_out)) # 256, 4, 4 -> 512, 2, 2

        ### Right Side of the UNet

        up_1_out = self.up1(layer_5_out) # 512, 2, 2 -> 256, 4, 4
        layer_6_out = self.layer_6(torch.cat([up_1_out,layer_4_out],1)) # 512, 4, 4 -> 256, 4, 4

        up_2_out = self.up2(layer_6_out) # 256, 4, 4 -> 128, 6, 6
        layer_7_out = self.layer_7(torch.cat([up_2_out,layer_3_out],1))

        up_3_out = self.up3(layer_7_out) # 128, 6, 6 -> 64, 8, 8
        layer_8_out = self.layer_8(torch.cat([up_3_out,layer_2_out],1))

        up_4_out = self.up4(layer_8_out) # 64, 8, 8 -> 32, 10, 10
        layer_9_out = self.layer_9(torch.cat([up_4_out,layer_1_out],1))

        out = self.out(layer_9_out) # 32, 10, 10 -> 1, 10, 10

        return out
    

class UNetSmall(torch.nn.Module):
    """Smaller UNet model for Value Function Approximation
    """
    def __init__(self) -> None:
        super(UNetSmall, self).__init__()

        # down sample and up sample layers
        self.down = Down()
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)

        # Double Convolution Layers

        self.layer_1 = DoubleConv(2, 16)
        self.layer_2 = DoubleConv(16, 32)
        self.layer_3 = DoubleConv(32, 64)
        self.layer_4 = DoubleConv(64, 128)
        self.layer_5 = DoubleConv(128, 256)

        self.layer_6 = DoubleConv(256, 128)
        self.layer_7 = DoubleConv(128, 64)
        self.layer_8 = DoubleConv(64, 32)
        self.layer_9 = DoubleConv(32, 16)

        # Output Layer
        self.out = nn.Conv2d(16, 1, kernel_size=1)
    
    def forward(self, x):

        ### Left Side of the UNet
        layer_1_out = self.layer_1(x)  # 2, 10, 10 -> 16, 10, 10

        layer_2_out = self.layer_2(self.down(layer_1_out))  # 16, 10, 10 -> 32, 8, 8

        layer_3_out = self.layer_3(self.down(layer_2_out))  # 32, 8, 8 -> 64, 6, 6

        layer_4_out = self.layer_4(self.down(layer_3_out))  # 64, 6, 6 -> 128, 4, 4

        layer_5_out = self.layer_5(self.down(layer_4_out))  # 128, 4, 4 -> 256, 2, 2

        ### Right Side of the UNet

        up_1_out = self.up1(layer_5_out)  # 256, 2, 2 -> 128, 4, 4

        layer_6_out = self.layer_6(torch.cat([up_1_out, layer_4_out], 1))  # 256, 4, 4 -> 128, 4, 4

        up_2_out = self.up2(layer_6_out)  # 128, 4, 4 -> 64, 6, 6
        layer_7_out = self.layer_7(torch.cat([up_2_out, layer_3_out], 1))

        up_3_out = self.up3(layer_7_out)  # 64, 6, 6 -> 32, 8, 8
        layer_8_out = self.layer_8(torch.cat([up_3_out, layer_2_out], 1))

        up_4_out = self.up4(layer_8_out)  # 32, 8, 8 -> 16, 10, 10
        layer_9_out = self.layer_9(torch.cat([up_4_out, layer_1_out], 1))

        out = self.out(layer_9_out)  # 16, 10, 10 -> 1, 10, 10

        return out




if __name__ == "__main__":
    model = UNetSmall()
    model = DeeperValueIterationModel()
    x = torch.randn(1,2,10,10)
    y = model(x)
    print(y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")




    


