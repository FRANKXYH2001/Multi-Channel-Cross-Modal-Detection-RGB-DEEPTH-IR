import torch
from torch import nn
from torchvision import models
import torch.fft

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class MultiStreamResNet(nn.Module):
    """ 
    This class represents a Two-stream densenet architecture with seperate architecture heads 
    for the branches and the joint model
    """

    def __init__(self, pretrained=True):

        """ Init function
        Parameters
        ----------
        pretrained: bool
            If true will use weights from ImageNet else retrain from scratch
        """

        # Initialize the [res block-se blocks] and additional layers
        super(MultiStreamResNet, self).__init__()
        self.in_channels = 64
        num_classes=1
        
        # First stage of each modality
        self.rgb_res1 = self._make_layer(ResidualBlock, 64, 3)
        self.depth_res1 = self._make_layer(ResidualBlock, 64, 3)
        
        # Second stage of each modality
        self.rgb_res2 = self._make_layer(ResidualBlock, 128, 4, stride=2)
        self.depth_res2 = self._make_layer(ResidualBlock, 128, 4, stride=2)
        
        # Third stage of each modality
        self.rgb_res3 = self._make_layer(ResidualBlock, 256, 6, stride=2)
        self.depth_res3 = self._make_layer(ResidualBlock, 256, 6, stride=2)
        
        # Global Average Pooling and linear layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Project to 384-dimensional space
        self.fc_rgb = nn.Linear(256, 384)
        self.fc_depth = nn.Linear(256, 384)
        
        # Joint classification layer
        self.fc_joint = nn.Linear(384*2, num_classes)
        
        # Separate heads
        self.head_rgb = nn.Linear(384, num_classes)
        self.head_depth = nn.Linear(384, num_classes)


    def forward(self, rgb_img, fourier_img):
        """ Propagate data through the network architecture. Expects 112*112 input RGB images and Deepth input as tensors

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 

        Returns
        -------
        output, output_rgb, output_depth: :py:class:`torch.Tensor`
        """

        # RGB Stream
        rgb = self.rgb_res1(rgb_img)
        rgb = self.rgb_res2(rgb)
        rgb = self.rgb_res3(rgb)
        rgb = self.avg_pool(rgb).view(rgb.size(0), -1)
        rgb = self.fc_rgb(rgb)
        rgb = torch.sigmoid(rgb)
        
        # Depth Stream
        depth = self.depth_res1(fourier_img)
        depth = self.depth_res2(depth)
        depth = self.depth_res3(depth)
        depth = self.avg_pool(depth).view(depth.size(0), -1)
        depth = self.fc_depth(depth)
        depth = torch.sigmoid(depth)
        
        # Concatenate and classify
        joint_embedding = torch.cat((rgb, depth), dim=1)
        output = self.fc_joint(joint_embedding)
        output = torch.sigmoid(output)
        
        # Separate heads
        output_rgb = self.head_rgb(rgb)
        output_rgb = torch.sigmoid(output_rgb)
        
        output_depth = self.head_depth(depth)
        output_depth = torch.sigmoid(output_depth)
        
        return output, output_rgb, output_depth
    
