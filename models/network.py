import torch.nn as nn
import torch.functional as F

from .resnet import ResNet
from .mynets import LocNet,RefineNet 
__all__=['Network','Networkv2']
class Bottleneck(nn.Module):
    multiple = 2
    def __init__(self,in_channels,channels,stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride=stride)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels,channels)
        if in_channels != channels*Bottleneck.multiple or stride!=1:
            self.downsample =  nn.Sequential(conv1x1(in_channels,channels*Bottleneck.multiple,stride),
                                            nn.BatchNorm2d(channels*Bottleneck.multiple))
        else:
            self.downsample = nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels,Bottleneck.multiple*channels)
        self.bn3 = nn.BatchNorm2d(Bottleneck.multiple*channels)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        x = self.downsample(x)
        y += x
        y = self.relu(y)

        return y
class Network(nn.Module):
    def __init__(self,res,int_shape,cls_num,pretrained=True):
        super(Network,self).__init__()
        self.feat = ResNet(res)
        if pretrained:
            self.feat.load_from_url()
        self.loc_net = LocNet(self.feat.channels,int_shape)
        self.loc_net.initialization()
        self.refine_net = RefineNet(self.loc_net.channel,cls_num)
        self.refine_net.initialization()
    def forward(self,x):
        feats = self.feat(x)
        locs,loc_feats = self.loc_net(feats)
        results = self.refine_net(loc_feats)
        return locs,results
class Networkv2(nn.Module):
    def __init__(self,res,int_shape,cls_num,pretrained=True):
        super(Networkv2,self).__init__()
        self.feat = ResNet(res)
        if pretrained:
            self.feat.load_from_url()
        self.pred = self.make_prediction(self.feat.channels[0],cls_num,Bottleneck)
        self.channel = 256
    def make_prediction(self,in_channel,out_channel,block):
        blocks = [block(in_channel,self.channel)]
        blocks.append(nn.Conv2d(self.channel*block.multiple,out_channel,kernel_size=3,padding=1,stride=1,bias=False))
        blocks.append(nn.BatchNorm2d(out_channel))
        return nn.Sequential(*blocks)

    def forward(self,x):
        feats = self.feat(x)
        feats = feats[0]
        return self.pred(feats)
    