import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from .resnet import conv3x3,conv1x1
class Bottleneck(nn.Module):
    multiple = 2
    def __init__(self,in_channels,channels,stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride=stride)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
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
class LocNet(nn.Module):
    def __init__(self,channels,out_shape):
        super(LocNet,self).__init__()
        layers= []
        predicts = []
        ups = []
        self.channel = 256 #generalize to the same channel before prediction
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        for i in range(len(channels)):
            layers.append(self.stack_blocks(channels[i]))
            if i==0:
                predicts.append(self.make_prediction(self.channel))
                ups.append(nn.Sequential(nn.Conv2d(self.channel,self.channel,kernel_size=1,bias=False),nn.BatchNorm2d(self.channel),self.relu))
            else:
                predicts.append(self.make_prediction(self.channel*2))
                ups.append(nn.Sequential(nn.Conv2d(self.channel*2,self.channel,kernel_size=1,bias=False),nn.BatchNorm2d(self.channel),self.relu))
        self.layers = nn.ModuleList(layers)
        self.predict = nn.ModuleList(predicts)
        self.ups = nn.ModuleList(ups)
        self.out_shape = out_shape

    def stack_blocks(self,in_channel):
        conv = nn.Conv2d(in_channel,self.channel,kernel_size=1,stride=1,bias=False)
        bn = nn.BatchNorm2d(self.channel)
        return nn.Sequential(conv,bn,self.relu)
    def make_prediction(self,in_channel):
        conv1 = nn.Conv2d(in_channel,self.channel,kernel_size=1,stride=1,bias=False)
        bn1 = nn.BatchNorm2d(self.channel)
        conv2 = nn.Conv2d(self.channel,1,kernel_size=3,stride=1,padding=1,bias=False)
        bn2 = nn.BatchNorm2d(1)
        return nn.Sequential(conv1,bn1,self.relu,conv2,bn2)
    def forward(self,x):
        results = []
        feats = []
        for i in range(len(x)):
            feat = self.layers[i](x[i])
            if i==0:
                out = self.predict[i](feat)
            else:
                feat = torch.cat((feat,up),dim=1)
                out = self.predict[i](feat)
            up = self.ups[i](feat)
            feats.append(up)
            results.append(F.interpolate(out,size=self.out_shape,mode='bilinear', align_corners=True))
            up = F.interpolate(up,scale_factor=2,mode='bilinear', align_corners=True)
        return results,feats

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class RefineNet(nn.Module):
    def __init__(self,in_channel,class_num):
        super(RefineNet,self).__init__()
        self.channel = in_channel
        layers =[self.stack_blocks(1,self.channel,Bottleneck,stride=1)]
        self.num = 4
        for i in range(1,self.num):
            layers.append(self.stack_blocks(i,self.channel,Bottleneck,stride=2))
        self.layers = nn.ModuleList(layers)
        self.predict = self.make_final_prediction(self.channel*4,class_num,Bottleneck)
    def stack_blocks(self,depth,channels,block,stride=1):
        blocks = [block(channels,channels//block.multiple,stride) for _ in range(depth)]
        return nn.Sequential(*blocks)
    def make_final_prediction(self,in_channel,cls_num,block):
        blocks = [block(in_channel,self.channel//block.multiple)]
        blocks.append(nn.Conv2d(self.channel,cls_num,kernel_size=3,padding=1,stride=1,bias=False))
        blocks.append(nn.BatchNorm2d(cls_num))
        return nn.Sequential(*blocks)
    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        feats=[]
        for i in range(self.num):
            feats.append(self.layers[i](x[i]))
        feats = torch.cat(feats,dim=1)
        res = self.predict(feats)
        return res




    
    



    

