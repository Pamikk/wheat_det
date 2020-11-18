import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .backbone import ResNet,conv1x1,conv3x3,Darknet,Darknet_GN 
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
def NetAPI(cfg,net,init=False):
    networks = {'yolo':YOLO,'yolo_spp':YOLO_SPP,'yolo_spp_gn':YOLO_SPP_GN}
    network = networks[net](cfg)
    if init:
        network.initialization()
    return network

class NonResidual(nn.Module):
    multiple=2
    def __init__(self,in_channels,channels,stride=1):
        super(NonResidual,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels,channels*NonResidual.multiple)
        self.bn2 = nn.BatchNorm2d(channels*NonResidual.multiple)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        return y

class YOLO(nn.Module):
    def __init__(self,cfg):
        super(YOLO,self).__init__()
        self.encoders = Darknet(os.path.join(cfg.pre_trained_path,'yolov3.weights'))
        self.out_channels = self.encoders.out_channels.copy()
        self.in_channel = self.out_channels.pop(0)
        self.relu = nn.LeakyReLU(0.1)
        decoders = []
        channels = [512,256,128]
        for i,ind in enumerate(cfg.anchor_divide):
            decoder = self.make_prediction(len(ind)*(cfg.cls_num+5),NonResidual,channels[i],upsample=i!=0)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
    def initialization(self):
        for m in self.modules():
            init_weights(m)
        self.encoders.load_dark_net()
    def make_prediction(self,out_channel,block,channel,upsample=True):
        if upsample:
            upsample = nn.Sequential(conv1x1(self.in_channel,channel),nn.BatchNorm2d(channel),
                                           self.relu,nn.Upsample(scale_factor=2,mode='bilinear'))
            cat_channel = self.out_channels.pop(0)
            self.in_channel = channel + cat_channel
        else:
            upsample = nn.Identity()
        decoders=[block(self.in_channel,channel),block(channel*block.multiple,channel)]
        decoders.append(nn.Sequential(conv1x1(channel*block.multiple,channel),nn.BatchNorm2d(channel),self.relu))        
        pred = nn.Sequential(conv3x3(channel,channel*block.multiple),nn.BatchNorm2d(channel*block.multiple),self.relu,
                conv1x1(channel*block.multiple,out_channel,bias=True))
        self.in_channel = channel
        return nn.ModuleList([upsample,nn.Sequential(*decoders),pred])
    def forward(self,x):
        feats = self.encoders(x)
        #channels:[1024,512,256,128,64]
        #spatial :[8,16,32,64,128] suppose inp is 256
        outs = list(range(len(self.decoders)))
        x = feats.pop(0)
        y = []
        for i,decoders in enumerate(self.decoders):
            up,decoder,pred = decoders
            x = torch.cat([up(x)]+y,dim=1)
            x = decoder(x)
            out = pred(x)
            outs[i] = out
            y = [feats.pop(0)]
        return outs
class YOLO_SPP(YOLO):
    def __init__(self,cfg):
        super(YOLO_SPP,self).__init__(cfg)
        self.encoders = Darknet(os.path.join(cfg.pre_trained_path,'yolov3-spp.weights'))
        self.out_channels = self.encoders.out_channels.copy()
        self.in_channel = self.out_channels.pop(0)
        self.relu = nn.LeakyReLU(0.1)
        decoders = []
        channels = [512,256,128]
        channel = channels[0]
        self.conv1 = nn.Sequential(NonResidual(self.in_channel,channel),
                                   conv1x1(channel*NonResidual.multiple,channel),nn.BatchNorm2d(channel),self.relu)
        pool_size = [1,5,9,13]
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=ks,stride=1,padding=(ks-1) // 2) for ks in pool_size])
        self.in_channel = channel * 4
        for i,ind in enumerate(cfg.anchor_divide):
            if i==0:
                decoder = self.make_prediction_SPP(len(ind)*(cfg.cls_num+5),NonResidual,channels[i])
            else:
                decoder = self.make_prediction(len(ind)*(cfg.cls_num+5),NonResidual,channels[i])
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
    def make_prediction_SPP(self,out_channel,block,channel):
        upsample = nn.Identity()
        decoders=[block(self.in_channel,channel)]
        decoders.append(nn.Sequential(conv1x1(channel*block.multiple,channel),nn.BatchNorm2d(channel),self.relu))        
        pred = nn.Sequential(conv3x3(channel,channel*block.multiple),nn.BatchNorm2d(channel*block.multiple),self.relu,
                conv1x1(channel*block.multiple,out_channel,bias=True))
        self.in_channel = channel
        return nn.ModuleList([upsample,nn.Sequential(*decoders),pred])
    def forward(self,x):
        feats = self.encoders(x)
        #channels:[1024,512,256,128,64]
        #spatial :[8,16,32,64,128] suppose inp is 256
        outs = []
        x = feats.pop(0)
        x = self.conv1(x)
        x = torch.cat([maxpool(x) for maxpool in self.pools],dim=1)
        y = []
        for decoders in self.decoders:
            up,decoder,pred = decoders
            x = torch.cat([up(x)]+y,dim=1)
            x = decoder(x)
            out = pred(x)
            outs.append(out)
            y = [feats.pop(0)]
        return outs
class YOLO_SPP_GN(YOLO_SPP):
    def __init__(self,cfg):
        super(YOLO_SPP_GN,self).__init__(cfg)
        self.encoders = Darknet_GN('')
    def make_prediction_SPP(self,out_channel,block,channel):
        upsample = nn.Identity()
        decoders=[block(self.in_channel,channel)]
        decoders.append(nn.Sequential(conv1x1(channel*block.multiple,channel),nn.GroupNorm(channel//4,channel),self.relu))        
        pred = nn.Sequential(conv3x3(channel,channel*block.multiple),nn.BatchNorm2d(channel*block.multiple//4,channel*block.multiple),self.relu,
                conv1x1(channel*block.multiple,out_channel,bias=True))
        self.in_channel = channel
        return nn.ModuleList([upsample,nn.Sequential(*decoders),pred])    




    