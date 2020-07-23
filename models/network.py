import torch.nn as nn
import torch.functional as F

from .resnet import ResNet
from .mynets import LocNet,RefineNet 
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
    