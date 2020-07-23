import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os

from loss_funcs import *
from utils import Logger
save=['acc','mAP']
class Trainer:
    def __init__(self,cfg,datasets,net,epoch):
        self.cfg = cfg
        self.trainset = datasets['train']
        self.valset = datasets['val']
        self.net = net
        name = cfg.exp_name
        self.name = name
        self.checkpoints = os.path.join(cfg.checkpoint,name)
        self.device = cfg.device
        self.net = self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.lr_sheudler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max', factor=cfg.lr_factor, threshold=0.0001,patience=15,min_lr=cfg.min_lr)
        if not(os.path.exists(self.checkpoints)):
            os.mkdir(self.checkpoints)
        start,total = epoch
        self.start = start
        if start!=0:
            self.load_epoch(str(start))
        if start==-1:
            self.load_epoch('last')
        if start==-2:
            self.load_epoch('best')
        self.total = total
        self.loss = MyLoss()
        log_dir = os.path.join(self.checkpoints,'logs')
        if not(os.path.exists(log_dir)):
            os.mkdir(log_dir)
        self.logger = Logger(log_dir)
        torch.cuda.empty_cache()
        self.save_every_k_epoch = cfg.save_every_k_epoch #-1 for not save and validate

    def load_epoch(self,idx):
        model_path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        if os.path.exists(model_path):
            print('load:'+model_path)
            info = torch.load(model_path)
            self.net.load_state_dict(info['net_dict'])
    def train(self):
        print("strat train:",self.name)
        print("start from epoch:",self.start)
        print("=============================")

        epoch = self.start

        while epoch < self.total:
            for i,data in tqdm(enumerate(self.trainset)):
                inputs,labels,heatmaps,info,size = data
                locs,outs = self.net(inputs.to(self.device).float())
                labels = labels.to(self.device).float()
                self.loss(outs,locs,labels,heatmaps,size)




