import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

from utils import Logger
from utils import cal_metrics as cal_metrics
from utils import non_maximum_supression as nms
tosave=['mAP']
thresholds = [0.5,0.75]
class Trainer:
    def __init__(self,cfg,datasets,net,loss,epoch):
        self.cfg = cfg
        if 'train' in datasets:
            self.trainset = datasets['train']
            self.valset = datasets['val']
        if 'trainval' in datasets:
            self.trainval = datasets['trainval']
        else:
            self.trainval = False
        if 'test' in datasets:
            self.testset = datasets['test']
        self.net = net
        name = cfg.exp_name
        self.name = name
        self.checkpoints = os.path.join(cfg.checkpoint,name)
        self.device = cfg.device
        self.net = self.net
        self.optimizer = optim.SGD(self.net.parameters(),lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
        self.lr_sheudler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=cfg.lr_factor, threshold=0.0001,patience=4,min_lr=cfg.min_lr)
        if not(os.path.exists(self.checkpoints)):
            os.mkdir(self.checkpoints)
        self.predictions = os.path.join(self.checkpoints,'pred')
        if not(os.path.exists(self.predictions)):
            os.mkdir(self.predictions)
        start,total = epoch
        self.start = start        
        self.total = total
        self.loss = loss
        log_dir = os.path.join(self.checkpoints,'logs')
        if not(os.path.exists(log_dir)):
            os.mkdir(log_dir)
        self.logger = Logger(log_dir)
        torch.cuda.empty_cache()
        self.save_every_k_epoch = cfg.save_every_k_epoch #-1 for not save and validate
        self.val_every_k_epoch = 5
        self.upadte_grad_every_k_batch = 1

        self.best_mAP = 0
        self.best_mAP_epoch = 0

        self.movingAvg = 0
        self.bestMovingAvg = 0
        self.bestMovingAvgEpoch = 1e9
        self.early_stop_epochs = 50
        self.alpha = 0.95 #for update moving Avg
        self.nms_threshold = cfg.nms_threshold
        self.conf_threshold = cfg.dc_threshold
        self.save_pred = False
        self.adjust_lr = False
        #load from epoch if required
        if start>0:
            self.load_epoch(str(start))
        if start==-1:
            self.load_epoch('best')
        if start==-2:
            self.load_epoch('bestm')#best moving
        self.net = self.net.to(self.device)
    def save_epoch(self,idx,epoch):
        saveDict = {'net':self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler':self.lr_sheudler.state_dict(),
                    'epoch':epoch,
                    'mAP':self.best_mAP,
                    'mAP_epoch':self.best_mAP_epoch,
                    'movingAvg':self.movingAvg,
                    'bestmovingAvg':self.bestMovingAvg,
                    'bestmovingAvgEpoch':self.bestMovingAvgEpoch}
        path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        torch.save(saveDict,path)                  
    def load_epoch(self,idx):
        model_path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        if os.path.exists(model_path):
            print('load:'+model_path)
            info = torch.load(model_path)
            self.net.load_state_dict(info['net'])
            if not(self.adjust_lr):
                self.optimizer.load_state_dict(info['optimizer'])#might have bugs about device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.lr_sheudler.load_state_dict(info['lr_scheduler'])
            self.start = info['epoch']+1
            self.best_mAP = info['mAP']
            self.best_mAP_epoch = info['mAP_epoch']
            self.movingAvg = info['movingAvg']
            self.bestMovingAvg = info['bestmovingAvg']
            self.bestMovingAvgEpoch = info['bestmovingAvgEpoch']
        else:
            print('no such model at:',model_path)
            exit()
    def _updateMetrics(self,mAP,epoch):
        if self.movingAvg ==0:
            self.movingAvg = mAP
        else:
            self.movingAvg = self.movingAvg * self.alpha + mAP*(1-self.alpha)
        if self.bestMovingAvg<self.movingAvg:
            self.bestMovingAvg = self.movingAvg
            self.bestMovingAvgEpoch = epoch
            self.save_epoch('bestm',epoch)
    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))

    def train_one_epoch(self):
        running_loss ={'xy':0.0,'wh':0.0,'conf':0.0,'cls':0.0,'obj':0.0,'all':0.0,'iou':0.0,'gou':0.0}
        self.net.train()
        n = len(self.trainset)
        for i,data in tqdm(enumerate(self.trainset)):
            inputs,labels = data
            outs = self.net(inputs.to(self.device).float())
            labels = labels.to(self.device).float()
            size = inputs.shape[-2:]
            display,loss = self.loss(outs,labels,size)
            del inputs,outs,labels
            for k in running_loss:
                if k in display.keys():
                    running_loss[k] += display[k]/n
            loss.backward()
            #solve gradient explosion problem caused by large learning rate or small batch size
            #nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2) 
            nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=2.0)
            if i == n-1 or (i+1) % self.upadte_grad_every_k_batch == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            del loss
        self.logMemoryUsage()
        return running_loss
    def train(self):
        print("strat train:",self.name)
        print("start from epoch:",self.start)
        print("=============================")
        self.optimizer.zero_grad()
        print(self.optimizer.param_groups[0]['lr'])
        epoch = self.start
        
        #torch.autograd.set_detect_anomaly(True)
        while epoch < self.total:
            running_loss = self.train_one_epoch()            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.write_loss(epoch,running_loss,lr)
            #step lr
            self.lr_sheudler.step(running_loss['all'])
            lr_ = self.optimizer.param_groups[0]['lr']
            if lr_ != lr:
                self.save_epoch(str(epoch),epoch)
            if (epoch+1)%self.save_every_k_epoch==0:
                self.save_epoch(str(epoch),epoch)
            if (epoch+1)%self.val_every_k_epoch==0:                
                metrics = self.validate(epoch,'val',self.save_pred)
                self.logger.write_metrics(epoch,metrics,tosave)
                mAP = metrics['mAP']
                self._updateMetrics(mAP,epoch)
                if mAP >= self.best_mAP:
                    self.best_mAP = mAP
                    self.best_mAP_epoch = epoch
                    self.save_epoch('best',epoch)
                print("best so far with:",self.best_mAP)
                if self.trainval:
                    metrics = self.validate(epoch,'train',self.save_pred)
                    self.logger.write_metrics(epoch,metrics,tosave,mode='Trainval')
            epoch +=1
                
        print("Best mAP: {:.4f} at epoch {}".format(self.best_mAP, self.best_mAP_epoch))
        self.save_epoch(str(epoch-1),epoch-1)
    def validate(self,epoch,mode,save=False):
        self.net.eval()
        res = {}
        print('start Validation Epoch:',epoch)
        if mode=='val':
            valset = self.valset
        else:
            valset = self.trainval
        with torch.no_grad():
            APs = dict.fromkeys(thresholds,0)
            precisions = dict.fromkeys(thresholds,0)
            recalls = dict.fromkeys(thresholds,0)
            mAP = 0
            count = 0
            for _,data in tqdm(enumerate(valset)):
                inputs,labels,info = data
                outs = self.net(inputs.to(self.device).float())
                size = inputs.shape[-2:]
                pds = self.loss(outs,size,infer=True)
                nB = pds.shape[0]
                for b in range(nB):
                    pred = pds[b].view(-1,self.cfg.cls_num+5)
                    gts = labels[labels[:,0]==b,1:]
                    name = info['img_id'][b]
                    size = info['size'][b]
                    pad = info['pad'][b]
                    pred[:,:4]*=size
                    pred[:,0] -= pad[1]
                    pred[:,1] -= pad[0]
                    if save:
                        pds_ = list(pred.cpu().numpy().astype(float))
                        pds_ = [list(pd) for pd in pds_]
                        res[name] = pds_
                    pred_nms = nms(pred,self.conf_threshold, self.nms_threshold)
                    count+=1
                    total = 0
                    for th in thresholds:
                        p,r,ap = cal_metrics(pred_nms,gts,threshold= th)
                        APs[th] += ap
                        precisions[th] += p
                        recalls[th] += r
                        total +=ap
                    mAP += 1.0*total/len(thresholds)
        metrics = {}
        for th in thresholds:
            metrics['AP/'+str(th)] = 1.0*APs[th]/count
            metrics['Precision/'+str(th)] = 1.0*precisions[th]/count
            metrics['Recall/'+str(th)] = 1.0*recalls[th]/count
        mAP = 1.0*mAP/count
        metrics['mAP'] = mAP
        
        
        if save:
            json.dump(res,open(os.path.join(self.predictions,'pred_epoch_'+str(epoch)+'.json'),'w'))
        
        return metrics
    def test(self):
        self.net.eval()
        res = {}
        with torch.no_grad():
            for _,data in tqdm(enumerate(self.testset)):
                inputs,info = data
                outs = self.net(inputs.to(self.device).float())
                size = inputs.shape[-2:]
                pds = self.loss(outs,size,infer=True)
                nB = pds.shape[0]
                for b in range(nB):
                    pred = pds[b].view(-1,self.cfg.cls_num+5)
                    name = info['img_id'][b]
                    size = info['size'][b]
                    pad = info['pad'][b]
                    pred[:,:4]*=size
                    pred[:,0] -= pad[1]
                    pred[:,1] -= pad[0]                    
                    pred_nms = nms(pred,self.conf_threshold, self.nms_threshold)
                    pds_ = list(pred_nms.cpu().numpy().astype(float))
                    pds_ = [list(pd) for pd in pds_]
                    res[name] = pds_
        
        json.dump(res,open(os.path.join(self.predictions,'pred_test.json'),'w'))

        


                


        




