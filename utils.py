import matplotlib.pyplot as plt 
import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os 
import json
class Logger(object):
    def __init__(self,log_dir):
        self.writer = SummaryWriter(log_dir)
        self.files = {'val':open(os.path.join(log_dir,'val.txt'),'a+'),'train':open(os.path.join(log_dir,'train.txt'),'a+')}
    def write_line2file(self,mode,string):
        self.files[mode].write(string+'\n')
        self.files[mode].flush()
    def write_loss(self,epoch,losses,lr):
        tmp = str(epoch)+'\t'+str(lr)+'\t'
        print('Epoch',':',epoch,'-',lr)
        self.writer.add_scalar('lr',lr,step=epoch)
        for k in losses:            
            self.writer.add_scalar(k,losses[k],step=epoch)
            tmp+= str(round(losses[k],3))+'\t'
            print(k,':',losses[k])
        self.writer.flush()
        self.write_line2file('train',tmp)
    def write_metrics(self,epoch,metrics,save=[]):
        tmp =str(epoch)+'\t'
        print("validation epoch:",epoch)
        for k in metrics:
            if k in save:
                tmp +=str(metrics[k])+'\t'            
            self.writer.add_scalar('val'+k,metrics[k],step=epoch)
            print(k,':',metrics[k])
        self.writer.flush()
        self.write_line2file('val',tmp)

def iou_wo_center(w1,h1,w2,h2):
    #only for torch, return a vector nx1
    inter = torch.min(w1,w2)*torch.min(h1,h2)
    union = w1*h1 + w2*h2 - inter
    return inter/union
def iou_wt_center(bbox1,bbox2):
    #only for torch, return a vector nx1
    bbox1 = bbox1.view(-1,4)
    bbox2 = bbox2.view(-1,4)
    
    #tranfer xc,yc,w,h to xmin ymin xmax ymax
    xmin1 = bbox1[:,0] - bbox1[:,2]/2
    xmin2 = bbox2[:,0] - bbox2[:,2]/2
    ymin1 = bbox1[:,1] - bbox1[:,3]/2
    ymin2 = bbox2[:,1] - bbox2[:,3]/2
    xmax1 = bbox1[:,0] + bbox1[:,2]/2
    xmax2 = bbox2[:,0] + bbox2[:,2]/2
    ymax1 = bbox1[:,1] + bbox1[:,3]/2
    ymax2 = bbox2[:,1] + bbox2[:,3]/2

    inter_xmin = torch.max(xmin1,xmin2)
    inter_xmax = torch.min(xmax1,xmax2)
    inter_ymin = torch.max(ymin1,ymin2)
    inter_ymax = torch.min(ymax1,ymax2)

    inter = abs((inter_ymax-inter_ymin)*(inter_xmax-inter_xmin))+1
    area1 = (ymax1-ymin1)*(xmax1-xmin1)
    area2 = (ymax2-ymin2)*(xmax2-xmin2)
    union = abs(area1+area2 - inter) + 1
    return inter/union
def iou_wt_center_np(bbox1,bbox2):
    #in numpy,only for evaluation,return a matrix m x n
    bbox1 = bbox1.reshape(-1,4)
    bbox2 = bbox2.reshape(-1,4)

    
    #tranfer xc,yc,w,h to xmin ymin xmax ymax
    xmin1 = bbox1[:,0] - bbox1[:,2]/2
    xmin2 = bbox2[:,0] - bbox2[:,2]/2
    ymin1 = bbox1[:,1] - bbox1[:,3]/2
    ymin2 = bbox2[:,1] - bbox2[:,3]/2
    xmax1 = bbox1[:,0] + bbox1[:,2]/2
    xmax2 = bbox2[:,0] + bbox2[:,2]/2
    ymax1 = bbox1[:,1] + bbox1[:,3]/2
    ymax2 = bbox2[:,1] + bbox2[:,3]/2

    #trigger broadcasting
    inter_xmin = np.maximum(xmin1.reshape(-1,1),xmin2.reshape(1,-1))
    inter_xmax = np.minimum(xmax1.reshape(-1,1),xmax2.reshape(1,-1))
    inter_ymin = np.maximum(ymin1.reshape(-1,1),ymin2.reshape(1,-1))
    inter_ymax = np.minimum(ymax1.reshape(-1,1),ymax2.reshape(1,-1))

    inter = abs((inter_ymax-inter_ymin)*(inter_xmax-inter_xmin))+1
    area1 = (ymax1-ymin1)*(xmax1-xmin1)
    area2 = (ymax2-ymin2)*(xmax2-xmin2)
    union = abs(area1+area2 - inter) +1
    return inter/union

def cal_metrics(pd,gt,threshold=0.5):
    pd = pd.cpu().numpy()#n
    gt = gt.cpu().numpy()#m
    pd_bboxes = pd[:,:4]
    m = len(gt)
    n = len(pd_bboxes)
    if n>0 and m>0:
        ious = iou_wt_center_np(pd_bboxes,gt) #nxm
        scores = ious.max(axis=1) 
        fp = scores <= 0.5

        #only keep trues
        ious = ious[1-fp,:]
        fp = fp.sum() # transfer to scalar

        select_ids = ious.argmax(axis=1)
        #discard fps hit gt boxes has been hitted by bboxes with higher conf
        tp = np.unique(select_ids)
        fp += len(select_ids)-len(tp)
        
        # groud truth with no associated predicted object
        fn = m-len(tp)
        p = tp/n
        r = tp/m
        ap = tp/(fp+fn)
        return p,r,ap
    elif m>0 or n >0 :
        return 0,0,0
    else:
        return 1,1,1


    
def non_maximum_supression(preds,loc_score,conf_threshold=0.5,nms_threshold = 0.4):
    preds = preds[preds[:,-1] >= conf_threshold]
    if len(preds) == 0:
        return preds
    idx = np.argsort(-1*preds[:,-1].cpu().numpy())
    pds = preds[idx]
    keep = []
    while len(pds)>0:
        ious = iou_wt_center(pds[0,:4],pds[:,:4])
        assert ious[0]>=0.7
        mask = ious>nms_threshold
        weights = pds[mask,-1].view(-1,1)
        #merge predictions
        new = (weights * pds[mask,:4]).sum(dim=0)/weights.sum()
        keep.append(new)
        pds = pds[~mask]
    return torch.stack(keep)



def visualization():
    pass
def test_pds():
    pass












