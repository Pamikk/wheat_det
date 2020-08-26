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
        self.writer.add_scalar('lr',math.log(lr),epoch)
        for k in losses:
            if losses[k]>0:            
                self.writer.add_scalar('Train/'+k,losses[k],epoch)            
                print(k,':',losses[k])
                #self.writer.flush()
        tmp+= str(round(losses['all'],5))+'\t'
        self.write_line2file('train',tmp)
    def write_metrics(self,epoch,metrics,save=[],mode='Val',log=True):
        tmp =str(epoch)+'\t'
        print("validation epoch:",epoch)
        for k in metrics:
            if k in save:
                tmp +=str(metrics[k])+'\t'
            if log:
                tag = mode+'/'+k            
                self.writer.add_scalar(tag,metrics[k],epoch)
                #self.writer.flush()
            print(k,':',metrics[k])
        
        self.write_line2file('val',tmp)

def iou_wo_center(w1,h1,w2,h2):
    #only for torch, return a vector nx1
    inter = torch.min(w1,w2)*torch.min(h1,h2)
    union = w1*h1 + w2*h2 - inter
    return inter/union
def gou(bbox1,bbox2):
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
    cover_xmin = torch.min(xmin1,xmin2)
    cover_xmax = torch.max(xmax1,xmax2)
    cover_ymin = torch.min(ymin1,ymin2)
    cover_ymax = torch.max(ymax1,ymax2)

    inter_w = inter_xmax-inter_xmin
    inter_h = inter_ymax-inter_ymin
    mask = ((inter_w>=0 )&( inter_h >=0)).to(torch.float)
    # detect not overlap
    cover = (cover_xmax-cover_xmin)*(cover_ymax-cover_ymin)
    #inter_h[inter_h<0] = 0
    inter = inter_w*inter_h*mask
    #keep iou<0 to avoid gradient diasppear
    area1 = bbox1[:,2]*bbox1[:,3]
    area2 = bbox2[:,2]*bbox2[:,3]
    union = area1+area2 - inter+1e-16
    iou = inter/union
    gou = iou-(cover-union)/cover
    return iou,gou
def cal_gous(bbox1,bbox2):
    #return mxn matrix
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

    inter_xmin = torch.max(xmin1.view(-1,1),xmin2.view(1,-1))
    inter_xmax = torch.min(xmax1.view(-1,1),xmax2.view(1,-1))
    inter_ymin = torch.max(ymin1.view(-1,1),ymin2.view(1,-1))
    inter_ymax = torch.min(ymax1.view(-1,1),ymax2.view(1,-1))
    cover_xmin = torch.min(xmin1.view(-1,1),xmin2.view(1,-1))
    cover_xmax = torch.max(xmax1.view(-1,1),xmax2.view(1,-1))
    cover_ymin = torch.min(ymin1.view(-1,1),ymin2.view(1,-1))
    cover_ymax = torch.max(ymax1.view(-1,1),ymax2.view(1,-1))

    inter_w = inter_xmax-inter_xmin
    inter_h = inter_ymax-inter_ymin
    mask = ((inter_w>=0 )&( inter_h >=0)).to(torch.float)
    # detect not overlap
    cover = (cover_xmax-cover_xmin)*(cover_ymax-cover_ymin)
    #inter_h[inter_h<0] = 0
    inter = inter_w*inter_h*mask
    #keep iou<0 to avoid gradient diasppear
    area1 = bbox1[:,2]*bbox1[:,3]
    area2 = bbox2[:,2]*bbox2[:,3]
    union = area1.view(-1,1)+area2.view(1,-1)+1e-16
    union-=inter

    iou = inter/union
    gou = iou-(cover-union)/cover
    return iou,gou
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

    inter_w = inter_xmax-inter_xmin
    inter_h = inter_ymax-inter_ymin
    mask = ((inter_w>=0 )&( inter_h >=0)).to(torch.float)
    
    # detect not overlap
    
    #inter_h[inter_h<0] = 0
    inter = inter_w*inter_h*mask
    #keep iou<0 to avoid gradient diasppear
    area1 = bbox1[:,2]*bbox1[:,3]
    area2 = bbox2[:,2]*bbox2[:,3]
    union = area1+area2 - inter+1e-16
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
    
    inter_w = inter_xmax-inter_xmin
    inter_h = inter_ymax-inter_ymin
    mask = ((inter_w>=0 )&( inter_h >=0))
    
    #inter_h[inter_h<0] = 0
    inter = inter_w*inter_h*mask.astype(float)
    inter = (inter_ymax-inter_ymin)*(inter_xmax-inter_xmin)
    area1 = ((ymax1-ymin1+1)*(xmax1-xmin1+1)).reshape(-1,1)
    area2 = ((ymax2-ymin2+1)*(xmax2-xmin2+1)).reshape(1,-1)
    union = area1+area2 - inter +1e-16
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
        fp = scores <= threshold

        #only keep trues
        ious = ious[~fp,:]
        fp = fp.sum() # transfer to scalar


        select_ids = ious.argmax(axis=1)
        #discard fps hit gt boxes has been hitted by bboxes with higher conf
        tp = len(np.unique(select_ids))
        fp += len(select_ids)- tp

        
        # groud truth with no associated predicted object
        assert (fp+tp)==n
        fn = m-tp
        p = tp/n
        r = tp/m
        assert(p<=1)
        assert(r<=1)
        ap = tp/(fp+fn+tp)
        return p,r,ap
    elif m>0 or n >0 :
        return 0,0,0
    else:
        return 1,1,1


    
def non_maximum_supression(preds,conf_threshold=0.5,nms_threshold = 0.4):
    preds = preds[preds[:,4]>conf_threshold]
    if len(preds) == 0:
        return preds      
    score = preds[:,4]
    idx = torch.argsort(score,descending=True)
    preds = preds[idx]
    dets = preds[:,:5]
    keep = []
    while len(dets)>0:
        new = dets[0]
        keep.append(new)
        ious = iou_wt_center(dets[0,:4],dets[:,:4])
        if not(ious[0]>=0.7):
            ious[0] = 1
        mask = (ious>nms_threshold)
        #hard-nms        
        dets = dets[~mask]
    return torch.stack(keep)
def non_maximum_supression_soft(preds,conf_threshold=0.5,nms_threshold=0.4):
    keep = []
    dets = preds[:,:5]
    dets = dets[dets[:,4]>conf_threshold]
    while len(dets)>0:
        val,idx = torch.max(dets[:,4],dim=0)       
        pd = dets[idx]
        dets = torch.cat((dets[:idx],dets[idx+1:]))
        ious = iou_wt_center(pd[:4],dets[:,:4])
        mask = (ious>nms_threshold)
        keep.append(pd)
        dets[mask,4] *= (1-ious[mask])*(1-val)
        dets = dets[dets[:,4]>conf_threshold]
    print(len(keep))
    return torch.stack(keep)
def visualization():
    pass
def test_pds():
    pass












