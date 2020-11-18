import matplotlib.pyplot as plt 
import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os 
import json
import pandas as pd
from tqdm import tqdm
voc_classes= {'__background__':0, 'aeroplane':1, 'bicycle':2, 
          'bird':3, 'boat':4, 'bottle':5,'bus':6, 'car':7,
           'cat':8, 'chair':9,'cow':10, 'diningtable':11, 'dog':12,
            'horse':13,'motorbike':14, 'person':15, 'pottedplant':16,
            'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
voc_indices = dict([(voc_classes[k]-1,k) for k in voc_classes])
class Logger(object):
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.files = {'val':open(os.path.join(log_dir,'val.txt'),'a+'),'train':open(os.path.join(log_dir,'train.txt'),'a+')}
    def write_line2file(self,mode,string):
        self.files[mode].write(string+'\n')
        self.files[mode].flush()
    def write_loss(self,epoch,losses,lr):
        tmp = str(epoch)+'\t'+str(lr)+'\t'
        print('Epoch',':',epoch,'-',lr)
        writer = SummaryWriter(log_dir=self.log_dir)
        writer.add_scalar('lr',math.log(lr),epoch)
        for k in losses:
            if losses[k]>0:            
                writer.add_scalar('Train/'+k,losses[k],epoch)            
                print(k,':',losses[k])
                #self.writer.flush()
        tmp+= str(round(losses['all'],5))+'\t'
        self.write_line2file('train',tmp)
        writer.close()
    def write_metrics(self,epoch,metrics,save=[],mode='Val',log=True):
        tmp =str(epoch)+'\t'
        print("validation epoch:",epoch)
        writer = SummaryWriter(log_dir=self.log_dir)
        for k in metrics:
            if k in save:
                tmp +=str(metrics[k])+'\t'
            if log:
                tag = mode+'/'+k            
                writer.add_scalar(tag,metrics[k],epoch)
                #self.writer.flush()
            print(k,':',metrics[k])
        
        self.write_line2file('val',tmp)
        writer.close()

def iou_wo_center(w1,h1,w2,h2):
    #assuming at the same center
    #return a vector nx1
    inter = torch.min(w1,w2)*torch.min(h1,h2)
    union = w1*h1 + w2*h2 - inter
    ious = inter/union
    ious[ious!=ious] = torch.tensor(0.0) #avoid nans
    return ious
def generalized_iou(bbox1,bbox2):
    #return shape nx1
    bbox1 = bbox1.view(-1,4)
    bbox2 = bbox2.view(-1,4)
    assert bbox1.shape[0]==bbox2.shape[0]
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
    union = area1+area2 - inter
    ious = inter/union
    gious = ious-(cover-union)/cover
    ious[ious!=ious] = torch.tensor(0.0) #avoid nans
    gious[gious!=gious] = torch.tensor(0.0) #avoid nans
    return ious,gious
def cal_gious_matrix(bbox1,bbox2):
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
    union = area1.view(-1,1)+area2.view(1,-1)
    union -= inter

    ious = inter/union
    gious = iou-(cover-union)/cover
    ious[ious!=ious] = torch.tensor(0.0) #avoid nans
    gous[gous!=gous] = torch.tensor(0.0) #avoid nans 
    return ious,gious
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
    union = area1+area2 - inter
    ious = inter/union
    ious[ious!=ious] = torch.tensor(0.0)
    return ious
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
    area1 = ((ymax1-ymin1)*(xmax1-xmin1)).reshape(-1,1)
    area2 = ((ymax2-ymin2)*(xmax2-xmin2)).reshape(1,-1)
    union = area1+area2 - inter
    ious = inter/union
    ious[ious!=ious] = 0
    return ious

def ap_per_class(tp, conf,n_gt):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]


    # Create Precision-Recall curve and compute AP for each class
    n_p = len(tp)
    if (n_gt==0)and(n_p):
        return 1,1,1
    elif (n_gt==0) or (n_p==0):
        return 0,0,0
    p = tp.sum()/n_p
    r = tp.sum()/n_gt
    ap = tp.sum()/(n_gt+n_p-tp.sum())
    return p,r,ap

def write_to_csv(res,fname="submission.csv"):
    cols = ["image_id","PredictionString"]
    for data in res:
        tmp = data['PredictionString']
        bboxes = []
        for bbox in tmp:
             bboxes.append(f"{bbox[-1]} {bbox[0]-bbox[2]/2} {bbox[1]-bbox[3]/2} {bbox[2]} {bbox[3]}")
        data['PredictionString']= ' '.join(bboxes)
    test_df = pd.DataFrame(res, columns=['image_id', 'PredictionString'])
    test_df.to_csv('submission.csv', index=False)
def cal_tp_per_item(pds,gts,threshold=0.5):
    assert (len(pds.shape)>1) and (len(gts.shape)>1)
    pds = pds.cpu().numpy()
    gts = gts.cpu().numpy()
    n = pds.shape[0]
    m = gts.shape[0]
    tps = np.zeros(n)
    scores = pds[:,-1]
    pdbboxes = pds[:,:4].reshape(-1,4)
    gtbboxes = gts.reshape(-1,4)
    selected = np.zeros(m)
    for i in range(n):
        if m==0:
            break 
        pdbbox = pdbboxes[i]
        ious = iou_wt_center_np(pdbbox,gtbboxes)
        iou = ious.max()
        best = ious.argmax()
        if iou >=threshold  and selected[best] !=1:
            selected[best] = 1
            tps[i] = 1.0
            m -=1          
    return [tps,scores]
    
def xyhw2xy(boxes_):
    boxes = boxes_.clone()
    boxes[:,0] = boxes_[:,0] - boxes_[:,2]/2
    boxes[:,1] = boxes_[:,1] - boxes_[:,3]/2
    boxes[:,2] = boxes_[:,0] + boxes_[:,2]/2
    boxes[:,3] = boxes_[:,1] + boxes_[:,3]/2
    return boxes
def xy2xyhw(boxes):
    boxes_ = boxes.clone()
    boxes_[:,0] = (boxes[:,0] + boxes[:,2])/2
    boxes_[:,1] = (boxes[:,1] + boxes[:,3])/2
    boxes_[:,2] = boxes[:,2] - boxes[:,0]
    boxes_[:,3] = boxes[:,3] - boxes[:,1]
    return boxes_

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes    

    
def non_maximum_supression(preds,conf_threshold=0.5,nms_threshold = 0.4):
    if len(preds)==0:
        return preds
    preds = preds[preds[:,4]>conf_threshold]
    if len(preds) == 0:
        return preds      
    score = preds[:,4]
    idx = torch.argsort(score,descending=True)
    dets = preds[idx]
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
    return torch.stack(keep).reshape(-1,5)
def non_maximum_supression_soft(preds,conf_threshold=0.5,nms_threshold=0.4):
    if len(preds)==0:
        return preds
    keep = []
    dets = preds[preds[:,4]>conf_threshold]
    if len(dets)==0:
        return dets
    while len(dets)>0:
        val,idx = torch.max(dets[:,4],dim=0)
        if val<=conf_threshold:
            continue        
        pd = dets[idx]
        dets = torch.cat((dets[:idx],dets[idx+1:]))
        ious = iou_wt_center(pd[:4],dets[:,:4])
        mask = (ious>nms_threshold)
        keep.append(pd)
        dets[mask,4] *= (1-ious[mask])
        dets = dets[dets[:,4]>conf_threshold]
    return torch.stack(keep).reshape(-1,5)
#nms:0.5,conf:0.95,mAP:0.6331328052886033












