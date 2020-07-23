import matplotlib.pyplot as plt 
import math
import torch
import numpy as np
import tensorflow as tf
import os 
class Logger(object):
    def __init__(self,log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.files = {'val':open(os.path.join(log_dir,'val.txt'),'a+'),'train':open(os.path.join(log_dir,'train.txt'),'a+')}
    def write_line2file(self,mode,string):
        self.files[mode].write(string+'\n')
        self.files[mode].flush()
    def write_loss(self,epoch,losses,lr):
        tmp = str(epoch)+'\t'+str(lr)+'\t'
        with self.writer.as_default():
            tf.summary.scalar('lr',lr,step=epoch)
            for k in losses:            
                tf.summary.scalar(k,losses[k],step=epoch)
                tmp+= str(round(losses[k],3))+'\t'
            self.writer.flush()
        self.write_line2file('train',tmp)
    def write_metrics(self,epoch,metrics,save=[],step=1):
        tmp =str(epoch)+'\t'
        with self.writer.as_default():
            for k in metrics:
                if k in save:
                    tmp +=str(metrics[k])+'\t'            
                tf.summary.scalar('val'+k,metrics[k],step=epoch)
            self.writer.flush()
        self.write_line2file('val',tmp)

def iou_wo_center(w1,h1,w2,h2):
    inter = torch.min(w1,w2)*torch.min(h1,h2)
    union = w1*h1 + w2*h2 - inter
    return inter/union
def iou_wt_center(bbox1,bbox2):
    #only for torch 
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

    inter = (inter_ymax-inter_ymin)*(inter_xmax-inter_xmin)
    area1 = (ymax1-ymin1)*(xmax1-xmin1)
    area2 = (ymax2-ymin2)*(xmax2-xmin2)
    union = area1+area2 - inter
    return inter/union
def cal_AP(iou,scores,threshold=0.5):


