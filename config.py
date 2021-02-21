
import numpy as np
import random
import json

from stats import kmeans
anchors =[[0.038, 0.036], [0.059, 0.048], [0.066, 0.063], [0.066, 0.071], [0.092, 0.079], [0.102, 0.085], [0.105, 0.105], [0.146, 0.138], [0.186, 0.168]]
#anchors = 
path ='data/train.json' #annotation path for anchor calculation
def cal_anchors(sizes=None,num=9):
    #As in https://github.com/eriklindernoren/PyTorch-YOLOv3
    # randomly scale as sizes if sizes is not None    
    annos = json.load(open(path,'r'))
    allb = []
    for name in annos:
        anno = annos[name]
        size = anno['size']
        h,w = size
        for bbox in anno['labels']:
            bw,bh = bbox[2:]
            if bw<0 or bh<0:
                print(name,bbox)
                exit()
            t = max(w,h)
            if sizes == None:
                scale = t
            else:
                scale = sizes
            allb.append((bw/t,bh/t))
    km = kmeans(allb,k=num,max_iters=1000)
    km.initialization()
    km.iter(0)
    km.print_cs()
    anchors = km.get_centers()
    km.cal_all_dist()  
    return anchors,km
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.img_path = '../dataset/global-wheat/train'
        self.checkpoint='../checkpoints'
        self.cls_num = 0       
        self.res = 50
        self.size = 512
        self.multiscale = 1
        self.sizes = [512]#list(range(self.size-32*self.multiscale,self.size+1,32)) 
        self.nms_threshold = 0.5
        self.dc_threshold = 0.95
        
        
        #loss args
        #self.anchors = [[0.26533935,0.33382434],[0.66550966,0.56042827],[0.0880948,0.11774004]] #w,h normalized by max size
        #self.anchors = [[0.76822971,0.57259308],[0.39598597,0.47268035],[0.20632625,0.26720238],[0.07779112,0.10330848]]
        self.anchors= anchors  
        self.anchor_divide=[(7,8),(4,5,6),(0,1,2,3)]
        self.anchor_num = len(self.anchors)
        
        self.bs = 8       
        self.pre_trained_path = '../network_weights'
        #train_setting
        self.lr = 0.001
        self.weight_decay=5e-4
        self.momentum = 0.9
        #lr_scheduler
        self.min_lr = 5e-5
        self.lr_factor = 0.25
        self.patience = 10
        #exp_setting
        self.save_every_k_epoch = 15
        self.val_every_k_epoch = 10
        self.adjust_lr = False
        #loss hyp
        self.obj_scale = 2
        self.noobj_scale = 10
        self.cls_scale = 1
        self.reg_scale = .5#for giou
        self.ignore_threshold = 0.7
        self.match_threshold = 0#regard as match above this threshold
        self.base_epochs = [-1]#base epochs with large learning rate,adjust lr_facter with 0.1
        if mode=='train':
            self.file=f'./data/train.json'
            self.bs = 32 # batch size
            
            #augmentation parameter
            self.flip = True
            self.rot = 25
            self.crop = 0.3
            self.trans = .3
            self.scale = 0.2
            self.valid_scale = 0.25
            self.mosaic = 0.01

        elif mode=='val':
            self.size = 512
            self.file = './data/val.json'
        elif mode=='trainval':
            self.size = 512
            self.file = './data/trainval.json'
        elif mode=='test':
            self.file = './data/val.json'
        
