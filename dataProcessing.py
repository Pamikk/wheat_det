import torch.utils.data as data
import torch
import json
import numpy as np
import random
import cv2
import os

from imutils import augment,resize,color_normalize

#stack functions for collate_fn
#Notice: all dicts need have same keys and all lists should have same length
def stack_dicts(dicts):
    if len(dicts)==1:
        return dicts[0]
    if len(dicts)==0:
        return None
    res = {}
    for k in dicts[0].keys():
        res[k] = [obj[k] for obj in dicts]
    return res

def stack_list(lists):
    if len(lists)==1:
        return lists[0]
    if len(lists)==0:
        return None
    res = list(range(len(lists[0])))
    for k in range(len(lists[0])):
        res[k] = torch.stack([obj[k] for obj in lists])
    return res
class WheatDet(data.Dataset):
    def __init__(self,cfg,train=True):
        self.img_path = cfg.img_path
        self.cfg = cfg
        file = json.load(open(cfg.file,'r'))
        self.imgs = file[0]
        self.annos = file[1]
        self.train = train
        self.max_num = 120
    def __len__(self):
        return len(self.imgs)

    def img_to_tensor(self,img):
        return torch.tensor(np.transpose(img,[2,0,1]),dtype=torch.float)
    def gen_gts(self,labels,pad=(0,0)):
        gts = torch.zeros(labels.shape,dtype=torch.float)
        if len(labels) == 0:
            return gts
        n = len(labels)
        xmins = labels[:,0] + pad[0]
        ymins = labels[:,1] + pad[1]
        ws = labels[:,2]
        hs = labels[:,3]
        xs,ys = xmins + ws/2, ymins + hs/2

        gts = torch.tensor(np.stack([xs,ys,ws,hs],axis=1),dtype=torch.float)
        return gts
    def fill_with_zeros(self,labels,n):
        gts = torch.zeros((self.max_num,labels.shape[-1]),dtype=torch.float)
        gts[:n,:] = labels
        return gts
    def get_trans_gts(self,labels,size,mat=np.eye(3),flip=True):
        #transfer
        if len(labels)== 0:
            return labels
        h,w = size
        cos = abs(mat[0,0])
        sin = abs(mat[0,1])        
        xs = labels[:,0]
        ys = labels[:,1]
        ws = labels[:,2]
        hs = labels[:,3]
        if flip:
            xs = w-1-xs
            ys = h-1-ys
        
        sy = 1/h #normalize to [0,1]
        sx = 1/w
        n = len(labels)

        pts = np.stack([xs,ys,np.ones([n])],axis=1).T
        tpts = torch.tensor(np.dot(mat,pts).T)
        labels[:,0] = tpts[:,0]*sx
        labels[:,1] = tpts[:,1]*sy
        mask = labels[:,0]>=0
        mask *= labels[:,0]<1
        mask *= labels[:,1]>=0
        mask *= labels[:,1]<1    

        labels[:,2] = (cos*ws + sin*hs)*sx
        labels[:,3] = (cos*hs + sin*ws)*sy
        labels = labels[mask,:]
        return labels
    #process ground truths
    #useful for keypoint heatmaps
    def gen_heatmap(self,labels,sigmas):
        tsize = self.cfg.int_shape
        heatmaps = [np.zeros(tsize) for _ in sigmas]
        for label in labels:
            tmp = np.zeros(tsize)
            x = int(label[0]*tsize[1])
            y = int(label[1]*tsize[0])
            if 0<=x< tsize[1] and 0<=y<tsize[0]:
                tmp[y,x] = 1
                for i,sigma in enumerate(sigmas):
                    tmp_ = cv2.GaussianBlur(tmp,sigma,0)
                    tmp_ /= tmp_.max()
                    heatmaps[i] = np.maximum(heatmaps[i],tmp_)
                # mutliple will calculate only once 
                # use max to avoid two center make one middle point with greater score
            else:
                print(label)
        heatmaps = [torch.tensor(heatmap,dtype=torch.float) for heatmap in heatmaps]
        return heatmaps
    def gen_attention_map(self,labels):
        tsize = self.cfg.int_shape
        heatmap = np.zeros(tsize)
        for label in labels:
            tmp = np.zeros(tsize)
            x = int(label[0]*tsize[1])
            y = int(label[1]*tsize[0])
            if 0<=x< tsize[1] and 0<=y<tsize[0]:
                heatmap = np.maximum(heatmaps,tmp)
                # mutliple will calculate only once 
                # use max to avoid two center make one middle point with greater score
        heatmap =torch.tensor(heatmap,dtype=torch.float)
        return heatmap

    def pad_to_square(self,img):
        h,w,_= img.shape
        diff = abs(h-w)
        if h>w:
            pad = (diff//2,0,diff-diff//2,0)
        else:
            pad = (0,diff//2,0,diff-diff//2)
        img = cv2.copyMakeBorder(img,pad[0],pad[1],pad[2],pad[3],cv2.BORDER_CONSTANT,0)
        return img,(pad[0],pad[1])

    def __getitem__(self,idx):
        name = self.imgs[idx]
        anno = self.annos[name]
        img = cv2.imread(os.path.join(self.img_path,name+'.jpg'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        if h!=w:
            img,pad = self.pad_to_square(img)
        else:
            pad = (0,0)
        h,w,_ = img.shape
        labels = self.gen_gts(np.array(anno["bbox"],dtype=np.float),pad)
        if self.train:
            if random.uniform(0,1)>=0.25:
                rot = random.uniform(-1,1)*self.cfg.rot
            else:
                rot = 0
            if random.uniform(0,1)>=0.5:
                flip = True
            else:
                flip = False
            if random.uniform(0,1)>=0.5:
                vs = self.cfg.valid_scale*random.uniform(-1,1)
            else:
                vs = 0
            dst,mat = augment(img,rot,vs,flip)
            data = resize(dst,self.cfg.inp_size)
            labels = self.get_trans_gts(labels,(h,w),mat,flip)
            data = color_normalize(data,self.cfg.RGB_mean)
            data = self.img_to_tensor(data)
            #labels = self.fill_with_zeros(labels,n)
            return data,labels        
        else:
            #validation set
            data = color_normalize(img,self.cfg.RGB_mean)
            data = resize(data,self.cfg.inp_size)
            data = self.img_to_tensor(data)
            #labels = self.fill_with_zeros(labels,n)
            info ={'size':(h,w),'img_id':name}
            return data,labels,info
    def collate_fn(self,batch):
        if self.train:
            data,labels = list(zip(*batch))
        else:
            data,labels,info = list(zip(*batch))
            info = stack_dicts(info)   
        tmp =[]
        data = torch.stack(data)            
                
        for i,bboxes in enumerate(labels):
            if len(bboxes)>0:
                label = torch.zeros(len(bboxes),5)
                label[:,1:] = bboxes
                label[:,0] = i
                tmp.append(label)
        if len(tmp)>0:
            labels = torch.cat(tmp,dim=0)
        else:
            labels = torch.tensor(tmp,dtype=torch.float)
        if self.train:
            return data,labels
        else:
            return data,labels,info

                





