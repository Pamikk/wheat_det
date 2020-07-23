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
    def gen_gts(self,labels):
        gts = torch.zeros(labels.shape,dtype=torch.float)
        if len(labels) == 0:
            return gts
        n = len(labels)
        xmins = labels[:,0]
        ymins = labels[:,1]
        ws = labels[:,2]
        hs = labels[:,3]
        xs,ys = xmins + ws/2, ymins + hs/2

        gts = torch.tensor(np.stack([xs,ys,ws,hs],axis=1),dtype=torch.float)
        return gts
    def fill_with_zeros(self,labels,n):
        gts = torch.zeros((self.max_num,labels.shape[-1]),dtype=torch.float)
        gts[:n,:] = labels
        return gts
    def get_trans_gts(self,labels,size,mat=np.eye(3),crop=(0,0),flip=True):
        #transfer
        if len(labels)== 0:
            return labels
        h,w,_ = size
        cos = abs(mat[0,0])
        sin = abs(mat[0,1])        
        xs = labels[:,0]
        ys = labels[:,1]
        ws = labels[:,2]
        hs = labels[:,3]
        if flip:
            xs = w-1-xs
            ys = h-1-ys
        xs -= crop[1]
        ys -= crop[0]
        
        sy = 1/(h-crop[0]) #normalize to [0,1]
        sx = 1/(w-crop[1])
        n = len(labels)
        
        tsize = self.cfg.inp_size
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
    def gen_heatmap(self,labels,sigma):
        tsize = self.cfg.int_shape
        gts = torch.zeros(tsize,dtype=torch.float)
        for label in labels:
            tmp = np.zeros(tsize,dtype=float)
            if label[2]!=0:
                x = int(label[0]*tsize[1])
                y = int(label[1]*tsize[0])
                if 0<=x< tsize[1] and 0<=y<tsize[0]:
                    tmp[y,x] = 1
                    tmp = torch.tensor(cv2.GaussianBlur(tmp,sigma,0),dtype=torch.float)
                    tmp /= tmp.max()
                    gts = torch.max(gts,tmp)
                    # mutliple will calculate only once 
                    # use max to avoid two center make one middle point with greater score
                else:
                    print(label)
            else:
                break
        return gts

    def __getitem__(self,idx):
        name = self.imgs[idx]
        anno = self.annos[name]
        img = cv2.imread(os.path.join(self.img_path,name+'.jpg'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        n = len(anno["bbox"])

        labels = self.gen_gts(np.array(anno["bbox"],dtype=np.float))
        if self.train:
            #data augmentation,0.25 not augment,0.25 only rotate, 0.25 only crop, 0.25 both rotate and crop
            # 0.5 for flip independently
            pr,pc = random.randint(0,1),random.randint(0,1)
            if pr == 1:
                rot = random.uniform(-1,1)*self.cfg.rot
            else:
                rot = 0
            if pc == 1:
                crop = (random.uniform(-1,1)*self.cfg.crop,random.uniform(-1,1)*self.cfg.crop)
                crop = (int(crop[0]*h),int(crop[1]*w))
            else:
                crop = (0,0)
            if random.uniform(0,1)>=0.5:
                flip = True
            else:
                flip = False
            if random.uniform(0,1)>=0.5:
                vs = self.cfg.valid_scale*random.uniform(-1,1)
            else:
                vs = 0
            dst,mat = augment(img,rot,crop,vs,flip)
            data = resize(dst,self.cfg.inp_size)
            labels = self.get_trans_gts(labels,img.shape,mat,crop,flip)
            n = len(labels)
            heatmaps = list(range(4))
            for i in range(4):
                heatmaps[3-i] = self.gen_heatmap(labels,self.cfg.sigmas[i])
            data = color_normalize(data,self.cfg.RGB_mean)
            data = self.img_to_tensor(data)
            #labels = self.fill_with_zeros(labels,n)
            info ={'size':(h,w)}
            return data,labels,heatmaps,info          
        else:
            #validation set
            data = color_normalize(img,self.cfg.RGB_mean)
            data = resize(data,self.cfg.inp_size)
            data = self.img_to_tensor(data)
            #labels = self.fill_with_zeros(labels,n)
            info ={'size':(h,w)}
            return data,labels,info
    def collate_fn(self,batch):
        if self.train:
            data,labels,heatmaps,info  = list(zip(*batch))
            heatmaps = stack_list(heatmaps)
        else:
            data,labels,info = list(zip(*batch))
        tmp =[]
        data = torch.stack(data)            
        info = stack_dicts(info)           
        for i,bboxes in enumerate(labels):
            if len(bboxes)>0:
                label = torch.zeros(len(bboxes),5)
                label[:,1:] = bboxes
                label[:,0] = i
                tmp.append(label)
        labels = torch.cat(tmp,dim=0)
        if self.train:
            return data,labels,heatmaps,info,self.cfg.inp_size
        else:
            return data,labels,info,self.cfg.inp_size

                





