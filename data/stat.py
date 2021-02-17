import os
import cv2
import json
import numpy as np
import time
class kmeans(object):
    def __init__(self,vals,k=3,max_iters=200):
        self.vals = np.array(vals)
        print(self.vals.shape)
        self.dim = self.vals.shape[-1]
        self.k = k
        self.maxi = max_iters
        self.num = len(vals)
        self.terminate = False
    def initialization(self):
        assign = np.zeros(self.num,dtype=int)
        self.centers = list(range(self.k))
        k = self.k
        partn = self.num//k
        for i in range(k):
            assign[i*partn:(i+1)*partn] = i
        self.assign = assign
    def update_center(self):
        for i in range(self.k):
            if np.sum(self.assign==i)>0:
                #avoid empty cluster leading Error
                self.centers[i] = np.mean(self.vals[self.assign==i],axis=0)
        if type(self.centers) != np.ndarray:
            self.centers = np.array(self.centers)
    def cal_distance(self,obj1,obj2):
        #1-iou
        obj1 = obj1.reshape(-1,2)
        obj2 = obj2.reshape(-1,2)
        inter = np.minimum(obj1[:,0].reshape(-1,1),obj2[:,0].reshape(1,-1))*np.minimum(obj1[:,1].reshape(-1,1),obj2[:,1].reshape(1,-1))
        union = (obj1[:,0]*obj1[:,1]).reshape(-1,1) + (obj2[:,0]*obj2[:,1]).reshape(1,-1) - inter +1e-16
        return 1-inter/union
    def update_assign(self):
        self.terminate = True
        centers = self.centers
        for i in range(self.num):
            val = self.vals[i]
            tmp = self.cal_distance(val,np.stack(centers,axis=0))
            id = np.argmin(tmp)
            if id != self.assign[i]:
                self.assign[i] = id
                self.terminate= False
    def iter(self,num):
        self.update_center()
        self.update_assign()
        if self.terminate:
            self.print_cs()
            return
        else:
            if num == self.maxi:
                print("reach max iterations")
                self.print_cs()
                return
            else:
                return self.iter(num+1)
    def print_cs(self):
        for i in range(self.k):
            print(list(self.centers[i]),np.sum(self.assign==i))
        print([list(c) for c in self.centers])
        print(self.cal_distance(self.centers,self.centers))

    
def cal_ratios(data):
    ratios=[]
    gts = data
    imgs = list(gts.keys())
    mh = 0
    mw = 0
    for img in imgs:
        h,w = gts[img]['size']
        t = max(h,w)
        for bbox in gts[img]["bbox"]:
           h = bbox[3]
           w = bbox[2]
           mh = max(h,mh)
           mw = max(w,mw)
           ratios.append((w,h))
    km = kmeans(ratios,k=6)
    km.initialization()
    _ = km.iter(0)
    print(mh,mw)
    #return km    
def count_overlap(annos,tsize):
    mc = 0
    for name in annos:
        size = annos[name]['size']
        h,w = size
        count = np.zeros(tsize)
        t = max(w,h)
        for bbox in annos[name]['bbox']:
            xmin,ymin,bw,bh = bbox            
            xc = (xmin + bw/2)/t
            yc = (ymin + bh/2)/t
            count[int(tsize[0]*yc),int(tsize[1]*xc)]+=1
        assert(count.sum()==annos[name]['obj_num'])
        mc = max(count.max(),mc)
    print(mc)
def analyze_hw(annos):
    allb = []
    mh,mw = 1,1
    mxh,mxw = 0,0
    for name in annos:
        size = annos[name]['size']
        h,w = size
        t = max(w,h)
        for bbox in annos[name]['bbox']:
            _,_,bw,bh = bbox            
            allb.append((bw,bh))
            mh = min(mh,bh)
            mxh = max(mxh,bh)
            mw = min(mw,bw)
            mxw = max(mxw,bw)
    km = kmeans(allb,k=6,max_iters=500)
    km.initialization()
    km.iter(0)  
    print(mh,mw,mxh,mxw)
def analyze_xy(annos):
    for name in annos:
        size = annos[name]['size']
        h,w,_ = size
        for bbox in annos[name]['bbox']:
            x,y,bw,bh= bbox
            if y+bh > h or x+bw >w:
                print('???')
    print('finish')
def analyze_num(annos):
    mc =0 
    for name in annos:
        mc = max(mc,annos[name]['obj_num'])
    print(mc)
def analyze_size(annos):
    res = {}
    res2 = {}
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        ts = max(w,h)
        if ts in res.keys():
            res[ts]+=1
        else:
            res[ts] = 1
        ts = round(max(h,w)/32)*32
        if ts in res2.keys():
            res2[ts]+= 1/len(annos)
        else:
            res2[ts] = 1/len(annos)
    res2 = {k: v for k, v in sorted(res2.items(), key=lambda item: item[1])}
    print(res)
    print(len(res))
    print(res2)
    print(len(res2))
def cal_color_mean(data):
    imgs = data[0]
    img_path = 'train'
    mm = []
    for name in imgs:
        img = cv2.imread(os.path.join(img_path,name+'.jpg'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mm.append(img.mean(axis=0).mean(axis=0))
    mm = np.array(mm)
    print(mm.shape)
    print(mm.mean(axis=0))

start = time.time()
train = json.load(open('annotation.json'))
analyze_hw(train)
#[0.10578342841802468, 0.06323927300767092] 36697
#[0.16164473925192566, 0.1285921540103818] 8958
#[0.060309932827680185, 0.043605377293796585] 35408
#[0.059888260849536704, 0.07869134364105129] 42738
#[0.09009376823524509, 0.11350195752099658] 23992
anchors =[[53.87225938102254, 44.48974039224577], [63.62329042081948, 76.3607212070875], 
          [98.12110508830222, 53.52514425598883],  [81.62511337056661, 118.2413193947205],
        [125.43391003460205, 84.6516435986159], [158.20213028712567, 156.73488731089842]]