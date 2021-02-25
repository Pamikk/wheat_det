import numpy as np
import random
import json

from stats import kmeans
anchors =[[0.002,0.00098],[0.038, 0.036], [0.059, 0.048], [0.066, 0.063], [0.066, 0.071], [0.092, 0.079], [0.102, 0.085], [0.105, 0.105], [0.146, 0.138], [0.186, 0.168],[0.96,0.7]]
#anchors = [[0.038, 0.037], [0.058, 0.048], [0.066, 0.064], [0.067, 0.072], [0.092, 0.079], [0.103, 0.084], [0.104, 0.106], [0.147, 0.138], [0.187, 0.168]]
path ='data/train.json' #annotation path for anchor calculation 
annos = json.load(open(path,'r'))
allb = []
mw,mh=0,0
miw,mih=1,1
target=[1,2,3,4]
for name in annos:
    anno = annos[name]
    size = anno['size']
    h,w = size
    for bbox in anno['labels']:
        bw,bh = bbox[2:]
        if bw<0 or bh<0:
            print(name,bbox)
            exit()
        if bw==0 or bh==0:
            print(bbox)
            continue
        t = max(w,h)
        allb.append((bw/t,bh/t))
        if mw < bw/t:
            target[0] = (bw/t,bh/t)
        if mh < bh/t:
            target[1] = (bw/t,bh/t)
        if miw > bw/t:
            target[2] = (bw/t,bh/t)
        if mih < bh/t:
            target[3] = (bw/t,bh/t)
        mw = max(bw/t,mw)
        miw = min(bw/t,miw)
        mh = max(bh/t,mh)
        mih = min(bh/t,mih)
print(target)
for num in range(7,14):
    km = kmeans(allb,k=num,max_iters=1000)
    km.initialization()
    km.iter(0)
    anchors = km.get_centers()
    #km.cal_all_dist() 
    print(anchors)
    km.print_cs()
    km.cal_all_dist() 