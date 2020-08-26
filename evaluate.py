import numpy as np
import json
import os
import torch
from tqdm import tqdm

from utils import cal_metrics
from utils import non_maximum_supression_soft as non_maximum_supression
from config import Config
cfg = Config(mode='trainval')
val = json.load(open(cfg.file))
nms_threshold = 0
conf_threshold = 0
if len(val)==3:
    img_list,gts,_ = val
else:
    img_list,gts = val
thresholds = np.arange(0.5,0.76,0.05)
thresholds = [round(th,2) for th in thresholds]
pds = json.load(open(os.path.join(cfg.checkpoint,'wtfv4','pred','pred_test.json')))
APs = dict.fromkeys(thresholds,0)
precisions = dict.fromkeys(thresholds,0)
recalls = dict.fromkeys(thresholds,0)
ap_ = dict.fromkeys(thresholds,0)
mAP = 0
for _,img in tqdm(enumerate(img_list)):
    pred = torch.tensor(pds[img])
    gt = torch.tensor(gts[img]["bbox"])
    #print(pred.shape)
    pred_nms = non_maximum_supression(pred,conf_threshold, nms_threshold)
    #pred_nms = pred[:,:4]
    total = 0
    for th in thresholds:
        p,r,ap = cal_metrics(pred_nms,gt,threshold= th)
        APs[th] += ap
        precisions[th] += p
        recalls[th] += r
        total +=ap
    mAP += 1.0*total/len(thresholds)
count = len(img_list)
for th in thresholds:
    print('AP/'+str(th)+':',1.0*APs[th]/count)
    print('Precision/'+str(th)+':',1.0*precisions[th]/count)
    print('Recall/'+str(th)+':',1.0*recalls[th]/count)
mAP = 1.0*mAP/count
print(mAP)
