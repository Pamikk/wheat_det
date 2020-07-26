import numpy as np
import json
import os
import torch
from tqdm import tqdm

from utils import non_maximum_supression,cal_metrics
from config import Config
cfg = Config(train=False)
val = json.load(open(cfg.file))
nms_threshold = 0.7
conf_threshold = 0
img_list,gts,_ = val
thresholds = np.arange(0.5,0.76,0.05)
thresholds = [round(th,2) for th in thresholds]
pds = json.load(open(os.path.join(cfg.checkpoint,'res50v1','pred','pred_epoch_13.json')))
APs = dict.fromkeys(thresholds,0)
precisions = dict.fromkeys(thresholds,0)
recalls = dict.fromkeys(thresholds,0)
mAP = 0
for _,img in tqdm(enumerate(img_list)):
    pred = torch.tensor(pds[img])
    gt = torch.tensor(gts[img]["bbox"])
    pred_nms = non_maximum_supression(pred,None,conf_threshold, nms_threshold)
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
