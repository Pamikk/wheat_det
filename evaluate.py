import numpy as np
import json
import os
import torch
from tqdm import tqdm

from utils import non_maximum_supression_soft as nms
from utils import cal_tp_per_item,ap_per_class
from config import Config
plot = [0.5,0.75]
ls = 0
def gen_gts(anno):
    gts = torch.zeros((anno['obj_num'],4),dtype=torch.float)
    if anno['obj_num'] == 0:
        return gts
    labels = torch.tensor(anno['labels']) #ignore hard
    assert labels.shape[-1] == 4
    labels[:,ls] += labels[:,2]/2
    labels[:,ls+1] += labels[:,3]/2
    return labels
def evaluate(path,nms_threshold,conf_threshold):
    gts = json.load(open('data/val.json'))
    nms_threshold = nms_threshold
    conf_threshold = conf_threshold
    thresholds = np.around(np.arange(0.5,0.76,0.05),2)
    pds = json.load(open(path))
    mAP = 0
    batch_metrics={}
    for th in thresholds:
        batch_metrics[th] = []
    n_gt = 0
    for img in tqdm(gts.keys()):
        pred = torch.tensor(pds[img])
        pred = pred.reshape(-1,5)
        gt = gen_gts(gts[img])
        n_gt += gt.shape[0]
        pred_nms = nms(pred,conf_threshold, nms_threshold)       
        for th in batch_metrics:
            batch_metrics[th].append(cal_tp_per_item(pred_nms,gt,th))
    metrics = {}
    for th in batch_metrics:
        tps,scores= [np.concatenate(x, 0) for x in list(zip(*batch_metrics[th]))]
        precision, recall, AP= ap_per_class(tps, scores, n_gt)
        mAP += np.mean(AP)
        if th in plot:
            metrics['AP/'+str(th)] = np.mean(AP)
            metrics['Precision/'+str(th)] = np.mean(precision)
            metrics['Recall/'+str(th)] = np.mean(recall)
    metrics['mAP'] = mAP/len(thresholds)
    for k in metrics:
        print(k,':',metrics[k])
    return metrics['mAP']
path = '../checkpoints/debug/pred/pred_test.json'
best = 0
nmst = 0.5
conf = 0.95
print(f'nms:{nmst},conf:{conf}')
val = evaluate(path,nmst,conf)
if val >best:
    best = val
    bestp = (nmst,conf)
print('++++++++++++++++++++++++++')
nmst,conf = bestp
print(f'nms:{nmst},conf:{conf},mAP:{best}')

#nms:0.5,conf:0.95,mAP:0.6331328052886033
