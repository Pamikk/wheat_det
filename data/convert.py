import json
import csv
import os
import cv2
import random
   
def train_val_split(data,ratio=0.2,num=200):
    train_gts={}
    val_gts={}
    trainval = {}
    gts = data
    imgs = list(gts.keys())
    random.shuffle(imgs)
    n=len(imgs)
    for i in range(n):
        if i <= ratio*n:
            val_gts[imgs[i]]=gts[imgs[i]]
        else:
            if i-ratio*n<num:
                trainval[imgs[i]]=gts[imgs[i]]
            train_gts[imgs[i]]=gts[imgs[i]]
    return train_gts,val_gts,trainval
def add_imgs_wo_bboxes(path,data):
    gts = data
    num=0
    for name in os.listdir(path):
        imgid,_=os.path.splitext(name)
        if imgid not in gts.keys():
            gts[imgid]={}
            gts[imgid]["labels"]=[]
            img = cv2.imread(os.path.join(path,name))
            h,w,_ = img.shape
            gts[imgid]['size']=[h,w]
            gts[imgid]['obj_num'] = 0
            num +=1
    data=gts
    return data

def read_csv(path):
    imgs=[]
    gt={}
    with open(path,newline='') as csvfile:
        readers = csv.DictReader(csvfile)
        for row in readers:
            img_id=row['image_id']
            tmp = row['bbox']
            tmp = tmp.strip('[')
            tmp = tmp.strip(']')
            bbox = tmp.split(',')
            h = int(row['height'])
            w = int(row['width'])
            bbox[0] = float(bbox[0])
            bbox[1] = float(bbox[1])
            bbox[2] = float(bbox[2])
            bbox[3] = float(bbox[3])
            if img_id in gt.keys():
                gt[img_id]["labels"].append(bbox)
                gt[img_id]['obj_num']+=1
            else:
                gt[img_id]={}
                gt[img_id]['labels']=[bbox]
                gt[img_id]['size']=[h,w]
                gt[img_id]['obj_num']= 1
    return gt        

path = '../../dataset/global-wheat'
data = read_csv(os.path.join(path,"train.csv"))
data = add_imgs_wo_bboxes(os.path.join(path,"train"),data)
json.dump(data,open('annotation.json','w'))
train,val,trainval= train_val_split(data)
json.dump(train,open('train.json','w'))
json.dump(val,open('val.json','w'))
json.dump(trainval,open('trainval.json','w'))