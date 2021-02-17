###libs
import os
import argparse
import torch
from torch.utils.data import DataLoader
import json
###files
from config import Config as cfg
from dataProcessing import VOC_dataset as dataset
from dataProcessing import Testset
from models.network import NetAPI
from trainer import Trainer
import warnings
from loss_funcs import LossAPI
warnings.filterwarnings('ignore')
def get_imgs(path):
    imgs =[]
    for dirname, dirs, filenames in os.walk(path,followlinks=True):
        #print(dirs)
        for filename in filenames:
            #print(filenames)
            if ".jpg" in filename:
                imgs.append(os.path.join(dirname, filename))
    json.dump(imgs,open("data/test.json","w"))
def main(args,cfgs):
    #get data config
    config  = cfgs['test']
    #get_imgs('../dataset/global-wheat')
    config.file="data/test.json"
    test_set = Testset(config)
    test_loader = DataLoader(test_set,batch_size=config.bs,shuffle=False,pin_memory=False)
    datasets = {'test':test_loader}
    config.exp_name = args.exp
    config.device = torch.device("cuda")
    torch.cuda.empty_cache()
    #network
    network = NetAPI(config,args.net,init=not args.resume)
    loss = LossAPI(config,args.loss)
    torch.cuda.empty_cache()
    det = Trainer(config,datasets,network,loss,(args.resume,1))
    det.test()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="start from epoch?")
    parser.add_argument("--exp",type=str,default='exp',help="name of exp")
    parser.add_argument("--net",type=str,default='yolo',help="network type:yolo")
    parser.add_argument("--bs",type=int,default=16,help="batchsize")
    parser.add_argument("--loss",type=str,default='yolo',help="loss type")
    args = parser.parse_args()
    cfgs={}
    cfgs['test'] = cfg('test')
    main(args,cfgs)
    
    

    