import torch.nn as nn
import torch
import numpy as np
from utils import iou_wo_center,iou_wt_center
Anchors = [(0.1066,0.17),(0.1604,0.093),(0.0571,0.0554),(0.0748,0.09763),(0.1003,0.0618)]
Anchors = [(0.11,0.11),(0.06,0.06),(0.12,0.06),(0.16,0.15),(0.07,0.095)]
#directly get by normalized anchor size, not accute according to YOLOv2, need change distance metric
__all__=["MyLoss","MyLoss_v2"]  

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

class MyLoss(nn.Module):
    def __init__(self,):
        super(MyLoss,self).__init__()
        self.bbox_loss = YOLOLoss()
        self.device='cuda'
    def forward(self,out,int_out=None,labels=None,heatmaps=None,infer=False):
        if infer:
            bboxes = self.bbox_loss(out,None,infer)
            return bboxes
        else:
            bbox_loss = self.bbox_loss(out,labels)
        self.device = out.device
        heatmap_loss = torch.tensor(0,dtype=out.dtype,device=out.device)
        for i in range(len(heatmaps)):
            heatmap_loss +=mse_loss(int_out[i],heatmaps[i].to(out.device))
        
        return bbox_loss,heatmap_loss/4

class MyLoss_v2(nn.Module):
    def __init__(self,):
        super(MyLoss_v2,self).__init__()
        self.bbox_loss = YOLOLoss()
        self.device='cuda'
    def forward(self,out,labels=None,infer=False):
        if infer:
            bboxes = self.bbox_loss(out,None,infer)
            return bboxes
        else:
            bbox_loss = self.bbox_loss(out,labels)
        self.device = out.device
        
        return bbox_loss

class YOLOLoss(nn.Module):
    #Reference:https://github.com/eriklindernoren/PyTorch-YOLOv3 YOLO layer in model
    def __init__(self,th=0.5,noobject_scale=100,coarse= False):
        super(YOLOLoss,self).__init__()
        self.anchors = np.array(Anchors)
        self.num_anchor = len(Anchors)
        self.object_scale = 1
        self.noobject_scale = noobject_scale
        #self.cls_num = 1#only wheat in this case,so ignore
        self.grid_size = 0
        self.ignore_thres = th
        self.device= 'cuda'
        self.target_num = 120
        self.coarse = coarse
    def get_mesh_grid(self,grid_size):
        self.grid_size = grid_size
        x = np.arange(0,grid_size,1)
        y = np.arange(0,grid_size,1)
        self.grid_x,self.grid_y = np.meshgrid(x,y)
        self.grid_x = torch.tensor(self.grid_x).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=self.device)
        self.grid_y = torch.tensor(self.grid_y).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=self.device)
    def get_anchors(self,grid_size):
        self.scaled_anchors = torch.tensor(self.anchors*grid_size,dtype=torch.float,device=self.device)
        self.anchor_w = torch.tensor(self.scaled_anchors[:,0]).view(1,self.num_anchor,1,1).to(dtype=torch.float,device=self.device)
        self.anchor_h = torch.tensor(self.scaled_anchors[:,1]).view(1,self.num_anchor,1,1).to(dtype=torch.float,device=self.device)
    def build_target_coarse(self,pd,labels):
        pass
    def build_target(self,pd,labels):
        self.device ='cuda' if pd.is_cuda else 'cpu'
        nB,nA,nG,_,_ = pd.shape
        anchors = self.scaled_anchors
        threshold = self.ignore_thres
        #create output tensors
        obj_mask = torch.zeros(nB,nA,nG,nG,dtype=torch.bool,device=self.device)
        noobj_mask = torch.ones(nB,nA,nG,nG,dtype=torch.bool,device=self.device)
        #cls_mask = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device)
        scores = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) #iou score
        tx = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device)  
        ty = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        tw = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        th = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        #tcls = torch.zeros(nB,nA,nG,nG,self.cls_num,dtype=torch.float,device=self.device) 
        if len(labels)==0:
            return scores,obj_mask,noobj_mask,tx,ty,tw,th,obj_mask.float()
        #convert target
        gts = labels[:,1:]*nG
        gxs = gts[:,0]
        gys = gts[:,1]
        gws = gts[:,2]
        ghs = gts[:,3]

        #get anchor with best iou
        ious = torch.stack([iou_wo_center(anchor[0],anchor[1],gws,ghs) for anchor in anchors])
        _,best_n = ious.max(dim=0)
        idx = labels[:,0].long()
        gi,gj = gxs.long(),gys.long()

        #best overlap, objecteness score maked as 1
        obj_mask[idx,best_n,gj,gi] = 1
        tconf = obj_mask.float()
        noobj_mask[idx,best_n,gj,gi] = 0

        #ignore the one not the best but over threshold 
        #not marked as negative
        for i,iou in enumerate(ious.t()):
            noobj_mask[idx[i],iou>threshold,gj[i],gi[i]]=0
        #get target labels
        tx[idx,best_n,gj,gi] = gxs - gxs.floor()
        ty[idx,best_n,gj,gi] = gys - gys.floor()
        tw[idx,best_n,gj,gi] = torch.log(gws/anchors[best_n][:,0])
        th[idx,best_n,gj,gi] = torch.log(ghs/anchors[best_n][:,1])
        # one-shot encoding of label
        #tcls[b,best_n,gj,gi,0]
        scores[idx,best_n,gj,gi] = iou_wt_center(pd[idx,best_n,gj,gi],gts)

        return scores,obj_mask,noobj_mask,tx,ty,tw,th,tconf
    
    def forward(self,out,gts,infer=False):
        nb,_,nh,nw = out.shape
        self.device ='cuda' if out.is_cuda else 'cpu'
        grid_size = nh
        pred = out.view(nb,self.num_anchor,5,nh,nw).permute(0,1,3,4,2).contiguous()
        #reshape to nB,nA,nH,nW,bboxes
        xs = torch.sigmoid(pred[:,:,:,:,0])#dxs
        ys = torch.sigmoid(pred[:,:,:,:,1])#dys
        ws = pred[:,:,:,:,2]
        hs = pred[:,:,:,:,3]
        conf = torch.sigmoid(pred[:,:,:,:,4]).unsqueeze(dim=-1)#Object score
        

        if grid_size != self.grid_size:
            self.get_mesh_grid(grid_size)
            self.get_anchors(grid_size)

        pd_bboxes = torch.zeros_like(pred[:,:,:,:,:4],dtype=torch.float,device=self.device)
        pd_bboxes[:,:,:,:,0] = (xs + self.grid_x)
        pd_bboxes[:,:,:,:,1] = (ys + self.grid_y)
        pd_bboxes[:,:,:,:,2] = torch.exp(ws) * self.anchor_w
        pd_bboxes[:,:,:,:,3] = torch.exp(hs) * self.anchor_h

        
        if infer:
            return torch.cat((pd_bboxes/grid_size,conf),axis=-1)
        else:
            scores,obj_mask,noobj_mask,tx,ty,tw,th,tconf = self.build_target(pd_bboxes,gts)

        loss_x = mse_loss(xs[obj_mask],tx[obj_mask])
        loss_y = mse_loss(ys[obj_mask],ty[obj_mask])
        loss_w = mse_loss(ws[obj_mask],tw[obj_mask])
        loss_h = mse_loss(hs[obj_mask],th[obj_mask])

        loss_obj = bce_loss(conf[obj_mask],tconf[obj_mask])
        loss_noobj = bce_loss(conf[noobj_mask],tconf[noobj_mask])
        loss_conf = self.object_scale*loss_obj+self.noobject_scale*loss_noobj

        #loss_cls = bce_loss(cls_conf[obj_mask],tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf

        return total_loss
        






        







        