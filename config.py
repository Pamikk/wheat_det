
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.img_path = '../dataset/global-wheat/train'
        self.checkpoint='../checkpoints'
        self.sizes = [256,512,1024]
        self.sizes_w = [0.3,0.3,0.4]
        self.cls_num = 0 
        self.anchors = [[0.16164473925192566, 0.1285921540103818],[0.10578342841802468, 0.06323927300767092],[0.060309932827680185, 0.043605377293796585],[0.059888260849536704, 0.07869134364105129],[0.09009376823524509, 0.11350195752099658]]
        self.anchor_num = len(self.anchors)
        self.res = 50
        self.RGB_mean = [80.31413238,80.7378002,54.63867023]
        self.nms_threshold = 0.5
        self.dc_threshold = 0.1
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.ignore_threshold = 0.5
        self.bs = 1 
        if mode=='train':
            self.file='pre_data/train.json'
            self.bs = 8 # batch size
            #augmentation parameter
            self.rot = 0
            self.crop = 0.2
            self.valid_scale = 0
            #train_setting
            self.lr = 0.1
            self.weight_decay=5e-4
            self.min_lr = 5e-6
            self.lr_factor = 0.25
            #exp_setting
            self.save_every_k_epoch = 10

        elif mode=='val':
            self.file = 'pre_data/val.json'
            self.bs = 1
        elif mode=='trainval':
            self.file = 'pre_data/trainval.json'
            self.bs = 1
        elif mode=='test':
            self.file = 'pre_data/trainval.json'
            self.bs = 1

        
