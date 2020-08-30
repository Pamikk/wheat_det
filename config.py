
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.img_path = '../dataset/global-wheat/train'
        self.checkpoint='../checkpoints'
        self.sizes = [256,512,1024]
        self.sizes_w = [0.3,0.3,0.4]
        self.cls_num = 0 
        self.anchors =[[53.87225938102254, 44.48974039224577], [63.62329042081948, 76.3607212070875], 
          [98.12110508830222, 53.52514425598883],  [81.62511337056661, 118.2413193947205],
        [125.43391003460205, 84.6516435986159], [158.20213028712567, 156.73488731089842]]
        self.anchor_divide = [(5,),(3,4),(0,1,2)]
        self.res = 50
        self.RGB_mean = [80.31413238,80.7378002,54.63867023]
        self.nms_threshold = 0.5
        self.dc_threshold = 0.3
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.ignore_threshold = 0.5
        self.bs = 1
        self.pre_trained_path = '../network_weights' 
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
            self.momentum = 0.9
            self.min_lr = 1e-3
            self.lr_factor = 0.1
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

        
