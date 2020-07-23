
#Train Setting
class Config:
    def __init__(self,train=True):
        #Path Setting
        self.test_img_path='../../dataset/global-wheat/test'
        self.img_path = '../../dataset/global-wheat/train'
        self.checkpoint='../../checkpoints'
        self.inp_size = (256,256)
        self.int_shape = (256,256)
        self.grid = (16,16) #inp//4
        self.cls_num = 25 # Ax5 as mentioned in YOLO,ignore cls_score
        self.res = 50
        self.RGB_mean = [80.31413238,80.7378002,54.63867023]
        if train:
            self.file='../../dataset/global-wheat/train.json'
            self.bs = 16 # batch size
            #augmentation parameter
            self.rot = 10
            #self.scale = 0.25
            self.crop = 0.2
            self.flip = True
            self.valid_scale = 0.25
            self.sigmas =[(3,3),(5,5),(9,9),(13,13)]
            #train_setting
            self.lr = 1e-4
            self.weight_decay=1e-5
            self.min_lr = 1e-7
            self.lr_factor = 0.5
            #exp_setting
            self.save_every_k_epoch = 1
            self.nms_threshold = 0.5
            self.dc_threshold = 0

        else:
            self.file = '../../dataset/global-wheat/val.json'
            self.bs = 8
        
        
