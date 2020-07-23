import cv2 
import numpy as np
import torch
def gen_rot_mat(ang,h,w):
    ''' generate rotation matrix with input in degrees,pad to keep all img'''
    ang = np.pi * ang/180 # map to [-pi,pi]
    mat = np.zeros([3,3])
    mat[0,0] = np.cos(ang)
    mat[0,1] = -np.sin(ang)
    mat[1,0] = np.sin(ang)
    mat[1,1] = np.cos(ang)
    mat[2,2] = 1   
    trans = np.eye(3)
    trans[0,2] = -w/2
    trans[1,2] = -h/2
    mat = np.dot(mat,trans)
    trans[0,2] = w/2
    trans[1,2] = h/2
    mat = np.dot(trans,mat)
    return mat
def brightness_scale(src,vs):
    img = cv2.cvtColor(src,cv2.COLOR_RGB2HSV).astype(np.float)
    img[:,:,2] *= (1+vs)
    img[:,:,2][img[:,:,2]>255] = 255
    return img
def augment(src,ang,crop,vs,flip=False):
    #flip
    if flip:
        dst = src[:,::-1,:]
    else:
        dst = src
    #rotation
    h,w,c = dst.shape
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, 1.0)
    dst = cv2.warpAffine(dst,mat,(w,h))
    #crop
    dh,dw = crop
    if dw>=0:
        dst = dst[:,dw:,:]
        padx =0
    else:
        padx = abs(dw)

    if dh>=0:
        dst = dst[dh:,:,:]
        pady = 0
    else:
        pady = abs(dh)
    dst = cv2.copyMakeBorder(dst,padx,0,pady,0,cv2.BORDER_CONSTANT,0)
  
    return brightness_scale(dst,vs),mat
def resize(src,tsize):
    dst = cv2.resize(src,(tsize[1],tsize[0]),interpolation=cv2.INTER_CUBIC)
    return dst


    

def color_normalize(img,mean):
    img = img.astype(np.float)
    if img.max()>1:
        img /= 255
    img -= np.array(mean)/255
    return img



