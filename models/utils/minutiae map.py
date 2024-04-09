import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def Cs(xt,yt,i,j,sigma):
    a = np.array((xt,yt))
    b = np.array((i,j))
    dist = np.sqrt(np.linalg.norm(a-b))
    return math.exp(-(dist/(2*sigma**2)))

def Co(theta_t,k,sigma):
    minutiae_angle = (2*k*math.pi)/6
    if theta_t-minutiae_angle<math.pi and theta_t-minutiae_angle>=-math.pi:
        d_theta = abs(theta_t-minutiae_angle)
    else:
        d_theta = (2*math.pi)-abs(theta_t-minutiae_angle)
    return math.exp(-(d_theta/(2*sigma**2)))

def extract_minutiae_map(img_name,minutiae_name,save_path):
    img = cv2.imread(img_name,0)
    or_img_w_shape,or_img_h_shape = img.shape[0],img.shape[1]
    img = cv2.resize(img,(30,30))
    w_ratio, h_ratio = or_img_w_shape/img.shape[0],or_img_h_shape/img.shape[1] 
    minutiae_map = np.zeros(shape=(img.shape[0],img.shape[1],6))
    minutiae_location = []
    i = 0
    with open(minutiae_name, 'r') as f:
        line = f.readlines()
        for i in line:
            value = i.split(' ')
            minutiae_location.append(value)
    del minutiae_location [0:2] 
    
    h,w,k = minutiae_map.shape[0],minutiae_map.shape[1],minutiae_map.shape[2]
    
    for i in range(h):
        for j in range(w):
            for l in range(k):
                single_value = 0
                for minutiae in minutiae_location:
                    yt = int(int(minutiae[0])/w_ratio) 
                    xt = int(int(minutiae[1])/h_ratio)
                    theta_t = float(minutiae[2]) 
                    Cs_value = Cs(xt,yt,i,j,0.6)
                    Co_value = Co(theta_t,l,0.6) 
                    single_value += Cs_value*Co_value
                minutiae_map[i,j,l] = int(single_value*255)
                
 
    resize_minutiae_map = []
    for i in range(k):
        img = cv2.resize(minutiae_map[:,:,i],(128,128))
        resize_minutiae_map.append(img)
    resize_minutiae_map = np.array(resize_minutiae_map)
    resize_minutiae_map = resize_minutiae_map.transpose((1,2,0))
    print(resize_minutiae_map.shape)
    np.save(save_path,resize_minutiae_map)    
