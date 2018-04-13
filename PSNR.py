# coding:utf-8
#!/usr/bin/python3

import os, cv2
#from cv2 import *
from sys import argv
from math import log10

##################################################################
def PSNR(I, K): # Require: I.shape == K.shape
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    assert(I.shape == K.shape) # assert if False
    h,w,d = K.shape; Max2 = 255**2; ee = Max2*1E-10
    MSE = cv2.norm(I, K, normType=cv2.NORM_L2)**2 / (w*h)
    PSNR = 10 * log10(Max2/(MSE+ee))
    return PSNR

def PSNR_DIR(*args, **kwargs): # args[0] = Image_DIR
    DIR = args[0]; im = os.listdir(DIR)
    im.sort(); psnr = [0]*(len(im)//2)
    for i in range(len(psnr)):
        I = cv2.imread(os.path.join(DIR, im[2*i]))
        K = cv2.imread(os.path.join(DIR, im[2*i+1]))
        psnr[i] = PSNR(I,K); print(im[2*i][:-4], psnr[i])
    return psnr,im

##################################################################
if __name__ == '__main__':
    PSNR_DIR(argv[1]) # input DIR

##################################################################
# python3 PSNR.py Tools/AI_CV_Test_1 > out.txt &