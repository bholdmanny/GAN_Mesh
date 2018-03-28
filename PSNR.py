# coding:utf-8
#!/usr/bin/python3

import os, math
from cv2 import *
from sys import argv

##################################################################
def PSNR(I, K): # require: I.shape == K.shape
    if type(I)==str: I = imread(I)
    if type(K)==str: K = imread(K)
    Max, ee = 255, 0.0000065025 # ee = Max**2/10**10
    MSE = norm(I, K, normType=NORM_L2)**2 / (K.shape[0]*K.shape[1])
    PSNR = 10 * math.log10(Max**2/(MSE+ee))
    return PSNR

def main(*args, **kwargs): # args[0] = Image_Dir
    im = os.listdir(args[0])[:10];
    im.sort(); psnr = []
    for i in range(0,len(im),2):
        I = imread(os.path.join(args[0], im[i]))
        K = imread(os.path.join(args[0], im[i+1]))
        psnr.append(PSNR(I,K))
        print(im[i][:-4], psnr[i//2])
    return psnr,im

##################################################################
if __name__ == '__main__':
	main(argv[1])

##################################################################
python3 PSNR.py Tools/AI_CV_Test_1 > out.txt &