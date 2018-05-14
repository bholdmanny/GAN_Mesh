# coding:utf-8
#!/usr/bin/python3
from sys import argv
import os, cv2, math # from cv2 import *

##################################################################
def PSNR(I, K): # Require: I.shape == K.shape
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    assert(I.shape == K.shape) # assert if False
    height, width, depth = K.shape; MAX = 255**2
    ee = MAX*1E-10 # normalize the max of PSNR to 100
    MSE = cv2.norm(I, K, normType=cv2.NORM_L2)**2 / (height*width)
    return 10 * math.log10(MAX/(MSE+ee)) # PSNR

def Batch(*args, **kwargs):
    if type(args[0])!=str: args = args[0] # parse
    if len(args)<2: # I,K in the same DIR
        IK = args[0]; im = os.listdir(IK); im.sort()
        I = [os.path.join(IK, i) for i in im[::2]]
        K = [os.path.join(IK, i) for i in im[1::2]]
    elif len(args)==2: # I,K in different DIRs
        Is, Ks = args[0], args[1]
        I = [os.path.join(Is, i) for i in os.listdir(Is)]
        K = [os.path.join(Ks, i) for i in os.listdir(Ks)]
    elif len(args)>2: # I=Origin, K=Mesh, R=Recover
        Is, Ks, Rs = args[0], args[1], args[2]
        I = [os.path.join(Is, i) for i in os.listdir(Is)]
        K = [os.path.join(Ks, i) for i in os.listdir(Ks)]
        R = [os.path.join(Rs, i) for i in os.listdir(Rs)]
        gain = [PSNR(i, r)/PSNR(i, k)-1 for i, k, r in zip(I, K, R)]
        for i in range(len(I)): print(gain[i], K[i], R[i])
        return sum(gain)/len(gain), gain, I, K, R
    res = [PSNR(i, k) for i, k in zip(I, K)]
    for i in range(len(I)): print(res[i], I[i], K[i])
    return res, I, K

##################################################################
if __name__ == "__main__":
    Batch(argv[1:]) # input_dir
    #IK = "E:/Hua/PyCharm/Crop"; a = Batch(IK)
    #Is, Ks = "E:/Hua/PyCharm/Crop1", "E:/Hua/PyCharm/Crop2"
    #b = Batch(Is, Ks); #c = Batch(Is, Ks, Is)

##################################################################
# python3 PSNR.py Tools/AI_CV_Test_1 > out.txt &
