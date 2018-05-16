# coding:utf-8
#!/usr/bin/python3
import os,cv2
import numpy as np
from sys import argv

##################################################################
# Peak Signal to Noise Ratio
def PSNR(I, K, ch=3, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    assert(I.shape == K.shape) # assert if False
    I, K = I.astype(int), np.array(K,int) # avoid overflow
    if ch<2: MSE = np.mean((I-K)**2) # combine/average channels
    # OR: MSE = cv2.norm(I,K,normType=cv2.NORM_L2)**2 / I.size
    else: MSE = np.mean((I-K)**2,axis=(0,1)) # separate channels
    MAX = L**2; ee = MAX*1E-10 # normalize PSNR to 100
    return 10 * np.log10(MAX / (MSE + ee)) # PSNR

# Structural Similarity (Index Metric)
def SSIM(I, K, ch=3, k1=0.01, k2=0.03, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    assert(I.shape == K.shape) # assert if False
    if ch<2: # combine/average channels->float
        mx, sx = np.mean(I), np.var(I,ddof=1)
        my, sy = np.mean(K), np.var(K,ddof=1)
        cov = np.sum((I-mx)*(K-my)) / (I.size-1)
    else: # separate/individual/independent channels->np.array
        mx, sx = np.mean(I,axis=(0,1)), np.var(I,axis=(0,1),ddof=1)
        my, sy = np.mean(K,axis=(0,1)), np.var(K,axis=(0,1),ddof=1)
        cov = np.sum((I-mx)*(K-my),axis=(0,1)) / (I.size/I.shape[-1]-1)
    c1, c2 = (k1*L)**2, (k2*L)**2 # stabilizer, avoid divisor=0
    SSIM = (2*mx*my+c1)/(mx**2+my**2+c1) * (2*cov+c2)/(sx+sy+c2)
    return SSIM # SSIM: separate or average channels

def Batch(*args, fun=SSIM, ch=3):
    if type(args[0])!=str: args = args[0] # parse
    ff = lambda I,K: np.mean(fun(I,K,ch)) # 1-value
    if len(args)<2: # I,K in the same DIR
        IK = args[0]; im = os.listdir(IK); im.sort()
        I = [os.path.join(IK, i) for i in im[::2]]
        K = [os.path.join(IK, i) for i in im[1::2]]
    elif len(args)==2: # I,K in different DIRs
        Is, Ks = args[0], args[1]
        I = [os.path.join(Is, i) for i in os.listdir(Is)]
        K = [os.path.join(Ks, i) for i in os.listdir(Ks)]
    else: # I=Origin, K=Mesh, R=Recover
        Is, Ks, Rs = args[0], args[1], args[2]
        I = [os.path.join(Is, i) for i in os.listdir(Is)]
        K = [os.path.join(Ks, i) for i in os.listdir(Ks)]
        R = [os.path.join(Rs, i) for i in os.listdir(Rs)]
        # res = [(ff(i,r)-ff(i,k))/abs(ff(i,k)) for i,k,r in zip(I,K,R)]
    res = np.array([ff(i, k) for i, k in zip(I, K)])
    if len(args)>2: # more efficient as reuse res
        n = [ff(i,r) for i,r in zip(I,R)]; res = (n-res)/abs(res)
    return np.mean(res), res

##################################################################
if __name__ == "__main__":
    a = Batch(argv[1:]); print(a[0],"\n",a[1])
    # IK = "E:/Hua/PyCharm/Crop"; a = Batch(IK, fun=PSNR)
    # Is, Ks = "E:/Hua/PyCharm/Crop1", "E:/Hua/PyCharm/Crop2"
    # b = Batch(Is, Ks, fun=PSNR);
    # c = Batch(Is, Ks, Is, fun=PSNR)

##################################################################
# python3 PSNR.py Tools/AI_CV_Test_1 > out.txt &
