# coding:utf-8
#!/usr/bin/python3
import os, cv2, matplotlib
matplotlib.use('Agg') # before import matplotlib.pyplot or pylab!
from pylab import * # import matplotlib & numpy

# tp = [Type=0-3, Fix/Scale=0/1, Noise Scale=0-1]
#####################################################################
# Create Normalized Curve:
def Curve(tp, p, t): # normalized curve
    if type(tp)!=str: tp = "sps" # tp=tp[0]
    s1 = p[2] * np.sin((p[0]*t+p[1])*np.pi)
    s2 = p[5] * np.sin((p[3]*t+p[4])*np.pi)
    if "sps" in tp: return (s1 + s2)/(abs(p[2]) + abs(p[5]))
    if "sms" in tp: return (s1 * s2)/(abs(p[2]) * abs(p[5]))
    if len(p)>=9: s3 = p[8] * np.sin((p[6]*t+p[7])*np.pi)
    if "sss" in tp: return (s1 + s2 + s3)/(abs(p[2])+abs(p[5])+abs(p[8]))

# Get Random [linewidth, alpha] Pairs:
def LwAl(tp, n=1, dx=180): # random [linewidth, alpha] pair
    wa = np.random.rand(2*n); f = tp[1]*(dx/180-1)+1 # scale ratio
    wa[::2] = [round(f*(1.1*i+2),2) for i in wa[::2]] # linewidth
    wa[1::2] = [round(0.4*i+0.2,2) for i in wa[1::2]] # alpha
    if tp[0]==3: wa[::2] = round(f,2); # linewidth for tp=3
    return wa # type: np.array

# Rotate or Affine the Curve:
def RoAf(t, y, ra=0, af=None): # rotate or affine the curve
    if type(ra)!=np.ndarray: # rotational angle -> matrix
        ra *= np.pi; ra = np.array([[cos(ra),-sin(ra)],[sin(ra),cos(ra)]])
    if type(af)==np.ndarray: ra = ra.dot(af); # affine & rotate
    y = ra.dot(np.array([t,y])) # rotate/affine the curve
    return y[0,:], y[1,:] # t'=y[0,:], y'=y[1,:]

# Draw Curve with Annotation:
def DrawCu(tp, p=None, xi=0, dx=20, yo=0, A=1, ra=0, af=0, wa=[]): # draw curve
    if type(tp)!=list and type(tp)!=tuple: tp = [tp,0,0] # default
    tp, fs, no = tp # tp[0]=Type, tp[1]=Fix/Scale, tp[2]=Noise Scale
    if p==None or len(p)<6: # default: random curve parameters
        p = [round(2*i,2) for i in np.random.rand(9)]; p[2]=p[5]=p[8]=1
    t = np.linspace(xi, xi+2*dx, round(2*dx*(np.random.rand()+1)), endpoint=True)
    no = no/5 * (1+(tp==3)) * (np.random.rand(len(t))-0.5) # noise
    y = A * (Curve(tp,p,t) + yo + no) # vertically scale + translate
    t,y = RoAf(t-(xi+dx), y, ra, af) # horizontally adjust -> rotate/affine
    if len(wa)<2: wa = LwAl([tp,fs], 1, dx); # get [linewidth,alpha] pair
    an = str(tp)+": "+", ".join([str(i) for i in p])+"->"+", ".join([str(i) for i in wa])
    plot(t, y, color="k", lw=wa[0], alpha=wa[-1], label=an)
    return t, y, wa, p

#####################################################################
# Extract sps Cell Parameters:
def Paras(tp, dx, A, f): # Extract sps Cell Parameters
    tp, yf = tp[0], tp[1]*(dx/180-1)+1 # Cell scale ratio
    if tp==0: # Reticulate Pattern Type0
        A = 42*yf; f = 12/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
    elif tp==1: # Reticulate Pattern Type1
        A = 30*yf; f = 8/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.75]
    elif tp==2: # Reticulate Pattern Type2
        A = 55*yf; f = 8/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1,-0.5],[-0.15,1]]); # Affine Matrix
    elif tp==3: # Reticulate Pattern Type3
        A = 10*yf; f = 7.5/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1.15,1.1],[-0.45,0.7]]); # Affine Matrix
    else: A *= yf; f /= dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8-tp%2/10]
    return A, p, f

# Draw Reticulate Pattern Cell(sps):
def DrawCell(tp, dx, yi=0, ra=0, wa=[], A=42, f=12): # draw sps cell
    xi = round(dx*np.random.rand(), 1)
    dy = round(0.2+(np.random.rand()-0.5)/10, 3)
    A,p1,f = Paras(tp,dx,A,f); # get sps Cell Parameters
    p2 = p1.copy(); p2[::-3] = [-i for i in p2[::-3]]
    # DrawCu set alpha=wa[-1], thus Curve Cells have same alpha:
    t1,y1,w1,p1 = DrawCu(tp, p1, xi, dx, yi+dy, A, ra, f, wa=wa[:])
    t2,y2,w2,p2 = DrawCu(tp, p2, xi, dx, yi-dy, A, ra, f, wa=wa[2:])
    return [t1,y1, t2,y2]

# Save Mesh Image with Mask:
def SaveIm(im, out, tp, qt=20, ms=None, ro=None, wa=None, gap=1.6):
    if type(im)==str: im = imread(im) # load image
    n = im.shape; y,x = n[0],n[1]; n = y//20 # width & height
    if ro==None: ro = 2*np.random.rand()-1 # randomly rotate
    if wa==None or len(wa)<4: wa = LwAl(tp,2,x) # [lw,alpha]
    ofs = round(1.5*np.random.rand(), 2)
    gap = round(gap+(np.random.rand()-0.3)/10, 2); net = []
    dpi = 72; figure(figsize=(x/dpi, y/dpi), dpi=dpi); axis("off")
    subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    for i in range(2*n): net += DrawCell(tp, x, gap*(i-n)+ofs, ro, wa=wa)
    if ms != None: # output mask image
        xlim(-x/2,x/2); ylim(-y/2,y/2); ms = out[:-4]+"_m.png"
        savefig(ms, facecolor="w", dpi=dpi); tmp = cv2.imread(ms, 0)
        gap,tmp = cv2.threshold(tmp, 250, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(ms, tmp, [int(cv2.IMWRITE_PXM_BINARY), 1])
    imshow(im, extent=(-x/2,x/2,-y/2,y/2)); savefig(out, dpi=dpi)
    if type(qt)!=int: qt = np.random.randint(qt[0],qt[1]) # quality
    cv2.imwrite(out, cv2.imread(out), [cv2.IMWRITE_JPEG_QUALITY, qt])
    close("all"); return net

#####################################################################
# Crop Randomly OR Resize Image:
def Crop(im, size, op="cut"): # size=(height,width), op=crop/resize
    if type(im)==str: im = cv2.imread(im) # OR: pylab.imread(im)
    if type(size)!=np.ndarray: size = np.array(size) # (height,width)
    imsz = np.array([im.shape[0],im.shape[1]]) # (height,width,depth)
    if ("c" in op) and (imsz>=size).all(): # True only if all elements are Ture
        xy = np.array([np.random.randint(i) for i in imsz-size+1]); br = xy+size
        return im[xy[0]:br[0], xy[1]:br[1]] # randomly crop/cut
    else: return cv2.resize(im, (size[1],size[0]), interpolation=cv2.INTER_AREA)

# Batch Adding Meshes to Images and Saving:
def BatchSave(src, tp, qt=20, num=None, ms=None):
    tp, fs, no = tp # parse: Type,Fix/Scale,Noise
    if type(tp)==int: tp = [tp] # tp=range/list/tuple
    N = len(tp); SZ = (220,178) # default (height,width)
    if num==None or type(num)!=int: num = N # default=N
    id = [i%N for i in range(num)] # default index of tp
    scr = os.path.abspath(src); ss = os.path.split(src)[-1]
    dst = ss + "_".join([str(i) for i in ["",fs,no]]) # target
    out = lambda im,k="": im.replace(ss,dst)[:-4]+"_"+str(k)+".jpg"
    for path,sub,file in os.walk(src): # traverse src
        os.mkdir(path.replace(ss,dst)); os.chdir(path) # cd path
        for im in file: # loop images in path
            im = os.path.join(path, im) # uesd in out()
            if num<N: id = np.random.choice(N, num, False) # non-repetitive
            for i in range(num): SaveIm(im, out(im,i), (tp[id[i]],fs,no), qt, ms)
            # Crop/Resize Images->Save->Add Meshes to res/out(im):
            #res = Crop(imread(im), SZ, "cut") # try: cv2.imread or "resize"
            #cv2.imwrite(out(im), res[:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            #for i in range(num): SaveIm(res, out(im,i), (tp[id[i]],fs,no), qt, ms)
    os.chdir(os.path.join(src,"..")) # cd parent directory

#####################################################################
if __name__ == "__main__":
    src = "E:/Hua/PyCharm/Test"
    #src = "/home/hua.fu/CASIA-WebFace/"
    tp = [range(4), 1, 0.3]
    BatchSave(src, tp, qt=95, num=3, ms=None)
