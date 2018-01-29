#!/usr/bin/python3
import os, cv2, matplotlib
matplotlib.use('Agg') # before import matplotlib.pyplot or pylab!
from pylab import * # import matplotlib & numpy

# tp = (Type=1-4, Fix/Scale=0/1, Noise Scale=0-1)
#####################################################################
# Create Normalized Curve:
def Curve(tp, p, t): # normalized curve
    if type(tp)!=str: tp = "sps" # tp=tp[0]
    s1 = p[2] * np.sin((p[0]*t+p[1])*np.pi);
    s2 = p[5] * np.sin((p[3]*t+p[4])*np.pi);
    if "sps" in tp: return (s1 + s2)/(abs(p[2]) + abs(p[5]))
    if "sms" in tp: return (s1 * s2)/(abs(p[2]) * abs(p[5]))
    if len(p)>=9: s3 = p[8] * np.sin((p[6]*t+p[7])*np.pi)
    if "sss" in tp: return (s1 + s2 + s3)/(abs(p[2])+abs(p[5])+abs(p[8]))

# Get Random [linewidth, alpha] Pairs:
def LwAl(tp, n=1, dx=180): # random [linewidth, alpha] pair
    wa = np.random.rand(2*n); f = 1+tp[1]*(dx/180-1); # scale ratio
    wa[::2] = [round(f*(i+2),1) for i in wa[::2]] # linewidth
    wa[1::2] = [round(0.4*i+0.4,2) for i in wa[1::2]] # alpha
    if tp[0]==4: wa[::2] = round(1.2*f,1); # only for tp=4
    return wa # type: np.array

# Rotate or Affine the Curve:
def RoAf(t, y, ra=0, af=None): # rotate or affine the curve
    if type(ra) != np.ndarray: # rotational angle -> matrix
        ra *= np.pi; ra = np.array([[cos(ra),-sin(ra)],[sin(ra),cos(ra)]])
    if type(af) == np.ndarray:   ra = ra.dot(af); # affine & rotate
    y = ra.dot(np.array([t,y])); # rotate/affine the curve
    return y[0,:], y[1,:] # t'=y[0,:], y'=y[1,:]

# Draw Curve with Annotation:
def DrawCu(tp, p=None, xi=0, dx=20, yo=0, A=1, ra=0, af=0, wa=[]): # draw curve
    if type(tp) != (tuple or list): tp = (tp,0,0) # set default tp
    if p==None or len(p)<6: # set random curve parameters
        p = [round(2*i,2) for i in np.random.rand(9)]; p[2]=p[5]=p[8]=1
    t = np.linspace(xi, xi+2*dx, round(2*dx*(np.random.rand()+1)), endpoint=True);
    no = tp[2]*(np.random.rand(len(t))-0.5)/5; # noise, tp[2]=ratio
    y = A * (Curve(tp[0],p,t) + yo + no); # vertically scale + translate
    t,y = RoAf(t-(xi+dx), y, ra, af); # horizontally adjust -> rotate/affine

    if len(wa)<2: wa = LwAl(tp,1,dx); # get [linewidth,alpha] pair
    an = str(tp[0])+": "+", ".join([str(i) for i in p])+"->"+", ".join([str(i) for i in wa])
    plot(t, y, color="k", lw=wa[0], alpha=wa[-1], label=an);
    return t, y, wa, p

#####################################################################
# Extract sps Cell Parameters:
def Paras(tp, dx, A, f): # Extract sps Cell Parameters
    yf,tp = 1+tp[1]*(dx/180-1),tp[0]; # Cell scale ratio
    if tp==1: # Reticulate Pattern Type1
        A = 42*yf; f = 12/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
    elif tp==2: # Reticulate Pattern Type2
        A = 30*yf; f = 8/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.75]
    elif tp==3: # Reticulate Pattern Type3
        A = 55*yf; f = 8/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1,-0.5],[-0.15,1]]); # Affine Matrix for Type3
    elif tp==4: # Reticulate Pattern Type4
        A = 10*yf; f = 7.5/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1.15,1.1],[-0.45,0.7]]); # Affine Matrix for Type4
    else: A *= yf; f /= dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8-tp%0.2]
    return A, p, f

# Draw Reticulate Pattern Cell(sps):
def DrawCel(tp, dx, yi=0, ra=0, wa=[], A=42, f=12): # draw sps cell
    xi = round(dx*np.random.rand(),1);
    dy = round(0.2+(np.random.rand()-0.5)/10, 3);
    A,p1,f = Paras(tp,dx,A,f); # get sps Cell Parameters
    p2 = p1.copy(); p2[::-3] = [-i for i in p2[::-3]]
    
    t = np.linspace(xi, xi+2*dx, round(2*dx*(np.random.rand()+1)), endpoint=True);
    no1,no2 = tp[2]*(np.random.rand(len(t))-0.5)/5, tp[2]*(np.random.rand(len(t))-0.5)/5;
    y1 = A * (Curve(tp[0],p1,t) + (yi+dy) + no1); # vertically scale + trans
    y2 = A * (Curve(tp[0],p2,t) + (yi-dy) + no2); # vertically scale + trans
    t -= xi+dx; t1,y1 = RoAf(t,y1,ra,f); t2,y2 = RoAf(t,y2,ra,f) # rotate/affine
    plot(t1, y1, color="b", lw=wa[0], alpha=wa[-1])
    plot(t2, y2, color="b", lw=wa[2], alpha=wa[-1])
    return [t1,y1, t2,y2]

# Draw Reticulate Pattern Cell(sps):
def DrawCell(tp, dx, yi=0, ra=0, wa=[], A=42, f=12): # draw sps cell
    xi = round(dx*np.random.rand(),1);
    dy = round(0.2+(np.random.rand()-0.5)/10, 3);
    A,p1,f = Paras(tp,dx,A,f); # get sps Cell Parameters
    p2 = p1.copy(); p2[::-3] = [-i for i in p2[::-3]]
    t1,y1,w1,p1 = DrawCu(tp, p1, xi, dx, yi+dy, A, ra, f, wa=wa[:])
    t2,y2,w2,p2 = DrawCu(tp, p2, xi, dx, yi-dy, A, ra, f, wa=wa[2:])
    return [t1,y1, t2,y2]

# Add Reticulate Net to Image:
def Add2Im(im, tp, ro=None, wa=None, gap=1.6, fun=DrawCell): # add to image
    if type(im)==str: im = imread(im) # load image
    n = im.shape; y,x = n[0],n[1]; n = y//20; # width & height
    if ro==None: ro = 2*np.random.rand()-1; # randomly rotate
    if wa==None or len(wa)<4: wa = LwAl(tp,2,x); # [lw,alpha]
    ofs = round(1.5*np.random.rand(), 2);
    gap = round(gap+(np.random.rand()-0.3)/10, 2); net = [];
    for i in range(2*n): net += fun(tp, x, gap*(i-n)+ofs, ro, wa=wa)
    subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    imshow(im, extent=(-x/2,x/2,-y/2,y/2)); axis("off"); xticks([]); yticks([]);
    return net

# Save Image with Reticulate Net:
def Save2Im(im, out, tp, ro=None, wa=None, gap=1.6, ms=None):
    if type(im)==str: im = imread(im) # load image
    dpi = 72; n = im.shape; y,x = n[0],n[1]; # width & height
    figure(figsize=(x/dpi, y/dpi), dpi=dpi); # set figsize
    net = Add2Im(im, tp, ro, wa, gap); savefig(out, dpi=dpi);
    if ms != None: # output mask image
        if type(ms) != str: ms = "./Test/blank.png";
        imshow(imread(ms), extent=(-x/2,x/2,-y/2,y/2));
        out = out[:-4]+"_m.png"; savefig(out); im = cv2.imread(out, 0);
        ret,im = cv2.threshold(im, 250, 255, cv2.THRESH_BINARY_INV);
        cv2.imwrite(out, im, [int(cv2.IMWRITE_PXM_BINARY),1]);
    close("all"); return net

# Save Image with Reticulate Net:
def SaveIm(im, out, tp, ro=None, wa=None, gap=1.6, ms=None):
    if type(im)==str: im = imread(im) # load image
    n = im.shape; y,x = n[0],n[1]; n = y//20; # width & height
    if ro==None: ro = 2*np.random.rand()-1; # randomly rotate
    if wa==None or len(wa)<4: wa = LwAl(tp,2,x); # [lw,alpha]
    ofs = round(1.5*np.random.rand(), 2);
    gap = round(gap+(np.random.rand()-0.3)/10, 2); net = [];
    dpi = 72; figure(figsize=(x/dpi, y/dpi), dpi=dpi); axis("off");
    subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    for i in range(2*n): net += DrawCell(tp, x, gap*(i-n)+ofs, ro, wa=wa)
    
    if ms != None: # output mask image
        xlim(-x/2,x/2); ylim(-y/2,y/2); ms = out[:-4]+"_m.png";
        savefig(ms, facecolor="w", dpi=dpi); tp = cv2.imread(ms, 0);
        gap,tp = cv2.threshold(tp, 250, 255, cv2.THRESH_BINARY_INV);
        cv2.imwrite(ms, tp, [int(cv2.IMWRITE_PXM_BINARY),1]);
    imshow(im, extent=(-x/2,x/2,-y/2,y/2)); savefig(out, dpi=dpi);
    close("all"); return net

# Batch to Save Images with Reticulate Net:
def Batch_SaveIm(org, tp, num=None, ms=None):
    out = lambda name,k: name[:-4]+"_"+str(k)+".jpg";
    if org[-1] != "/": org += "/"; # original image path
    dst = org.split("/"); dst[-2] += "2"; dst = "/".join(dst)
    if not os.path.exists(dst): os.mkdir(dst); # dst dir
    for i in os.listdir(org)[:1]: # loop subdirs of org dir
        if not os.path.exists(dst+i): os.mkdir(dst+i); # dst subdir
        os.chdir(dst+i); outlist = os.listdir(dst+i); # pwd = dst+i
        for im in os.listdir(org+i): # loop images in org subdir
            if num==None or type(num)!=int: tps = tp[0] # loop all types
            else: tps = np.random.randint(tp[0].start, tp[0].stop, num);
            tps = [k for k in set(tps) if out(im,k) not in outlist]
            for k in tps: # loop for types
                k = (k, tp[1], (1+(k==4))*tp[2]);
                SaveIm(org+i+"/"+im, out(im,k[0]), tp=k, ms=ms);

#####################################################################
org = "E:/FacePic/WebFace";
tp = [range(1,5), 0, 0.4];
Batch_SaveIm(org, tp, num=1, ms=None);
