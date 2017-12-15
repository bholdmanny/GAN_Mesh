#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from pylab import * #import matplotlib & numpy

# Create Normalized Curve:
def Curve(tp, p, t): # normalized curve
    s1 = p[2] * np.sin((p[0]*t+p[1])*np.pi);
    s2 = p[5] * np.sin((p[3]*t+p[4])*np.pi); no = 0; # noise
    if type(tp) != str: # add extra noise when tp<0 for "sps"
        no, tp = int(tp<0)*(np.random.rand(len(t))-0.5)/5, "sps";
    if "sps" in tp: return (s1 + s2)/(abs(p[2]) + abs(p[5])) + no
    if "sms" in tp: return (s1 * s2)/(abs(p[2]) * abs(p[5])) + no
    if len(p)>=9: s3 = p[8] * np.sin((p[6]*t+p[7])*np.pi)
    if "sss" in tp: return (s1 + s2 + s3)/(abs(p[2])+abs(p[5])+abs(p[8])) + no

# Get Random [linewidth, alpha] Pairs:
def LwAl(n=1, tp=1, dx=180): # random [linewidth, alpha] pair
    wa = np.random.rand(2*n); f = int(abs(tp)<5)*(dx/180-1)+1;
    wa[::2] = [round(f*(i+2),1) for i in wa[::2]] # linewidth
    wa[1::2] = [round(0.4*i+0.4,2) for i in wa[1::2]] # alpha
    if abs(tp) in (4,8): wa[::2] = round(1.2*f,1); # for tp=4
    return wa # type: np.array

# Rotate or Affine the Curve:
def RoAf(t, y, ra=0, af=None): # rotate or affine the curve
    if type(ra) != np.ndarray: # rotational angle -> matrix
        ra *= np.pi; ra = np.array([[cos(ra),-sin(ra)],[sin(ra),cos(ra)]])
    if type(af) == np.ndarray:   ra = ra.dot(af); # affine & rotate
    y = ra.dot(np.array([t,y])); # rotate/affine the curve
    return y[0,:], y[1,:]

# Draw Curve with Annotation:
def DrawCu(tp, p=None, xi=0, dx=20, yo=0, A=1, ra=0, af=0, wa=[]): # draw curve
    if p==None or len(p)<6: # get curve parameters
        p = [round(2*i,2) for i in np.random.rand(9)]; p[2]=p[5]=p[8]=1
    t = np.linspace(xi, xi+2*dx, round(2*dx*(np.random.rand()+1)), endpoint=True);
    y = A * (Curve(tp,p,t) + yo); # vertically scale + translate
    t,y = RoAf(t-(xi+dx), y, ra, af); # horizontally adjust -> rotate/affine

    if len(wa)<2: wa = LwAl(1,tp,dx); # get [linewidth,alpha] pair
    an = str(tp)+": "+", ".join([str(i) for i in p])+"->"+", ".join([str(i) for i in wa])
    plot(t, y, color="k", lw=wa[0], alpha=wa[-1], label=an);
    return p, wa


# Extract sps Cell Parameters:
def Paras(tp, dx, A, f): # Extract sps Cell Parameters
    tp = abs(tp); # add noise when tp<0
    if tp>4: tp -= 4; dx = 180; # keep sps Cell size
    yf = dx/180; # Cell Ratio: Amplitude scale as image width
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
def DrawCel(dx, yi=0, tp=1, ra=0, wa=[], A=42, f=12): # draw sps cell
    xi = round(dx*np.random.rand(),1);
    dy = round(0.2+(np.random.rand()-0.5)/10, 3);
    A,p1,f = Paras(tp,dx,A,f); # get sps Cell Parameters
    p2 = p1.copy(); p2[::-3] = [-i for i in p2[::-3]]
    
    t = np.linspace(xi, xi+2*dx, round(2*dx*(np.random.rand()+1)), endpoint=True);
    y1 = A * (Curve(tp,p1,t) + (yi+dy)); # vertically scale + trans
    y2 = A * (Curve(tp,p2,t) + (yi-dy)); # vertically scale + trans
    t -= xi+dx; t1,y1 = RoAf(t,y1,ra,f); t2,y2 = RoAf(t,y2,ra,f) # rotate/affine
    plot(t1, y1, color="b", lw=wa[0], alpha=wa[-1])
    plot(t2, y2, color="b", lw=wa[2], alpha=wa[-1])

# Draw Reticulate Pattern Cell(sps):
def DrawCell(dx, yi=0, tp=1, ra=0, wa=[], A=42, f=12): # draw sps cell
    xi = round(dx*np.random.rand(),1);
    dy = round(0.2+(np.random.rand()-0.5)/10, 3);
    A,p1,f = Paras(tp,dx,A,f); # get sps Cell Parameters
    p2 = p1.copy(); p2[::-3] = [-i for i in p2[::-3]]
    DrawCu(tp, p1, xi, dx, yi+dy, A, ra, f, wa=wa[:])
    DrawCu(tp, p2, xi, dx, yi-dy, A, ra, f, wa=wa[2:])

# Add Reticulate Net to Image:
def Add2Im(im, tp=None, ro=None, wa=None, gap=1.65, fun=DrawCell): # add to image
    if type(im)==str: im = imread(im) # load image
    y,x,n = im.shape; n = y//20; # get width & height
    if tp==None: tp = np.random.randint(-8,9); print(tp)
    if ro==None: ro = 2*np.random.rand()-1; # random rotate
    if wa==None or len(wa)<4: wa = LwAl(2,tp,x); # [lw,alpha]
    ofs = round(1.5*np.random.rand(), 2);
    gap = round(gap+(np.random.rand()-0.3)/10, 2);
    for i in range(2*n): fun(x, gap*(i-n)+ofs, tp, ro, wa=wa)
    subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    imshow(im, extent=(-x/2,x/2,-y/2,y/2)); axis("off"); xticks([]); yticks([]);

def Save2Im(im, tp=None, ro=None, wa=None, gap=1.6, out=None):
    if type(im)==str: im = imread(im) # load image
    dpi = 72; y,x,c = im.shape; c = (x/dpi, y/dpi);
    figure(figsize=c, dpi=dpi); Add2Im(im, tp, ro, wa, gap);
    if out != None: savefig(out, dpi=dpi); close("all");
