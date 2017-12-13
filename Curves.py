#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from pylab import * #import matplotlib & numpy

# Create Normalized Curve:
def Curve(typ, p, t): # normalized curve
    s1 = p[2] * np.sin((p[0]*t+p[1])*np.pi);
    s2 = p[5] * np.sin((p[3]*t+p[4])*np.pi);
    if typ=="sms": return (s1 * s2)/(abs(p[2]) * abs(p[5]))
    if typ=="sps": return (s1 + s2)/(abs(p[2]) + abs(p[5]))
    if len(p)>=9: s3 = p[8] * np.sin((p[6]*t+p[7])*np.pi)
    if typ=="sps3": return (s1 + s2 + s3)/(abs(p[2])+abs(p[5])+abs(p[8]))

# Get Random [linewidth, alpha] Pairs:
def LwAl(n=1): # random [linewidth, alpha]
    wa = np.random.rand(2*n); # n paris: [linewidth, alpha]
    wa[::2] = [round(3.5*i+0.5,1) for i in wa[::2]] # linewidth
    wa[1::2] = [round(0.6*i+0.2,2) for i in wa[1::2]] # alpha
    return wa # type: np.array

# Rotate or Affine the Curve:
def RoAf(t, y, ra=0, af=0): # rotate or affine the curve
    if type(ra) != np.ndarray: # rotational angle -> matrix
        ra *= np.pi; ra = np.array([[cos(ra),-sin(ra)],[sin(ra),cos(ra)]])
    if type(af) == np.ndarray:   ra = ra.dot(af); # affine & rotate
    y = ra.dot(np.array([t,y])); # rotate/affine the curve
    return y[0,:], y[1,:]

# Draw Curve with Annotation:
def DrawCu(typ, p=[], xi=0, dx=40, yo=0, A=1, ra=0, af=0, wa=[]): # draw curve
    if len(p)<6: p = [round(2*i,2) for i in np.random.rand(9)]; p[2]=p[5]=p[8]=1
    t = np.linspace(xi, xi+dx, 30*dx, endpoint=True); # horizontally trans
    y = A * (Curve(typ, p, t) + yo); # vertically scale + translate
    t,y = RoAf(t-(xi+dx/2), y, ra, af); # horizontally adjust -> rotate/affine
    
    if len(wa)<2: wa = LwAl(); # get [linewidth, alpha]
    an = typ+": "+", ".join([str(i) for i in p])+"->"+", ".join([str(i) for i in wa])
    plot(t, y, color="k", lw=wa[0], alpha=wa[1], label=an);
    return p, wa


# Default sps Cell Parameters:
def Paras(typ, dx): # Default sps Cell Parameters
    if typ==1: # Reticulate Pattern Type1
        A = 42; f = 24/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
    if typ==2: # Reticulate Pattern Type2
        A = 30; f = 16/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.75]
    if typ==3: # Reticulate Pattern Type3
        A = 55; f = 16/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1,-0.5],[-0.15,1]]); # Affine Matrix for Type3
    if typ==4: # Reticulate Pattern Type4
        A = 10; f = 15/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1.15,1.1],[-0.45,0.7]]); # Affine Matrix for Type4
    return A, p, f

# Draw Reticulate Pattern Cell(sps):
def DrawCell(dx=180, yi=0, typ=1, ra=0, wa=[], A=42, f=24): # draw sps cell
    xi = round(dx*np.random.rand(),1); dx *= 2;
    dy = round(0.2+(np.random.rand()-0.5)/10, 3);
    if typ in (1,2,3,4):  A,p1,f = Paras(typ,dx); # use default
    else: f /= dx; p1 = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8-(typ%2)/10]
    p2 = p1.copy(); p2[::-3] = [-i for i in p2[::-3]]
    if len(wa)<4: wa = LwAl(2) # get [linewidth, alpha]
    DrawCu("sps", p1, xi, dx, yi+dy, A, ra, f, wa=wa[:2])
    DrawCu("sps", p2, xi, dx, yi-dy, A, ra, f, wa=wa[2:])

# Add Reticulate Net to Image:
def Add2Im(im, typ=1, ra=0, wa=[], gap=1.65): # add to image
    y,x,n = im.shape; n = y//20;
    ofs = round(1.5*np.random.rand(), 2)
    gap = round(gap+(np.random.rand()-0.3)/10, 2);
    if len(wa)<4: wa = [3,0.4, 2,0.6] # linewidth, alpha
    for i in range(2*n): DrawCell(x, gap*(i-n)+ofs, typ, ra, wa)
    imshow(im, extent=(-x/2,x/2,-y/2,y/2)); axis("off")

def Test(iName, oName="out.jpg"):
    im = imread(fName); # load image
	typ = np.random.randint(1,5); # set Reticulate Net type
	ro = 2*np.random.rand()-1; # set rotational angle
    wa = LwAl(2); # get 2*[linewidth, transparence] pairs
    Add2Im(im, typ, ro, wa, gap=1.6); # add Net to image
    savefig(oName); # save the output image
