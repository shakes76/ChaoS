# -*- coding: utf-8 -*-
"""
Create the Farey FInite Fractal as a sampling pattern for MRI

All figures and code pertaining to the display, saving and generation of fractals, 
are covered under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International Public License: http://creativecommons.org/licenses/by-nc-sa/4.0/.
For publication and commercial use of this content, please obtain a suitable license 
from Shekhar S. Chandra.
"""
from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import finitetransform.numbertheory as nt #local modules
import finitetransform.mojette as mojette
import finitetransform.radon as radon
import finitetransform.imageio as imageio #local module
import finitetransform.farey as farey #local module
import numpy as np

#parameters
N = 512
M = 2*N
K = 1
twoQuads = True
print("N:", N, "M:", M)
p = nt.nearestPrime(M)
print("p:", p)
pDash = nt.nearestPrime(N)
print("p':", pDash)
#angles = mojette.angleSet_Finite(pDash, 2)
angles, lengths = mojette.angleSet_Symmetric(N,N,1,True,K)
perpAngle = farey.farey(1,0)
angles.append(perpAngle)
print("Number of Angles:", len(angles))
print("angles:", angles)

powerSpect = np.zeros((p,p))

#compute lines
print("Computing Finite lines...")
centered = True
lines = []
mValues = []
for angle in angles:
    m, inv = farey.toFinite(angle, p)
    u, v = radon.getSliceCoordinates2(m, powerSpect, centered, p)
    lines.append((u,v))
    mValues.append(m)
    #second quadrant
    if twoQuads:
        if m != 0 and m != p: #dont repeat these
            m = p-m
            u, v = radon.getSliceCoordinates2(m, powerSpect, centered, p)
            lines.append((u,v))
            mValues.append(m)
mu = len(lines)
print("Number of lines:", len(lines))
print("Proportion of p:", len(lines)/float(p))
print("Proportion of 2D space:", 2.0-len(lines)/float(p))
print(mValues)

#plot slices responsible for reconstruction
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

plt.gray()
plt.tight_layout()

maxLines = len(lines)
#maxLines = 12
ax[0].imshow(powerSpect)
ax[1].imshow(powerSpect)
#color=iter(cm.rainbow(np.linspace(0,1,len(lines))))
color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
fareyImage = np.zeros_like(powerSpect)
for i, line in enumerate(lines):
    u, v = line
    c=next(color)
    ax[0].plot(u, v, '.', c=c)
    ax[1].plot(u, v, '.r',markersize=1)
    fareyImage[u,v] = 255
    if i == maxLines:
        break
ax[0].set_title('Sampling (colour per line) for prime size:'+str(p))
ax[1].set_title('Sampling (same colour per line) for prime size:'+str(p))
#ax[0].set_xlim([0,M])
#ax[0].set_ylim([0,M])
#ax[1].set_xlim([0,M])
#ax[1].set_ylim([0,M])

imageio.imsave("farey_image_"+str(p)+"_"+str(K)+".png", fareyImage)

plt.show()
