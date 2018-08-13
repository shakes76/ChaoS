# -*- coding: utf-8 -*-
"""
Generate and Compare the Ghost artefacts created by various sampling strategies.

#Reduction Factors
#0.5:
#N=256, K=1.2, k=1, r=2;
#0.25:
#N=256, K=0.4, k=1, r=4;
#0.125:
#N=256, K=0.15, k=1, r=8;

Copyright 2018 Shekhar S. Chandra

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import finitetransform.mojette as mojette
import finitetransform.tomography as tomo
import finitetransform.farey as farey #local module
import finitetransform.numbertheory as nt #local modules
import numpy as np
import finite
import math

import matplotlib
matplotlib.use('Qt4Agg')

#parameters
N = 256
k = 1
M = k*N
K = 0.4
twoQuads = True
print("N:", N, "M:", M)

angles, lengths = mojette.angleSet_Symmetric(N,N,1,True,K)
#angles, lengths = mojette.angleSet_Symmetric(M,M,1,True,K)
perpAngle = farey.farey(1,0)
angles.append(perpAngle)
print("Number of Angles:", len(angles))
print("angles:", angles)

p = nt.nearestPrime(M)
print("p:", p)

#empty space
kSpace = np.zeros((p,p), np.complex64)
kSpace2 = np.zeros((M,M), np.complex64)
powSpect = np.log10(np.abs(kSpace))

#compute finite lines
centered = True
lines, mValues = finite.computeLines(kSpace, angles, centered, twoQuads)
mu = len(lines)
print("Number of finite lines:", len(lines))
print("Number of finite points:", len(lines)*(p-1))
print(mValues)

#compute radial lines
r = 4
print("N/r:", N/r)
radialAngles = np.linspace(0, 720.0, float(N)/r, endpoint=True)
#print(radialAngles)
radialLines = []
for angle in radialAngles:
    u, v = tomo.sliceSamples(angle, M, M, kSpace.shape)
    location = v + 1j*u
    radialLines.append(location)
print("Number of radial lines:", len(radialLines))
print("Number of radial points:", len(radialLines)*M)
    
#random samples
radius = N/24
centerX = M/2
centerY = M/2
randomSamples = []
for i, row in enumerate(kSpace2):
    line = tomo.randomSamples(i, N/r, kSpace2.shape)
    u, v = line
    randomSamples.append(line)
#centre
count = 0
u = []
v = []
for i, row in enumerate(kSpace2):
    for j, col in enumerate(row):
        distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
        if distance <= radius:
            count += 1
            u.append(i)
            v.append(j)
line = np.array(u), np.array(v)
randomSamples.append(line)
print("Number of random lines:", len(randomSamples))
print("Number of random points:", len(randomSamples)*N/r+count)

#plot slices responsible for reconstruction
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
plt.gray()

maxLines = len(lines)
ax[0].imshow(powSpect)
color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
for i, line in enumerate(lines):
    u, v = line
    c=next(color)
    ax[0].plot(u, v, '.', c=c, markersize=2)
ax[0].set_title('Fractal Sampling (colour per line)')
maxLines = len(radialLines)
ax[1].imshow(powSpect)
color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
for i, line in enumerate(radialLines):
    u = line.real
    v = line.imag
    c=next(color)
    ax[1].plot(u, v, '.', c=c, markersize=2)
ax[1].set_title('Radial Sampling (colour per line)')
maxLines = len(randomSamples)
ax[2].imshow(powSpect)
color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
for i, line in enumerate(randomSamples):
    u, v = line
    c=next(color)
    ax[2].plot(u, v, '.', c=c, markersize=2)
ax[2].set_title('Random Sampling (colour per line)')

plt.tight_layout()

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 8))
plt.gray()

maxLines = len(lines)
#maxLines = 12
#finite
ax[0,0].imshow(powSpect)
ax[0,1].imshow(powSpect)
#color=iter(cm.rainbow(np.linspace(0,1,len(lines))))
color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
for i, line in enumerate(lines):
    u, v = line
    c=next(color)
    ax[0,0].plot(u, v, '.', c=c)
    ax[0,1].plot(u, v, '.r',markersize=0.5)
    if i == maxLines:
        break
ax[0,0].set_title('Fractal Sampling (colour per line) for prime:'+str(p))
ax[0,1].set_title('Fractal Sampling (same colour per line) for prime:'+str(p))
#ax[0].set_xlim([0,M])
#ax[0].set_ylim([0,M])
#ax[1].set_xlim([0,M])
#ax[1].set_ylim([0,M])
#radial
maxLines = len(radialLines)
ax[1,0].imshow(powSpect)
ax[1,1].imshow(powSpect)
color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
for i, line in enumerate(radialLines):
    u = line.real
    v = line.imag
    c=next(color)
    ax[1,0].plot(u, v, '.', c=c)
    ax[1,1].plot(u, v, '.r',markersize=0.5)
    if i == maxLines:
        break
ax[1,0].set_title('Radial Sampling (colour per line) for prime:'+str(p))
ax[1,1].set_title('Radial Sampling (same colour per line) for prime:'+str(p))
#random
maxLines = len(randomSamples)
ax[2,0].imshow(powSpect)
ax[2,1].imshow(powSpect)
color=iter(cm.jet(np.linspace(0,1,maxLines+1)))
for i, line in enumerate(randomSamples):
    u, v = line
    c=next(color)
    ax[2,0].plot(u, v, '.', c=c)
    ax[2,1].plot(u, v, '.r',markersize=0.5)
    if i == maxLines:
        break
ax[2,0].set_title('Random Sampling (colour per line) for prime:'+str(p))
ax[2,1].set_title('Random Sampling (same colour per line) for prime:'+str(p))

plt.tight_layout()
plt.show()