# -*- coding: utf-8 -*-
"""
Output error metrics and figures based on results of reconstructions

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
import finitetransform.imageio as imageio #local module
import finitetransform.numbertheory as nt #local modules
import numpy as np
from scipy.io import loadmat
from skimage import exposure
#import scipy.fftpack as fftpack
#import finite
import math

#parameters
N = 256
k = 1
M = k*N
SNR = 30
print("N:", N, "M:", M)
plotCroppedImages = True
plotIncrement = 2
plotInset = True
plotEqualised = True 
#addNoise = True
fontsize = 18
insetColor = 'magenta'
p = nt.nearestPrime(M)
print("p:", p)

#create test image
#image, mask = imageio.lena(N, p, True, np.float64, True)
#image2, mask2 = imageio.lena(N, M, True, np.float64, True) #non-prime
image, mask = imageio.phantom(N, p, True, np.float64, True)
image2, mask2 = imageio.phantom(N, M, True, np.float64, True) #non-prime
#image, mask = imageio.cameraman(N, p, True, np.float64, True)
#image2, mask2 = imageio.cameraman(N, M, True, np.float64, True) #non-prime
#image2 /= np.max(image2)
minValue = np.min(image)
print("Min Pixel Value:", minValue)
maxValue = np.max(image2)
print("Max Pixel Value:", maxValue)

#-------------------------------
#load results
#csReconMat = loadmat('result_lena.mat')
csReconMat = loadmat('result_phantom_2.mat')
#csReconMat = loadmat('result_camera.mat')
print("Keys:", csReconMat.keys())
if plotEqualised:
    csRecon = np.real(csReconMat['im_res'])*maxValue
else:
    csRecon = np.real(csReconMat['im_res'])
csIters = csReconMat['iters']
csPSNRS = csReconMat['psnrs']
csSSIMS = csReconMat['ssims']
print("CS Recon Size:", csRecon.shape)
print("CS Recon Min/Max:", csRecon.min(), csRecon.max())

#mlemReconDict = np.load('result_osem.npz')
mlemReconDict = np.load('result_phantom_osem_2.npz')
#mlemReconDict = np.load('result_camera_osem.npz')
mlemRecon = mlemReconDict['recon']
mlemPSNRS = mlemReconDict['psnrs']
mlemSSIMS = mlemReconDict['ssims']
print("MLEM Recon Size:", mlemRecon.shape)
print("MLEM Recon Min/Max:", mlemRecon.min(), mlemRecon.max())

#sirtReconDict = np.load('result_ossirt.npz')
sirtReconDict = np.load('result_phantom_ossirt_2.npz')
#sirtReconDict = np.load('result_camera_ossirt.npz')
sirtRecon = sirtReconDict['recon']
sirtPSNRS = sirtReconDict['psnrs']
sirtSSIMS = sirtReconDict['ssims']
print("SIRT Recon Size:", sirtRecon.shape)
print("SIRT Recon Min/Max:", sirtRecon.min(), sirtRecon.max())

#radialReconDict = np.load('result_radial.npz')
radialReconDict = np.load('result_phantom_radial.npz')
#radialReconDict = np.load('result_camera_radial.npz')
radialRecon = radialReconDict['recon']
radialPSNRS = radialReconDict['psnrs'] #1 iteration
radialSSIMS = radialReconDict['ssims']
print("Radial Recon Size:", radialRecon.shape)
print("Radial Recon Min/Max:", radialRecon.min(), radialRecon.max())

#compute metrics
mse = imageio.immse(imageio.immask(image, mask, N, N), imageio.immask(mlemRecon, mask, N, N))
ssim = imageio.imssim(imageio.immask(image, mask, N, N).astype(float), imageio.immask(mlemRecon, mask, N, N).astype(float))
psnr = imageio.impsnr(imageio.immask(image, mask, N, N), imageio.immask(mlemRecon, mask, N, N))
print("MLEM Results ------")
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

mse = imageio.immse(imageio.immask(image, mask, N, N), imageio.immask(sirtRecon, mask, N, N))
ssim = imageio.imssim(imageio.immask(image, mask, N, N).astype(float), imageio.immask(sirtRecon, mask, N, N).astype(float))
psnr = imageio.impsnr(imageio.immask(image, mask, N, N), imageio.immask(sirtRecon, mask, N, N))
print("SIRT Results ------")
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)

mse = imageio.immse(imageio.immask(image2, mask2, N, N), imageio.immask(csRecon, mask2, N, N))
ssim = imageio.imssim(imageio.immask(image2, mask2, N, N).astype(float), imageio.immask(csRecon, mask2, N, N).astype(float))
psnr = imageio.impsnr(imageio.immask(image2, mask2, N, N), imageio.immask(csRecon, mask2, N, N))
print("CS Results ------")
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim) #1 iteration
print("PSNR:", psnr)

print("Radial Results ------")
#print("RMSE:", math.sqrt(mse))
print("SSIM:", radialSSIMS)
print("PSNR:", radialPSNRS)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from cycler import cycler

if plotEqualised:
    # Equalization
#    image = exposure.equalize_hist(image.astype(np.int32))
#    csRecon = exposure.equalize_hist(csRecon.astype(np.int32))
#    sirtRecon = exposure.equalize_hist(sirtRecon.astype(np.int32))
#    mlemRecon = exposure.equalize_hist(mlemRecon.astype(np.int32))

    # Adaptive Equalization
#    image = exposure.equalize_adapthist(image.astype(np.int32), clip_limit=0.02)
#    csRecon = exposure.equalize_adapthist(csRecon.astype(np.int32), clip_limit=0.02)
#    sirtRecon = exposure.equalize_adapthist(sirtRecon.astype(np.int32), clip_limit=0.02)
#    mlemRecon = exposure.equalize_adapthist(mlemRecon.astype(np.int32), clip_limit=0.02)
    
    #rescale for gamma 
    image_eq = exposure.rescale_intensity(image, out_range=(0, 255))
    image2_eq = exposure.rescale_intensity(image2, out_range=(0, 255))
    csRecon_eq = exposure.rescale_intensity(csRecon, out_range=(0, 255))
    sirtRecon_eq = exposure.rescale_intensity(sirtRecon, out_range=(0, 255))
    mlemRecon_eq = exposure.rescale_intensity(mlemRecon, out_range=(0, 255))
    radialRecon_eq = exposure.rescale_intensity(radialRecon, out_range=(0, 255))
    
    # Gamma
    image_eq = exposure.adjust_gamma(image_eq, 0.5)
    image2_eq = exposure.adjust_gamma(image2_eq, 0.5)
    csRecon_eq = exposure.adjust_gamma(csRecon_eq, 0.5)
    sirtRecon_eq = exposure.adjust_gamma(sirtRecon_eq, 0.5)
    mlemRecon_eq = exposure.adjust_gamma(mlemRecon_eq, 0.5)
    radialRecon_eq = exposure.adjust_gamma(radialRecon_eq, 0.5)
    
    # Logarithmic
#    image = exposure.adjust_log(image, 2)
#    csRecon = exposure.adjust_log(csRecon, 2)
#    sirtRecon = exposure.adjust_log(sirtRecon, 2)
#    mlemRecon = exposure.adjust_log(mlemRecon, 2)
    
    minValue = np.min(image_eq)
    print("Min Pixel Value:", minValue)
    maxValue = np.max(image_eq)
    print("Max Pixel Value:", maxValue)
    
else:
    image_eq = image
    image2_eq = image2
    csRecon_eq = csRecon
    sirtRecon_eq = sirtRecon
    mlemRecon_eq = mlemRecon
    radialRecon_eq = radialRecon
    
#adjust range for better display
#minValue = 40
#maxValue = 128

#pp = PdfPages('finite_metric_plots_lena.pdf')
pp = PdfPages('finite_metric_plots_phantom.pdf')
#pp = PdfPages('finite_metric_plots_camera.pdf')

fig, ax = plt.subplots(figsize=(24, 9))

plt.tight_layout()
plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

plt.tight_layout()

if plotCroppedImages:
    image_eq = imageio.immask(image_eq, mask, N, N)
    image2_eq = imageio.immask(image2_eq, mask2, N, N)
    mlemRecon_eq = imageio.immask(mlemRecon_eq, mask, N, N)
    sirtRecon_eq = imageio.immask(sirtRecon_eq, mask, N, N)
    csRecon_eq = imageio.immask(csRecon_eq, mask2, N, N)
    radialRecon_eq = imageio.immask(radialRecon_eq, mask2, N, N)
    
#inset parameters
#lena
#x1, x2, y1, y2 = 90, 170, 170, 90 # specify the limits
#phantom
x1, x2, y1, y2 = 40, 130, 160, 80 # specify the limits
x1, x2, y1, y2 = 40, 130, 160, 80 # specify the limits
#camera
#x1, x2, y1, y2 = 100, 190, 140, 40 # specify the limits
#x1, x2, y1, y2 = 100, 190, 140, 40 # specify the limits

#coordinates of profile
offset1 = 20
offset2 = 30
profileX1 = int( x1+7*(x2-x1)/8 )
profileY1 = np.arange(y2+offset1, y1-offset1)
profileX2 = np.arange(x1+offset2/3, x2-offset2)
profileY2 = int( y2 + (y1-y2)/3 )
#print("profileX1:", profileX1)
#print("profileY1:", profileY1)
#print("profileX2:", profileX2)
#print("profileY2:", profileY2)

plt.subplot(151)
rax = plt.imshow(image_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax = plt.imshow(image_eq, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax, cmap='gray')
for i, yVal in enumerate(profileY1):
    plt.plot(profileX1, yVal, '.r', markersize=2)
for i, xVal in enumerate(profileX2):
    plt.plot(xVal, profileY2, '.b', markersize=2)
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset Image', fontsize=fontsize)
plt.subplot(152)
rax5 = plt.imshow(radialRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax5 = plt.imshow(radialRecon_eq, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax5, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset Radial', fontsize=fontsize)
plt.subplot(153)
rax2 = plt.imshow(csRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax2 = plt.imshow(csRecon_eq, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax2, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset CS', fontsize=fontsize)
plt.subplot(154)
rax3 = plt.imshow(sirtRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax3 = plt.imshow(sirtRecon_eq, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax3, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset SIRT', fontsize=fontsize)
plt.subplot(155)
rax4 = plt.imshow(mlemRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax4 = plt.imshow(mlemRecon_eq, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax4, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset MLEM', fontsize=fontsize)
pp.savefig()

fig, ax = plt.subplots(figsize=(24, 9))

plt.tight_layout()
plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

plt.subplot(151)
rax = plt.imshow(image, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
#rax = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax, cmap='gray')
for i, yVal in enumerate(profileY1):
    plt.plot(profileX1, yVal, '.r', markersize=2)
for i, xVal in enumerate(profileX2):
    plt.plot(xVal, profileY2, '.b', markersize=2)
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset Image', fontsize=fontsize)
plt.subplot(152)
rax5 = plt.imshow(image2-radialRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax5 = plt.imshow(csRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax5, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset Radial', fontsize=fontsize)
plt.subplot(153)
rax2 = plt.imshow(image2-csRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax2 = plt.imshow(csRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax2, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset CS', fontsize=fontsize)
plt.subplot(154)
rax3 = plt.imshow(image-sirtRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax3 = plt.imshow(sirtRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax3, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset SIRT', fontsize=fontsize)
plt.subplot(155)
rax4 = plt.imshow(image-mlemRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax4 = plt.imshow(mlemRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax4, cmap='gray')
if plotInset:
    plt.xlim(x1, x2) # apply the x-limits
    plt.ylim(y1, y2) # apply the y-limits
plt.title('Inset MLEM', fontsize=fontsize)
pp.savefig()

#plot diffs
fig, ax = plt.subplots(figsize=(24, 9))

plt.tight_layout()
plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

plt.subplot(141)
rax5 = plt.imshow(image2-radialRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax5 = plt.imshow(image2-csRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax5, cmap='gray')
plt.title('Radial Errors', fontsize=fontsize)
plt.subplot(142)
rax2 = plt.imshow(image2-csRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax2 = plt.imshow(image2-csRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax2, cmap='gray')
plt.title('CS Errors', fontsize=fontsize)
plt.subplot(143)
rax3 = plt.imshow(image-sirtRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax4 = plt.imshow(image-sirtRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax3, cmap='gray')
plt.title('SIRT Errors', fontsize=fontsize)
plt.subplot(144)
rax4 = plt.imshow(image-mlemRecon, interpolation='nearest', cmap='gray', vmin=-15, vmax=15)
#rax4 = plt.imshow(image-mlemRecon, cmap='gray', vmin=0, vmax=255)
rcbar = plt.colorbar(rax4, cmap='gray')
plt.title('MLEM Errors', fontsize=fontsize)
pp.savefig()

#plot convergence
fig, ax = plt.subplots(figsize=(24, 9))

plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize) 

xVals1 = np.arange(0, len(mlemSSIMS))*plotIncrement
xVals2 = np.arange(0, len(csIters))*8
xVals3 = np.arange(0, len(sirtSSIMS))*plotIncrement
#print(csIters)
#print(xVals2)

keysPSNR = []
keysSSIM = []

plt.subplot(121)
linePSNR, = plt.plot(xVals2, csPSNRS, label='CS', marker='x')
keysPSNR.append(linePSNR)
linePSNR, = plt.plot(xVals1, mlemPSNRS, label='Finite MLEM', marker='^')
keysPSNR.append(linePSNR)
#plt.plot(x2, cgPSNRS)
linePSNR, = plt.plot(xVals3, sirtPSNRS, label='Finite SIRT', marker='+')
keysPSNR.append(linePSNR)
plt.ylim(5, 50)
plt.xlim(0, 340)
plt.title('PSNR Convergence', fontsize=fontsize)
plt.legend(handles=keysPSNR, loc=4, fontsize=fontsize)

plt.subplot(122)
lineSSIM, = plt.plot(xVals2, csSSIMS, label='CS', marker='x')
keysSSIM.append(lineSSIM)
lineSSIM, = plt.plot(xVals1, mlemSSIMS, label='Finite MLEM', marker='^')
keysSSIM.append(lineSSIM)
#plt.plot(x2, cgSSIMS)
lineSSIM, = plt.plot(xVals3, sirtSSIMS, label='Finite SIRT', marker='+')
keysSSIM.append(lineSSIM)
plt.ylim(0.2, 1.0)
plt.xlim(0, 340)
plt.title('SSIM Convergence', fontsize=fontsize)
plt.legend(handles=keysSSIM, loc=4, fontsize=fontsize)
pp.savefig()

#plot profiles

#extract profiles
prof1 = np.zeros_like(profileY1)
prof2 = np.zeros_like(profileY1)
prof3 = np.zeros_like(profileY1)
prof4 = np.zeros_like(profileY1)
prof5 = np.zeros_like(profileY1)
for i, yVal in enumerate(profileY1):
#    prof1[i] = image[profileX1, yVal]
#    prof2[i] = csRecon[profileX1, yVal]
#    prof3[i] = mlemRecon[profileX1, yVal]
#    prof4[i] = sirtRecon[profileX1, yVal]
    prof1[i] = image[yVal, profileX1]
    prof2[i] = csRecon[yVal, profileX1]
    prof3[i] = mlemRecon[yVal, profileX1]
    prof4[i] = sirtRecon[yVal, profileX1]
    prof5[i] = radialRecon[yVal, profileX1]

fig, ax = plt.subplots(figsize=(8, 5))

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm']) +
#                           cycler('linestyle', ['-', '--', ':', '-.', '-'])))
#                           cycler('linestyle', ['solid', 'densely dashed', 'densely dashdotted', 'densely dashdotdotted', 'dotted'])))
                           cycler('linestyle', [(0, ()), (0, (3, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 2))])))

plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize) 

xVals1 = profileY1
#xVals3 = np.arange(0, len(sirtSSIMS))*plotIncrement

keysProf = []
profImg, = plt.plot(xVals1, prof1, label='Original', linewidth=3)
profCS, = plt.plot(xVals1, prof2, label='CS', linewidth=3)
profMLEM, = plt.plot(xVals1, prof3, label='Finite MLEM', linewidth=3)
profSIRT, = plt.plot(xVals1, prof4, label='Finite SIRT', linewidth=3)
profRadial, = plt.plot(xVals1, prof5, label='Radial', linewidth=3)
keysProf.append(profImg)
keysProf.append(profRadial)
keysProf.append(profCS)
keysProf.append(profMLEM)
keysProf.append(profSIRT)
plt.legend(handles=keysProf, loc=0, fontsize=fontsize)
pp.savefig()

#extract profiles
prof1 = np.zeros_like(profileX2)
prof2 = np.zeros_like(profileX2)
prof3 = np.zeros_like(profileX2)
prof4 = np.zeros_like(profileX2)
prof5 = np.zeros_like(profileX2)
for i, xVal in enumerate(profileX2):
#    prof1[i] = image[xVal, profileY2]
#    prof2[i] = csRecon[xVal, profileY2]
#    prof3[i] = mlemRecon[xVal, profileY2]
#    prof4[i] = sirtRecon[xVal, profileY2]
    prof1[i] = image[profileY2, xVal]
    prof2[i] = csRecon[profileY2, xVal]
    prof3[i] = mlemRecon[profileY2, xVal]
    prof4[i] = sirtRecon[profileY2, xVal]
    prof5[i] = radialRecon[profileY2, xVal]

fig, ax = plt.subplots(figsize=(8, 5))

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm']) +
#                           cycler('linestyle', ['-', '--', ':', '-.', '-'])))
#                           cycler('linestyle', ['solid', 'densely dashed', 'densely dashdotted', 'densely dashdotdotted', 'dotted'])))
                           cycler('linestyle', [(0, ()), (0, (3, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 2))])))
plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize) 

xVals1 = profileX2
#xVals3 = np.arange(0, len(sirtSSIMS))*plotIncrement

keysProf = []
profImg, = plt.plot(xVals1, prof1, label='Original', linewidth=3)
profCS, = plt.plot(xVals1, prof2, label='CS', linewidth=3)
profMLEM, = plt.plot(xVals1, prof3, label='Finite MLEM', linewidth=3)
profSIRT, = plt.plot(xVals1, prof4, label='Finite SIRT', linewidth=3)
profRadial, = plt.plot(xVals1, prof5, label='Radial', linewidth=3)
keysProf.append(profImg)
keysProf.append(profRadial)
keysProf.append(profCS)
keysProf.append(profMLEM)
keysProf.append(profSIRT)
plt.legend(handles=keysProf, loc=1, fontsize=fontsize)
pp.savefig()

##############
#Inset Image
fig, ax = plt.subplots(figsize=(8, 6))

plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

rax = plt.imshow(image_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image', fontsize=fontsize)

axins = zoomed_inset_axes(ax, 2.0, loc=4) # zoom-factor: 2.5, location: upper-left
#x1, x2, y1, y2 = 110, 140, 190, 220 # specify the limits
x1, x2, y1, y2 = 110, 140, 110, 140 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins.spines[axis].set_linewidth(2)
    axins.spines[axis].set_color(insetColor)
mark_inset(ax, axins, loc1=1, loc2=3, ec=insetColor)
axins.imshow(image_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

#radial
fig2, ax2 = plt.subplots(figsize=(8, 6))

rax2 = plt.imshow(radialRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Radial Reconstruction', fontsize=fontsize)

axins2 = zoomed_inset_axes(ax2, 2.0, loc=4) # zoom-factor: 2.5, location: upper-left
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_linewidth(2)
    axins2.spines[axis].set_color(insetColor)
mark_inset(ax2, axins2, loc1=1, loc2=3, ec=insetColor)
axins2.imshow(radialRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

#CS
fig2, ax2 = plt.subplots(figsize=(8, 6))

rax2 = plt.imshow(csRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('CS Reconstruction', fontsize=fontsize)

axins2 = zoomed_inset_axes(ax2, 2.0, loc=4) # zoom-factor: 2.5, location: upper-left
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_linewidth(2)
    axins2.spines[axis].set_color(insetColor)
mark_inset(ax2, axins2, loc1=1, loc2=3, ec=insetColor)
axins2.imshow(csRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

#fSIRT
fig2, ax2 = plt.subplots(figsize=(8, 6))

rax2 = plt.imshow(sirtRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Finite SIRT Reconstruction', fontsize=fontsize)

axins2 = zoomed_inset_axes(ax2, 2.0, loc=4) # zoom-factor: 2.5, location: upper-left
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_linewidth(2)
    axins2.spines[axis].set_color(insetColor)
mark_inset(ax2, axins2, loc1=1, loc2=3, ec=insetColor)
axins2.imshow(sirtRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

#fMLEM
fig2, ax2 = plt.subplots(figsize=(8, 6))

rax2 = plt.imshow(mlemRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Finite MLEM Reconstruction', fontsize=fontsize)

axins2 = zoomed_inset_axes(ax2, 2.0, loc=4) # zoom-factor: 2.5, location: lower-right
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_linewidth(2)
    axins2.spines[axis].set_color(insetColor)
mark_inset(ax2, axins2, loc1=1, loc2=3, ec=insetColor)
axins2.imshow(mlemRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

plt.show()

print("Complete")