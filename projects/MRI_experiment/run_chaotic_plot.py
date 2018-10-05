# -*- coding: utf-8 -*-
"""
Re-Plot the Chaotic ABMLEM results with insets.

http://akuederle.com/matplotlib-zoomed-up-inset

Created on Sun Aug 27 20:59:27 2017

@author: shakes
"""
from __future__ import print_function    # (at top of module)
import _libpath #add custom libs
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import finitetransform.imageio as imageio #local module
from scipy.io import loadmat
from skimage import exposure
import math

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from cycler import cycler

fontsize = 18
insetColor = 'magenta'
plotEqualised = True 

#-------------------------------
#load results
#filename = 'result_abmlem_2.npz'
filename = 'result_abmlem_4.npz'
#filename = 'result_abmlem_8.npz'
with np.load(filename) as data:
    image = data['image']
    rows, cols = image.shape
    image2 = data['image'][1:rows, 1:cols]
    recon = data['recon']
    diff = data['diff']
    iterations = data['iterations']
    K = data['K']
minValue = np.min(image)
print("Min Pixel Value:", minValue)
maxValue = np.max(image)
print("Max Pixel Value:", maxValue)
    
#csReconMat = loadmat('result_lego_2.mat')
csReconMat = loadmat('result_lego_4.mat')
#csReconMat = loadmat('result_lego_8.mat')
csRecon = np.abs(csReconMat['im_res'])*maxValue
diff2 = image2-csRecon

data2 = np.load('result_radial.npz')
radialRecon = data2['recon']
diff3 = image2-radialRecon
    
#-------------------------------
#compute metrics
mse = imageio.immse(image, np.abs(recon))
ssim = imageio.imssim(image.astype(float), np.abs(recon).astype(float))
psnr = imageio.impsnr(image, np.abs(recon))
print("ChaoS RMSE:", math.sqrt(mse))
print("ChaoS SSIM:", ssim)
print("ChaoS PSNR:", psnr)
mse = imageio.immse(image2, np.abs(csRecon))
ssim = imageio.imssim(image2.astype(float), np.abs(csRecon).astype(float))
psnr = imageio.impsnr(image2, np.abs(csRecon))
print("CS RMSE:", math.sqrt(mse))
print("CS SSIM:", ssim)
print("CS PSNR:", psnr)
mse = imageio.immse(image2, np.abs(radialRecon))
ssim = imageio.imssim(image2.astype(float), np.abs(radialRecon).astype(float))
psnr = imageio.impsnr(image2, np.abs(radialRecon))
print("Radial RMSE:", math.sqrt(mse))
print("Radial SSIM:", ssim)
print("Radial PSNR:", psnr)
    
#-------------------------------
#Plot
print('Plotting results for K =', K, 'with', iterations, 'iterations')  

x1, x2, y1, y2 = 40, 130, 160, 80 # specify the limits
x1, x2, y1, y2 = 40, 130, 160, 80 # specify the limits

#coordinates of profile
offset1 = 20
profileX1 = int( x1+(x2-x1) + offset1 )
profileY1 = np.arange(y2+offset1/2, y1-offset1/2)
#print("profileX1:", profileX1)
#print("profileY1:", profileY1)

if plotEqualised:
    
    #rescale for gamma 
    image_eq = exposure.rescale_intensity(image, out_range=(0, 255))
    image2_eq = exposure.rescale_intensity(image2, out_range=(0, 255))
    csRecon_eq = exposure.rescale_intensity(csRecon, out_range=(0, 255))
    recon_eq = exposure.rescale_intensity(recon, out_range=(0, 255))
    radialRecon_eq = exposure.rescale_intensity(radialRecon, out_range=(0, 255))
    
    # Gamma
    image_eq = exposure.adjust_gamma(image_eq, 0.5)
    image2_eq = exposure.adjust_gamma(image2_eq, 0.5)
    csRecon_eq = exposure.adjust_gamma(csRecon_eq, 0.5)
    recon_eq = exposure.adjust_gamma(recon_eq, 0.5)
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
    recon_eq = recon  
    radialRecon_eq = radialRecon  

fig, ax = plt.subplots(figsize=(24, 9))

plt.subplot(141)
rax = plt.imshow(image_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax = plt.imshow(image_eq, cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image')
plt.subplot(142)
rax2 = plt.imshow(radialRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax2 = plt.imshow(csRecon_eq, cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Radial Reconstruction')
plt.subplot(143)
rax2 = plt.imshow(csRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax2 = plt.imshow(csRecon_eq, cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('CS Reconstruction')
plt.subplot(144)
rax3 = plt.imshow(recon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax3 = plt.imshow(recon_eq, cmap='gray')
rcbar3 = plt.colorbar(rax3, cmap='gray')
#plt.title('Reconstruction Errors\n(RMSE='+'{:3.2f}'.format(math.sqrt(mse))+')')
plt.title('ChaoS Reconstruction')

fig, ax = plt.subplots(figsize=(24, 9))

plt.subplot(131)
rax = plt.imshow(image_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax = plt.imshow(image_eq, cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image')
plt.subplot(132)
rax2 = plt.imshow(recon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax2 = plt.imshow(recon_eq, cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('ChaoS Reconstruction')
plt.subplot(133)
rax3 = plt.imshow(diff, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)
#rax3 = plt.imshow(diff, cmap='gray')
rcbar3 = plt.colorbar(rax3, cmap='gray')
#plt.title('Reconstruction Errors\n(RMSE='+'{:3.2f}'.format(math.sqrt(mse))+')')
plt.title('Reconstruction Errors')

fig, ax = plt.subplots(figsize=(24, 9))

plt.subplot(131)
csrax = plt.imshow(image2_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax = plt.imshow(image_eq, cmap='gray')
csrcbar = plt.colorbar(csrax, cmap='gray')
plt.title('Image')
plt.subplot(132)
csrax2 = plt.imshow(csRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
#rax2 = plt.imshow(csRecon_eq, cmap='gray')
csrcbar2 = plt.colorbar(csrax2, cmap='gray')
plt.title('CS Reconstruction')
plt.subplot(133)
csrax3 = plt.imshow(diff2, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)
#rax3 = plt.imshow(diff, cmap='gray')
csrcbar3 = plt.colorbar(csrax3, cmap='gray')
#plt.title('Reconstruction Errors\n(RMSE='+'{:3.2f}'.format(math.sqrt(mse))+')')
plt.title('Reconstruction Errors')

fig, ax = plt.subplots(figsize=(8, 6))

plt.rc('xtick', labelsize=fontsize-4) 
plt.rc('ytick', labelsize=fontsize-4) 

rax = plt.imshow(image_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
for i, yVal in enumerate(profileY1):
    plt.plot(profileX1, yVal, '.r', markersize=1)
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image', fontsize=fontsize)

axins = zoomed_inset_axes(ax, 2.0, loc=3) # zoom-factor: 2.5, location: upper-left
x1, x2, y1, y2 = 130, 175, 120, 160 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins.spines[axis].set_linewidth(2)
    axins.spines[axis].set_color(insetColor)
mark_inset(ax, axins, loc1=2, loc2=4, ec=insetColor)
axins.imshow(image_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

##############
fig2, ax2 = plt.subplots(figsize=(8, 6))

rax2 = plt.imshow(radialRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Radial Reconstruction', fontsize=fontsize)

axins2 = zoomed_inset_axes(ax2, 2.0, loc=3) # zoom-factor: 2.5, location: upper-left
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_linewidth(2)
    axins2.spines[axis].set_color(insetColor)
mark_inset(ax2, axins2, loc1=2, loc2=4, ec=insetColor)
axins2.imshow(radialRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

###############
fig3, ax3 = plt.subplots(figsize=(8, 6))

rax3 = plt.imshow(diff3, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)
rcbar3 = plt.colorbar(rax3, cmap='gray')
plt.title('Radial Reconstruction Errors', fontsize=fontsize)

axins3 = zoomed_inset_axes(ax3, 2.0, loc=3) # zoom-factor: 2.5, location: upper-left
axins3.set_xlim(x1, x2) # apply the x-limits
axins3.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins3.spines[axis].set_linewidth(2)
    axins3.spines[axis].set_color(insetColor)
mark_inset(ax3, axins3, loc1=4, loc2=2, ec=insetColor)
axins3.imshow(diff3, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)

##############
fig2, ax2 = plt.subplots(figsize=(8, 6))

rax2 = plt.imshow(csRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('CS Reconstruction', fontsize=fontsize)

axins2 = zoomed_inset_axes(ax2, 2.0, loc=3) # zoom-factor: 2.5, location: upper-left
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_linewidth(2)
    axins2.spines[axis].set_color(insetColor)
mark_inset(ax2, axins2, loc1=2, loc2=4, ec=insetColor)
axins2.imshow(csRecon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

###############
fig3, ax3 = plt.subplots(figsize=(8, 6))

rax3 = plt.imshow(diff2, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)
rcbar3 = plt.colorbar(rax3, cmap='gray')
plt.title('CS Reconstruction Errors', fontsize=fontsize)

axins3 = zoomed_inset_axes(ax3, 2.0, loc=3) # zoom-factor: 2.5, location: upper-left
axins3.set_xlim(x1, x2) # apply the x-limits
axins3.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins3.spines[axis].set_linewidth(2)
    axins3.spines[axis].set_color(insetColor)
mark_inset(ax3, axins3, loc1=4, loc2=2, ec=insetColor)
axins3.imshow(diff2, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)

##############
fig2, ax2 = plt.subplots(figsize=(8, 6))

rax2 = plt.imshow(recon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('ChaoS Reconstruction', fontsize=fontsize)

axins2 = zoomed_inset_axes(ax2, 2.0, loc=3) # zoom-factor: 2.5, location: upper-left
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins2.spines[axis].set_linewidth(2)
    axins2.spines[axis].set_color(insetColor)
mark_inset(ax2, axins2, loc1=2, loc2=4, ec=insetColor)
axins2.imshow(recon_eq, interpolation='nearest', cmap='gray', vmin=minValue, vmax=maxValue)

###############
fig3, ax3 = plt.subplots(figsize=(8, 6))

rax3 = plt.imshow(diff, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)
rcbar3 = plt.colorbar(rax3, cmap='gray')
plt.title('ChaoS Reconstruction Errors', fontsize=fontsize)

axins3 = zoomed_inset_axes(ax3, 2.0, loc=3) # zoom-factor: 2.5, location: upper-left
axins3.set_xlim(x1, x2) # apply the x-limits
axins3.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
for axis in ['top','bottom','left','right']:
    axins3.spines[axis].set_linewidth(2)
    axins3.spines[axis].set_color(insetColor)
mark_inset(ax3, axins3, loc1=4, loc2=2, ec=insetColor)
axins3.imshow(diff, interpolation='nearest', cmap='gray', vmin=-2, vmax=2)

###############
#plot profiles

#extract profiles
prof1 = np.zeros_like(profileY1)
prof2 = np.zeros_like(profileY1)
prof3 = np.zeros_like(profileY1)
prof4 = np.zeros_like(profileY1)
for i, yVal in enumerate(profileY1):
#    prof1[i] = image[profileX1, yVal]
#    prof2[i] = csRecon[profileX1, yVal]
#    prof3[i] = mlemRecon[profileX1, yVal]
#    prof4[i] = sirtRecon[profileX1, yVal]
    prof1[i] = image[yVal, profileX1]
    prof2[i] = csRecon[yVal, profileX1]
    prof3[i] = recon[yVal, profileX1]
    prof4[i] = radialRecon[yVal, profileX1]

fig, ax = plt.subplots(figsize=(8, 5))

plt.rc('lines', linewidth=4)
plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
#                           cycler('linestyle', ['-', '--', ':', '-.', '-'])))
#                           cycler('linestyle', ['solid', 'densely dashed', 'densely dashdotted', 'dotted'])))
                           cycler('linestyle', [(0, ()), (0, (3, 1)), (0, (3, 1, 1, 1)), (0, (1, 2))])))


plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize) 

xVals1 = profileY1

keysProf = []
profImg, = plt.plot(xVals1, prof1, label='Original', linewidth=3)
profCS, = plt.plot(xVals1, prof2, label='CS', linewidth=3)
profMLEM, = plt.plot(xVals1, prof3, label='Finite MLEM', linewidth=3)
profRadial, = plt.plot(xVals1, prof4, label='Radial', linewidth=3)
keysProf.append(profImg)
keysProf.append(profRadial)
keysProf.append(profCS)
keysProf.append(profMLEM)
plt.legend(handles=keysProf, loc=0, fontsize=fontsize)

plt.draw()
plt.show()

print("Complete")
