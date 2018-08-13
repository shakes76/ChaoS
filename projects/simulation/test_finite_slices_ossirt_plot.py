# -*- coding: utf-8 -*-
"""
Create a finite fractal sampling of k-space and reconstruct using MLEM

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
import finitetransform.radon as radon
import finitetransform.imageio as imageio #local module
import finitetransform.farey as farey #local module
import finitetransform.numbertheory as nt #local modules
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import finite
import time
import math

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#parameter sets (K, k, i, s, h)
#phantom
#parameters = [1.2, 1, 381, 4, 8.0] #r=2
parameters = [0.4, 1, 761, 2, 8.0] #r=4
#camera
#parameters = [1.2, 1, 380, 4, 6.0] #r=2
#parameters = [0.4, 1, 760, 2, 6.0] #r=2

#parameters
N = 256 
k = parameters[1]
M = k*N
K = parameters[0]
s = parameters[3]
epsilon = 0.005
t = 6/(1+epsilon) #Gregor 2008
iterations = parameters[2]
subsetsMode = 1
SNR = 30
floatType = np.complex64
twoQuads = True
addNoise = True
plotCroppedImages = True
plotColourBar = True
plotIncrement = 2
smoothReconMode = 2 #0-None,1-TV,2-NL,3-Median
smoothIncrement = 10
smoothMaxIteration = iterations/2
relaxIterationFactor = int(0.02*iterations)
smoothMaxIteration2 = iterations-relaxIterationFactor*smoothIncrement
print("N:", N, "M:", M, "s:", s, "i:", iterations, "t:", t)

pDash = nt.nearestPrime(N)
print("p':", pDash)
angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,N,N,1,True,K)
#angles, subsetsAngles, lengths = mojette.angleSubSets_Symmetric(s,subsetsMode,M,M,1,True,K)
perpAngle = farey.farey(1,0)
angles.append(perpAngle)
subsetsAngles[0].append(perpAngle)
print("Number of Angles:", len(angles))
print("angles:", angles)

p = nt.nearestPrime(M)
print("p:", p)

#check if Katz compliant
if not mojette.isKatzCriterion(M, M, angles):
    print("Warning: Katz Criterion not met")

#create test image
#lena, mask = imageio.lena(N, p, True, np.uint32, True)
lena, mask = imageio.phantom(N, p, True, np.uint32, True)
#lena, mask = imageio.cameraman(N, p, True, np.uint32, True)

#-------------------------------
#k-space
#2D FFT
print("Creating kSpace")
fftLena = fftpack.fft2(lena) #the '2' is important
fftLenaShifted = fftpack.fftshift(fftLena)
#power spectrum
powSpectLena = np.abs(fftLenaShifted)

#add noise to kSpace
noise = finite.noise(fftLenaShifted, SNR)
if addNoise:
    fftLenaShifted += noise

#Recover full image with noise
print("Actual noisy image")
reconLena = fftpack.ifft2(fftLenaShifted) #the '2' is important
reconLena = np.abs(reconLena)
reconNoise = lena - reconLena

mse = imageio.immse(lena, np.abs(reconLena))
ssim = imageio.imssim(lena.astype(float), np.abs(reconLena).astype(float))
psnr = imageio.impsnr(lena, np.abs(reconLena))
print("Acutal RMSE:", math.sqrt(mse))
print("Acutal SSIM:", ssim)
print("Acutal PSNR:", psnr)

#compute lines
centered = True
subsetsLines = []
subsetsMValues = []
mu = 0
for angles in subsetsAngles:
    lines = []
    mValues = []
    for angle in angles:
        m, inv = farey.toFinite(angle, p)
        u, v = radon.getSliceCoordinates2(m, powSpectLena, centered, p)
        lines.append((u,v))
        mValues.append(m)
        #second quadrant
        if twoQuads:
            if m != 0 and m != p: #dont repeat these
                m = p-m
                u, v = radon.getSliceCoordinates2(m, powSpectLena, centered, p)
                lines.append((u,v))
                mValues.append(m)
    subsetsLines.append(lines)
    subsetsMValues.append(mValues)
    mu += len(lines)
print("Number of lines:", mu)
print(subsetsMValues)

#samples used
sampleNumber = (p-1)*mu
print("Samples used:", sampleNumber, ", proportion:", sampleNumber/float(N*N))

#-------------
# Measure finite slice
from scipy import ndimage

print("Measuring slices")
drtSpace = np.zeros((p+1, p), floatType)
for lines, mValues in zip(subsetsLines, subsetsMValues):
    for i, line in enumerate(lines):
        u, v = line
        sliceReal = ndimage.map_coordinates(np.real(fftLenaShifted), [u,v])
        sliceImag = ndimage.map_coordinates(np.imag(fftLenaShifted), [u,v])
        slice = sliceReal+1j*sliceImag
    #    print("slice", i, ":", slice)
        finiteProjection = fftpack.ifft(slice) # recover projection using slice theorem
        drtSpace[mValues[i],:] = finiteProjection
#print("drtSpace:", drtSpace)

#-------------------------------
#define MLEM
def ossirt_expand_complex(iterations, t, p, g_j, os_mValues, projector, backprojector, image, mask, epsilon=1e3, dtype=np.int32):
    '''
    # Gary's implementation
    # From Lalush and Wernick;
    # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
    # where g = \sum (h f^\hat)                                   ... (**)
    #
    # self.f is the current estimate f^\hat
    # The following g from (**) is equivalent to g = \sum (h f^\hat)
    '''
    norm = False
    center = False
    fdtype = floatType
    f = np.zeros((p,p), fdtype)
    
    mses = []
    psnrs = []
    ssims = []
    for i in range(0, iterations):
        print("Iteration:", i)
        for j, mValues in enumerate(os_mValues):
#            print("Subset:", j)
            muFinite = len(mValues)
            
            g = projector(f, p, fdtype, mValues)
        
            # form parenthesised term (g_j / g) from (*)
            r = np.zeros_like(g)
            for m in mValues:
#                r[m,:] = g_j[m,:] - g[m,:]
                for y in range(p):
                    r[m,y] = (g[m,y] - g_j[m,y]) / (muFinite*muFinite)
        
            # backproject to form \sum h * (g_j / g)
            g_r = backprojector(r, p, norm, center, 1, 0, mValues) / muFinite
        
            # Renormalise backprojected term / \sum h)
            # Normalise the individual pixels in the reconstruction
            f -= t * g_r
        
        if smoothReconMode > 0 and i % smoothIncrement == 0 and i > 0: #smooth to stem growth of noise
            if smoothReconMode == 1:
                print("Smooth TV")
                f = denoise_tv_chambolle(f, 0.02, multichannel=False)
            elif smoothReconMode == 2:
                h = parameters[4] #6, phantom; 4, camera
                if i > smoothMaxIteration:
                    h /= 2.0
                if i > smoothMaxIteration2:
                    h /= 4.0
                print("Smooth NL h:",h)
                fReal = denoise_nl_means(np.real(f), patch_size=3, patch_distance=7, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                fImag = denoise_nl_means(np.imag(f), patch_size=3, patch_distance=7, h=h, multichannel=False, fast_mode=True).astype(fdtype)
                f = fReal +1j*fImag
            elif smoothReconMode == 3:
                print("Smooth Median")
                f = ndimage.median_filter(f, 3)
            
        if i%plotIncrement == 0:
            img = imageio.immask(image, mask, N, N)
            recon = imageio.immask(f, mask, N, N)
            recon = np.abs(recon)
            mse = imageio.immse(img, recon)
            psnr = imageio.impsnr(img, recon)
            ssim = imageio.imssim(img.astype(float), recon.astype(float))
            print("RMSE:", math.sqrt(mse), "PSNR:", psnr, "SSIM:", ssim)
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
        
    return f, mses, psnrs, ssims

#-------------------------------
#reconstruct test using MLEM   
start = time.time() #time generation
recon, mses, psnrs, ssims = ossirt_expand_complex(iterations, t, p, drtSpace, subsetsMValues, finite.frt_complex, finite.ifrt_complex, lena, mask)
recon = np.abs(recon)
print("Done")
end = time.time()
elapsed = end - start
print("OSSIRT Reconstruction took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

mse = imageio.immse(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
ssim = imageio.imssim(imageio.immask(lena, mask, N, N).astype(float), imageio.immask(recon, mask, N, N).astype(float))
psnr = imageio.impsnr(imageio.immask(lena, mask, N, N), imageio.immask(recon, mask, N, N))
print("RMSE:", math.sqrt(mse))
print("SSIM:", ssim)
print("PSNR:", psnr)
diff = lena - recon

#save mat file of result
#np.savez('result_ossirt.npz', recon=recon, diff=diff, psnrs=psnrs, ssims=ssims)
np.savez('result_phantom_ossirt.npz', recon=recon, diff=diff, psnrs=psnrs, ssims=ssims)
#np.savez('result_camera_ossirt.npz', recon=recon, diff=diff, psnrs=psnrs, ssims=ssims)

#plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#pp = PdfPages('finite_ossirt_plots.pdf')
pp = PdfPages('finite_ossirt_phantom_plots.pdf')
#pp = PdfPages('finite_ossirt_camera_plots.pdf')

fig, ax = plt.subplots(figsize=(24, 9))

if plotCroppedImages:
    print(lena.shape)
    print(mask.shape)
    lena = imageio.immask(lena, mask, N, N)
    reconLena = imageio.immask(reconLena, mask, N, N)
    reconNoise = imageio.immask(reconNoise, mask, N, N)
    recon = imageio.immask(recon, mask, N, N)
    diff = imageio.immask(diff, mask, N, N)

plt.subplot(121)
rax = plt.imshow(reconLena, interpolation='nearest', cmap='gray')
#rax = plt.imshow(reconLena, cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image (w/ Noise)')
plt.subplot(122)
rax2 = plt.imshow(recon, interpolation='nearest', cmap='gray')
#rax2 = plt.imshow(recon, cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Reconstruction')
pp.savefig()

fig, ax = plt.subplots(figsize=(24, 9))

plt.subplot(151)
#rax = plt.imshow(lena, interpolation='nearest', cmap='gray')
rax = plt.imshow(lena, cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image')
plt.subplot(152)
#rax = plt.imshow(reconLena, interpolation='nearest', cmap='gray')
rax = plt.imshow(reconLena, cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Image (w/ Noise)')
plt.subplot(153)
#rax = plt.imshow(reconNoise, interpolation='nearest', cmap='gray')
rax = plt.imshow(reconNoise, cmap='gray')
rcbar = plt.colorbar(rax, cmap='gray')
plt.title('Noise')
plt.subplot(154)
#rax2 = plt.imshow(recon, interpolation='nearest', cmap='gray')
rax2 = plt.imshow(recon, cmap='gray')
rcbar2 = plt.colorbar(rax2, cmap='gray')
plt.title('Reconstruction')
plt.subplot(155)
#rax3 = plt.imshow(diff, interpolation='nearest', cmap='gray', vmin=-24, vmax=24)
rax3 = plt.imshow(diff, cmap='gray')
rcbar3 = plt.colorbar(rax3, cmap='gray')
plt.title('Reconstruction Errors')
pp.savefig()

#plot convergence
fig, ax = plt.subplots(figsize=(24, 9))

mseValues = np.array(mses)
psnrValues = np.array(psnrs)
ssimValues = np.array(ssims)
incX = np.arange(0, len(mses))*plotIncrement

plt.subplot(131)
plt.plot(incX, np.sqrt(mseValues))
plt.title('Error Convergence of the Finite OSSIRT')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.subplot(132)
plt.plot(incX, psnrValues)
plt.ylim(0, 40.0)
plt.title('PSNR Convergence of the Finite OSSIRT')
plt.xlabel('Iterations')
plt.ylabel('PSNR')
plt.subplot(133)
plt.plot(incX, ssimValues)
plt.ylim(0, 1.0)
plt.title('Simarlity Convergence of the Finite OSSIRT')
plt.xlabel('Iterations')
plt.ylabel('SSIM')
pp.savefig()
pp.close()

plt.show()

print("Complete")
