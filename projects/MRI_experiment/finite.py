# -*- coding: utf-8 -*-
"""
Finite measurement module for MRI data

Created on Tue Sep 27 13:57:13 2016

@author: uqscha22
"""
import _libpath #add custom libs
import finitetransform.radon as radon
import finitetransform.farey as farey #local module
from scipy import ndimage
import scipy.fftpack as fftpack
import pyfftw
import numpy as np
import math

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def computeLines(kSpace, angles, centered = True, twoQuads = False):
    '''
    compute finite lines coordinates
    Returns a list or list of slice 2-tuples and corresponding list of m values
    '''
    p, s = kSpace.shape
    lines = []
    mValues = []
    for angle in angles:
        m, inv = farey.toFinite(angle, p)
        u, v = radon.getSliceCoordinates2(m, kSpace, centered, p)
        lines.append((u,v))
        mValues.append(m)
        #second quadrant
        if twoQuads:
            if m != 0 and m != p: #dont repeat these
                m = p-m
                u, v = radon.getSliceCoordinates2(m, kSpace, centered, p)
                lines.append((u,v))
                mValues.append(m)
    
    return lines, mValues
    
# Measure finite slices
def measureSlices(dftSpace, lines, mValues, dtype=np.float64):
    '''
    Measure finite slices of the DFT and convert to FRT projections
    Returns FRT projections
    '''
    p, s = dftSpace.shape
    drtSpace = np.zeros((p+1, p), dtype=dtype)
    for i, line in enumerate(lines):
        u, v = line
        sliceReal = ndimage.map_coordinates(np.real(dftSpace), [u,v])
        sliceImag = ndimage.map_coordinates(np.imag(dftSpace), [u,v])
        slice = sliceReal+1j*sliceImag
        finiteProjection = np.real(fftpack.ifft(slice)) # recover projection using slice theorem
        drtSpace[mValues[i],:] = finiteProjection
        
    return drtSpace
    
def measureSlices_complex(dftSpace, lines, mValues):
    '''
    Measure finite slices of the DFT and convert to FRT projections
    Returns FRT projections
    '''
    p, s = dftSpace.shape
    drtSpace = np.zeros((p+1, p), dtype=np.complex64)
    for i, line in enumerate(lines):
        u, v = line
        sliceReal = ndimage.map_coordinates(np.real(dftSpace), [u,v])
        sliceImag = ndimage.map_coordinates(np.imag(dftSpace), [u,v])
        slice = sliceReal+1j*sliceImag
        finiteProjection = fftpack.ifft(slice) # recover projection using slice theorem
        drtSpace[mValues[i],:] = finiteProjection
        
    return drtSpace

def frt(image, N, dtype=np.float32, mValues=None):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins), where N is prime.
    Float type is returned by default to ensure no round off issues.
    '''
    mu = N+1
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        if mValues and m not in mValues:
            continue 
        
        slice = radon.getSlice(m, fftLena)
#        print slice
#        slice /= N #norm FFT
        projection = np.real(fftpack.ifft(slice))
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
    
    return bins
    
def frt_complex(image, N, dtype=np.complex, mValues=None, center=False):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins), where N is prime.
    Float type is returned by default to ensure no round off issues.
    '''
    mu = N+1
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        if mValues and m not in mValues:
            continue 
        
        slice = radon.getSlice(m, fftLena)
#        print slice
#        slice /= N #norm FFT
        projection = fftpack.ifft(slice)
        if center:
            projection = fftpack.ifftshift(projection)
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
    
    return bins

def ifrt(bins, N, norm = True, center = False, projNumber = 0, Isum = -1, mValues=None):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image, where N is prime.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum
    dftSpace = np.zeros((N,N),dtype=np.complex)
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        if mValues and k not in mValues:
            continue
        
        slice = fftpack.fft(row)
        radon.setSlice(k,dftSpace,slice)
#    print "filter:", filter
    dftSpace[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(dftSpace)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.fftshift(result)

    return np.real(result)
    
def ifrt_complex(bins, N, norm = True, center = False, projNumber = 0, Isum = -1, mValues=None):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image, where N is prime.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum
    dftSpace = np.zeros((N,N),dtype=np.complex)
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        if mValues and k not in mValues:
            continue
        
        slice = fftpack.fft(row)
        radon.setSlice(k,dftSpace,slice)
#    print "filter:", filter
    dftSpace[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(dftSpace)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.ifftshift(result)

    return result

def mse(img1, img2):
    '''
    Compute the MSE of two images using mask if given
    '''
    error = ((img1 - img2) ** 2).mean(axis=None)
    
    return error
    
def psnr(img1, img2, maxPixel=255):
    '''
    Compute the MSE of two images using mask if given
    '''
    error = mse(img1,img2)
    psnr_out = 20 * math.log(maxPixel / math.sqrt(error), 10)
    
    return psnr_out

#import random

def noise(kSpace, snr):
    '''
    Create noise in db for given kSpace and SNR
    '''
    r, s = kSpace.shape
    #pwoer of signal and noise
    P = np.sum(np.abs(kSpace)**2)/(r*s)
    P_N = P / (10**(snr/10))
    #P_N is equivalent to sigma**2 and signal usually within 3*sigma
    sigma = math.sqrt(P_N)
    
    noise = np.zeros_like(kSpace)
    for u, row in enumerate(kSpace):
        for v, coeff in enumerate(row):
            noiseReal = np.random.normal(0, sigma)
            noiseImag = np.random.normal(0, sigma)
            noise[u,v] = noiseReal + 1j*noiseImag
            
    return noise
