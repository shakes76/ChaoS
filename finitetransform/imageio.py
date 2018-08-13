'''
This module extends scipy modules for saving and loading images losslessly.
It ensures files are written and arrays returned are in 32-bit integer format
PIL Modes: L - 8 bit, I - 32 bit, RGB, RGBA, 1 - 1 bit

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
'''
from PIL import Image
import numpy as np
import numpy.ma as ma
import scipy.misc
from skimage.measure import compare_ssim as ssim
import os
import math

package_directory = os.path.dirname(os.path.abspath(__file__))

def imread(name):
    '''
    Load image into an array losslessly for 32-bit integers.
    Returns the numpy array object created
    '''
    im = Image.open(name)
    arr = scipy.misc.fromimage(im)
    
    return arr
    
def imsave(name, arr, datamode=False):
    '''
    Save an array to an image file losslessly for 32-bit integers.
    Returns the PIL object used. Data Mode is whether to use large integers
    in the file rather than 8-bit integers. The former will not be displayable
    via standard image viewers
    '''
    if datamode:
        im = scipy.misc.toimage(arr.astype(int), high=np.max(arr), low=np.min(arr), mode='I')
    else:
        im = scipy.misc.toimage(arr.astype(int), high=np.max(arr), low=np.min(arr))

    im.save(name)
    
    return im
    
def imcrop(img, n, m=0, center=True, out_dtype=np.uint32):
    '''
    Given img, crop or pad image given square size m
    '''
    lx, ly = img.shape
    mid = int(n/2.0)
    midx = int(lx/2.0+0.5)
    midy = int(ly/2.0+0.5)
    newLengthX = midx - mid
    if n % 2 == 1:#prime size
        newLengthY = midx + mid + 1
    else:
        newLengthY = midy + mid        
    
    crop_img = img[newLengthX:newLengthY, newLengthX:newLengthY]
    
    if m > 0:
        #zero pad to larger prime size to allow non-periodic rotations.
        image = np.zeros((m, m), dtype=out_dtype)
        midx = int(m/2.0+0.5)
        midy = int(m/2.0+0.5)
        if center:
            newLengthX = midx - mid
            if n % 2 == 1:#prime size
                newLengthY = midy + mid + 1
            else:
                newLengthY = midy + mid
        else:
            newLengthX = 0
            newLengthY = n
        image[newLengthX:newLengthY, newLengthX:newLengthY] = crop_img.astype(out_dtype)
    else:
        image = crop_img.astype(out_dtype)
        
    return image
    
def immask(img, mask, P, Q):
    '''
    Mask and crop an image given mask image
    '''
    #print img[mask].shape
    #return img[mask].reshape((P,Q))
    
    maskArray = ma.make_mask(mask)
    imgMasked = ma.array(img, mask=~maskArray)
    
    return imgMasked.compressed().reshape((P,Q))
    
def immse(img1, img2):
    '''
    Compute the MSE of two images
    '''
    mse = ((img1 - img2) ** 2).mean(axis=None)
    
    return mse
    
def imssim(img1, img2):
    '''
    Compute the SSIM of two images
    '''
    return ssim(img1, img2, data_range=img1.max() - img1.min())
    
def impsnr(img1, img2, maxPixel=255):
    '''
    Compute the MSE of two images
    '''
    mse = immse(img1,img2)
    psnr_out = 20 * math.log(maxPixel / math.sqrt(mse), 10)
    
    return psnr_out

def lena(n, m=0, center=True, out_dtype=np.uint32, mask=False):
    '''
    Return image of lena as numpy array of size nx, ny from the centre.
    Image will be padded to m if provided
    
    Example: crop_lena = imageio.lena(p)
    '''
    #~ lena = scipy.misc.lena() # load lena
    lena = imread(os.path.join(package_directory, '../data', 'lena.png'))
    maskImg = np.ones_like(lena, out_dtype)
    
    if mask:
        return imcrop(lena, n, m, center, out_dtype), imcrop(maskImg, n, m, center, out_dtype)
    else:
        return imcrop(lena, n, m, center, out_dtype)
    
def phantom(n, m=0, center=True, out_dtype=np.uint32, mask=False):
    '''
    Return image of the Shepp Logan phantom as numpy array of size nx, ny from the centre.
    Image will be padded to m if provided
    
    Example: crop_phantom = imageio.phantom(p)
    '''
    phan = imread(os.path.join(package_directory, '../data', 'phantom.png'))
    maskImg = np.ones_like(phan, out_dtype)
    
    if mask:
        return imcrop(phan, n, m, center, out_dtype), imcrop(maskImg, n, m, center, out_dtype)
    else:
        return imcrop(phan, n, m, center, out_dtype)
        
def phantom_complex(n, m=0, center=True, norm=False, mask=False):
    '''
    Return image of the Shepp Logan phantom as numpy array of size nx, ny from the centre.
    Image will be padded to m if provided
    
    Example: crop_phantom = imageio.phantom(p)
    '''
    if mask:
        phan, maskImg = phantom(n, m, center, np.float64, mask)
    else:
        phan = phantom(n, m, center, np.float64)
        
    if norm:
        maxValue = np.max(phan)
        phan /= maxValue
    
    phan_complex = np.zeros(phan.shape, np.complex64)
    phan_complex = phan + 1j
    
    if mask:
        return phan_complex, maskImg
    else:
        return phan_complex
    
def cameraman(n, m=0, center=True, out_dtype=np.uint32, mask=False):
    '''
    Return image of the Shepp Logan phantom as numpy array of size nx, ny from the centre.
    Image will be padded to m if provided
    
    Example: crop_phantom = imageio.phantom(p)
    '''
    cam = imread(os.path.join(package_directory, '../data', 'camera.png'))
    maskImg = np.ones_like(cam, out_dtype)
    
    if mask:
        return imcrop(cam, n, m, center, out_dtype), imcrop(maskImg, n, m, center, out_dtype)
    else:
        return imcrop(cam, n, m, center, out_dtype)
    
def ones(n, m=0, center=True, out_dtype=np.uint32):
    '''
    Ones image to match the padding of the lena() and phantom() functions
    Useful as a mask for above functions
    '''
    lena = imread(os.path.join(package_directory, '../data', 'lena.png'))
    maskImg = np.ones_like(lena, out_dtype)

    return imcrop(maskImg, n, m, center, out_dtype)
    
def mask(n, m=0, center=True, out_dtype=np.uint32):
    '''
    Mask image to match the padding of the lena() and phantom() functions
    '''
    return ones(n, m, center, out_dtype).astype(bool)
    
def readPGM(name):
    '''
    Read a PGM image, where the PGM format is that of the FTL library for NTTs and FRTs.
    This is a slightly modified format where values are not limited to 8-bit values.
    
    Returns array of image and bit depth (image, depth)
    '''
    inFile = open(name,"r")
    
    #read header
    formatLine = inFile.readline()
    commentLine = inFile.readline()
    
    if not "P2" in formatLine:
        print("Error: PGM not in correct format (P2)")
    print("Comment:", commentLine)
    
    width, height = [int(x) for x in inFile.readline().split()] # read dimensions
    print("PGM Size:", width, "x", height)
    
    bitDepth = [int(x) for x in inFile.readline().split()] # read bit Depth
    
    imageList = []
    for line in inFile: # read remaining lines
        valueList = [int(x) for x in line.split()] #read integers on each line
        for value in valueList:
            imageList.append(value) #append as 1D list
#    print imageList
    #store as array
    image = np.array(imageList).reshape(height, width)
    
    return image, bitDepth
