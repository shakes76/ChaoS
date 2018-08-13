'''
Discrete Radon Transforms module

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
import numpy as np
import finitetransform.numbertheory as nt #local modules
import random

#-------------
# project arrays
def drt(image, p, dtype=np.int32):
    '''
    Compute the Finite/Discrete Radon Transform of Grigoryan, Gertner, Fill and others using a polynomial time method.
    Returns the bins 'image'.
    Method is integer-only and doesn't need floats. Only uses additions and supports prime sizes.
    '''
    y = index = row = 0
    bins = np.zeros( (p+1, p), dtype )
    
    cimage = image.ravel() #get c-style array of image
    cbins = bins.ravel() #get c-style array of image
    
    for m in range(p):
        y = 0
        for x in range(p):
            row = x*p
            for t in range(p):
                index = row + (t + y)%p #Compute Column (in 1D)
                cbins[m*p+t] += cimage[index]
            y += m # Next pixel
            
    for t in range(p): #columns
        for y in range(p): #translates
            cbins[p*p+t] += cimage[t*p+y]
    
    return bins
    
def idrt(bins, p, norm = True):
    '''
    Compute the inverse Finite/Discrete Radon Transform of Grigoryan, Gertner, Fill and others using a polynomial time method.
    Returns the recovered image.
    Method is exact and integer-only and doesn't need floats
    '''
    y = index = row = 0
    image = np.zeros( (p, p) )
    
    cimage = image.ravel() #get c-style array of image
    cbins = bins.ravel() #get c-style array of image
    
    Isum = 0
    for t in range(p):
        Isum += cbins[t]
        
    for m in range(p):
        y = 0
        for x in range(p):
            row = x*p
            for t in range(p):
                index = row + (t + y)%p #Compute Column (in 1D)
                cimage[m*p+t] += cbins[index]
            y += p-m # Next pixel
            
    for t in range(p): #columns
        for y in range(p): #translates
            cimage[t*p+y] += cbins[p*p+t]
            cimage[t*p+y] -= Isum
            if norm:
                cimage[t*p+y] /= p
    
    return image

#-------------
#fast versions with FFTs and NTTs
import scipy.fftpack as fftpack
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def frt(image, N, dtype=np.float64):
    '''
    Compute the DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be an NxN image to recover the DRT projections (bins).
    Float type is returned by default to ensure no round off issues.
    Assumes object is real valued
    '''
    mu = nt.integer(N+N/2)
    if N % 2 == 1: # if odd, assume prime
        mu = nt.integer(N+1)
        
    #FFT image
    fftLena = fftpack.fft2(image) #the '2' is important
    
    bins = np.zeros((mu,N),dtype=dtype)
    for m in range(0, mu):
        slice = getSlice(m, fftLena)
#        print slice
#        slice /= N #norm FFT
        projection = np.real(fftpack.ifft(slice))
        #Copy and norm
        for j in range(0, N):
            bins[m, j] = projection[j]
#        print projection - bins[m, :]
    
    return bins
        
def ifrt(bins, N, norm = True, center = False, projNumber = 0, Isum = -1):
    '''
    Compute the inverse DRT in O(n logn) complexity using the discrete Fourier slice theorem and the FFT.
    Input should be DRT projections (bins) to recover an NxN image.
    projNumber is the number of non-zero projections in bins. This useful for backprojecting mu projections where mu < N.
    Isum is computed from first row if -1, otherwise provided value is used
    '''
    if Isum < 0:
        Isum = bins[0,:].sum()
#    print "ISUM:", Isum
    
    if projNumber == 0:
        projNumber = N + N/2 #all projections filled 
    result = np.zeros((N,N),dtype=np.complex64)
    filter = oversampling_1D_filter(N,2,norm) #fix DC for dyadic
    
    if N % 2 == 1: # if odd, assume prime
        if projNumber == 0:
            projNumber = N + 1 #all projections filled 
        filter = np.ones(N)
#        filter[0] = 1.0/(projNumber+1) #DC fix
        filter[0] = 1.0
        
    if projNumber < 0:
        filter[0] = 0 #all projections zero mean 
    
    #Set slices (0 <= m <= N)
    for k, row in enumerate(bins): #iterate per row
        slice = fftpack.fft(row)
        slice *= filter
#        print "m:",k
        setSlice(k,result,slice)
#    print "filter:", filter
    result[0,0] -= float(Isum)*N

    #iFFT 2D image
    result = fftpack.ifft2(result)
    if not norm:
        result *= N #ifft2 already divides by N**2
    if center:
        result = fftpack.fftshift(result)

    return np.real(result)

#-------------
#helper functions
def getSlice(m, data, center=False):
    '''
    Get the slice (at m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays.
    '''
    rows, cols = data.shape
    p = rows #base of image size, prime size this is p

    slice = np.zeros(rows, dtype=data.dtype)
    if m < cols and m >= 0:
#        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
        for k in range(0,cols):
            index = (rows-(k*m)%rows)%rows
            slice[k] = data[index,k]
    else: #perp slice, assume dyadic
        s = m #consistent notation
#        for k, row in enumerate(data): #iterate per row
        for k in range(0,rows):
            index = (cols-(k*p*s)%cols)%cols
            slice[k] = data[k,index]
            
    if center:
        sliceCentered = np.copy(slice)
        offset = int(rows/2.0)
        for index, value in enumerate(sliceCentered):
            newIndex = (index+offset)%rows
            slice[newIndex] = value

    return slice
    
def getSliceCoordinates(m, data, center=False, b=0):
    '''
    Get the slice coordinates u, v arrays (in pixel coordinates at finite angle m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays and is good for drawing sample points of the slice on plots.
    b can be used to stop the slice earlier, say sampling b<p.
    '''
    rows, cols = data.shape
    p = rows #base of image size, prime size this is p
    B = (b-1)/2
    
    offset = 0
    if center:
        offset = int(rows/2.0)
#    print "offset:", offset

    u = []
    v = []
    if m < cols and m >= 0:
        extentMax = (rows-(B*m)%rows + offset)%rows
        extentMin = (rows-(B*m)%rows - offset)%rows
#        print "extentMax:", extentMax, "extentMin:", extentMin
        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
            x = (k+offset)%rows
            index = (rows-(k*m)%rows + offset)%rows
            if b > 0 and (index > extentMax or index < extentMin):
                continue
            v.append(x)
            u.append(index)
    else: #perp slice, assume dyadic
        s = m #consistent notation
        extentMax = ((cols-B*2*s)%cols + cols + offset)%cols
        extentMin = ((cols-B*2*s)%cols + cols - offset)%cols
#        print "extentMax:", extentMax, "extentMin:", extentMin
        for k, row in enumerate(data): #iterate per row
            x = (k+offset)%cols
            index = ((cols-k*p*s)%cols + cols + offset)%cols
            if b > 0 and (index > extentMax or index < extentMin):
                continue
            v.append(x)
            u.append(index)

    return np.array(u), np.array(v)
    
def getSliceCoordinates2(m, data, center=False, p=2):
    '''
    Get the slice coordinates u, v arrays (in pixel coordinates at finite angle m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays and is good for drawing sample points of the slice on plots.
    '''
    rows, cols = data.shape
    
    offset = 0
    if center:
        offset = int(rows/2.0)
#    print "offset:", offset

    u = []
    v = []
    if m < cols and m >= 0:
        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
            x = (k+offset)%rows
            index = (rows-(k*m)%rows + offset)%rows
#            print "k:",k,"\tindex:",index
            v.append(x)
            u.append(index)
    else: #perp slice, assume dyadic
        s = m-cols #consistent notation
#        print "s:",s
        for k, row in enumerate(data): #iterate per row
            x = (k+offset)%cols
            index = (cols-(k*p*s)%cols + offset)%cols
#            print "k:",k,"\tindex:",index
            u.append(x)
            v.append(index)

    return np.array(u), np.array(v)

def getProjectionCoordinates(m, data, center=False, p=2):
    '''
    Get the projection coordinates x, y arrays (in pixel coordinates at finite angle m) of a NxN discrete array using the discrete Radon transform.
    This can be applied to the image arrays and is good for drawing sample points of the slice on plots.
    '''
    rows, cols = data.shape
    
    offset = 0
    if center:
        offset = int(rows/2)
#    print "offset:", offset

    x = []
    y = []
    if m < cols and m >= 0:
        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
            u = (k+rows-offset)%rows
            index = (k%rows*m + cols - offset)%cols
#            print "u:", u, "v:", index
            x.append(u)
            y.append(index)
    else: #perp slice, assume dyadic
        s = m #consistent notation
        for k, row in enumerate(data): #iterate per row
            v = (k+cols-offset)%cols
            index = (k*p*s + rows - offset)%rows
            y.append(v)
            x.append(index)

    return np.array(x), np.array(y)
    
def setSlice(m, data, slice):
    '''
    Set the slice (at m) of a NxN discrete array using the discrete Fourier slice theorem.
    This can be applied to the DFT or the NTT arrays.
    '''
    rows, cols = data.shape
    p = rows #base of image size, prime size this is p
    
    if m < cols and m >= 0:
#        for k, col in enumerate(data.T): #iterate per column, transpose is cheap
        for k in range(0,cols):
            index = (rows-(k*m)%rows)%rows
#            print "k:",k,"\tindex:",index
            data[index,k] += slice[k]
    else: #perp slice, assume dyadic
        s = m-cols #consistent notation
#        print "s:",s
#        for k, row in enumerate(data): #iterate per row
        for k in range(0,rows):
            index = (cols-(k*p*s)%cols)%cols
#            print "k:",k,"\tindex:",index
            data[k,index] += slice[k]
    
#-------------
#filters
def oversampling(n, p = 2):
    '''
    Produce the nxn oversampling filter that is needed to exactly filter dyadic DRT etc.
    '''
    gcd_table = np.zeros(n)

    gcd_table[0] = n + n/p
    gcd_table[1] = 1
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )

    filter = np.zeros((n,n))
    for j in range(0,n):
        for k in range(0,n):
            if gcd_table[j] < gcd_table[k]:
                filter[j,k] = gcd_table[j]
            else:
                filter[j,k] = gcd_table[k]

    return filter
    
def oversamplingFilter(n, M, p = 2):
    '''
    Produce the nxn oversampling filter that is needed to exactly filter dyadic DRT etc.
    This version returns values as multiplicative inverses (mod M) for use with the NTT
    '''
    gcd_table = np.zeros(n)
    gcdInv_table = np.zeros(n)

    gcd_table[0] = n + int(n)/p
    gcdInv_table[0] = nt.minverse(gcd_table[0], M)
    gcd_table[1] = 1
    gcdInv_table[1] = nt.minverse(gcd_table[1], M)
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )
        gcdInv_table[j] = nt.minverse(gcd_table[j], M)

    filter = nt.zeros((n,n))
    for j in range(0,n):
        for k in range(0,n):
            if gcd_table[j] < gcd_table[k]:
                filter[j,k] = nt.integer(gcdInv_table[j])
            else:
                filter[j,k] = nt.integer(gcdInv_table[k])

    return filter

def oversampling_1D(n, p = 2, norm = False):
    '''
    The 1D oversampling. All values are GCDs.
    Use the filter versions to remove the oversampling.
    '''
    gcd_table = np.zeros(n)

    gcd_table[0] = n + int(n)/p
    gcd_table[1] = 1
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )

    return gcd_table

def oversampling_1D_filter(n, p = 2, norm = False):
    '''
    The 1D filter for removing oversampling. All values are multiplicative inverses.
    To apply this filter multiply with 1D FFT slice
    '''
    normValue = 1
    if norm:
        normValue = n
        
    gcd_table = np.zeros(n)
    gcdInv_table = np.zeros(n)

    gcd_table[0] = n + int(n)/p
    gcdInv_table[0] = 1.0/gcd_table[0]
    gcd_table[1] = 1
    gcdInv_table[1] = 1.0/gcd_table[1]
    for j in range(2,n):
        u, v, gcd_table[j] = nt.extended_gcd( int(j), int(n) )
        gcdInv_table[j] = 1.0/(gcd_table[j]*normValue)

    return gcdInv_table
    
def convolve(x, y):
    '''
    Compute the nD convolution of two real arrays of the same size.
    '''
    xHat = fftpack.fftn(x+0j)
    yHat = fftpack.fftn(y+0j)
    xHat *= yHat
    
    return np.real(fftpack.ifftn(xHat))
