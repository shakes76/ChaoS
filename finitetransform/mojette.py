# -*- coding: utf-8 -*-
"""
Python module for computing methods related to the Mojette transform.

The transform (resulting in projections) is computed via the 'transform' member.

Assumes coordinate system with rows as x-axis and cols as y-axis. Thus angles are taken as complex(q,p) with q in the column direction.
Use the farey module to generate the angle sets.

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
import finitetransform.farey as farey #local module
import finitetransform.radon as radon
import numpy as np
import scipy.fftpack as fftpack
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def projectionLength(angle, P, Q):
    '''
    Return the number of bins for projection at angle of a PxQ image.
    Wraps function from Farey module
    '''
    return farey.projectionLength(angle, P, Q) #no. of bins
    
def toFinite(fareyVector, N):
    '''
    Return the finite vector corresponding to the Farey vector provided for a given modulus/length N
    and the multiplicative inverse of the relevant Farey angle
    Wraps function from Farey module
    '''
    return farey.toFinite(fareyVector, N)
    
def finiteTranslateOffset(fareyVector, N, P, Q):
    '''
    Translate offset required when mapping Farey vectors to finite angles
    Returns translate offset and perp Boolean flag pair
    Wraps function from Farey module
    '''
    return farey.finiteTranslateOffset(fareyVector, N, P, Q)

def isKatzCriterion(P, Q, angles, K = 1):
    '''
    Return true if angle set meets Katz criterion for exact reconstruction of
    discrete arrays
    '''
    sumOfP = 0
    sumOfQ = 0
    n = len(angles)
    for j in range(0, n):
        p, q = farey.get_pq(angles[j])
        sumOfP += abs(p)
        sumOfQ += abs(q)
        
#    if max(sumOfP, sumOfQ) > max(rows, cols):
    if sumOfP > K*P or sumOfQ > K*Q:
        return True
    else:
        return False

def project(image, q, p, dtype=np.int32):
    '''
    Projects an array at rational angle given by p, q.
    
    Returns a list of bins that make up the resulting projection
    '''
    offsetMojette = 0
    rows, cols = image.shape
    totalBins = abs(q)*(rows-1) + abs(p)*(cols-1) + 1
#    print "Projection (%d, %d) has %d bins" % (q, p, totalBins)
    
    if q*p >= 0: #If positive slope
        offsetMojette = p*(rows-1)

    projection = np.zeros(totalBins, dtype)
    for x in range(0, rows):
        for y in range(0, cols):
            if q*p >= 0:
                translateMojette = q*y - p*x + offsetMojette #GetY = q, GetX = p
            else:
                translateMojette = p*x - q*y; #GetY = q, GetX = p
#            print "t:", translateMojette, "x:", x, "y:", y, "p:", p, "q:", q
            projection[translateMojette] += image[x,y]
    
    return projection
    
def transform(image, angles, dtype=np.int32, prevProjections = []):
    '''
    Compute the Mojette transform for a given angle set and return list of projections.
    
    The angle set is assumed to be a list of 2-tuples as (q, p). Returns a list of projections.
    Previous projections can be provided (for use in iterative reconstruction methods), but must be of correct size.
    '''
    mu = len(angles)

    #Compute Mojette
    projections = []
    for n in range(0, mu):
        p = int(angles[n].imag)
        q = int(angles[n].real)
        projection = project(image, q, p, dtype)
        if not prevProjections:
            projections.append(projection)
        else:
            projections.append(projection+prevProjections[n])
        
    return projections

def backproject(projections, angles, P, Q, norm = True, dtype=np.int32, prevImage = np.array([])):
    '''
    Directly backprojects (smears) a set of projections at rational angles given by angles in image space (PxQ).
    
    Returns an image of size PxQ that makes up the reconstruction
    '''
    image = np.zeros((P,Q),dtype)

    normValue = 1.0
    if norm:
        normValue = float(len(angles))
    
    for projection, angle in zip(projections, angles):
        p = int(angle.imag)
        q = int(angle.real)
        offsetMojette = 0
        if q*p >= 0: #If positive slope
            offsetMojette = p*(Q-1)
        for x in range(0, P):
            for y in range(0, Q):
                if q*p >= 0:
                    translateMojette = q*y - p*x + offsetMojette #GetY = q, GetX = p
                else:
                    translateMojette = p*x - q*y #GetY = q, GetX = p
    #            print "t:", translateMojette, "x:", x, "y:", y, "p:", p, "q:", q
                prevValue = 0
                if prevImage.size > 0:
                    prevValue = prevImage[x,y]
                    
                try:
                    image[x,y] += projection[translateMojette]/normValue + prevValue
                except IndexError:
                    image[x,y] += 0 + prevValue
    
    return image
    
def finiteProjection(projection, angle, P, Q, N, center=False):
    '''
    Convert a Mojette projection taken at angle into a finite (FRT) projection.
    '''
    dyadic = True
    if N % 2 == 1: # if odd, assume prime
        dyadic = False    
    shiftQ = int(N/2.0+0.5)-int(Q/2.0+0.5)
    shiftP = int(N/2.0+0.5)-int(P/2.0+0.5)
    
    finiteProj = np.zeros(N)
    p, q = farey.get_pq(angle)
    m, inv = farey.toFinite(angle, N)
#    print "p:", p, "q:", q, "m:", m, "inv:", inv
    translateOffset, perp = farey.finiteTranslateOffset(angle, N, P, Q)
    angleSign = p*q    
    
    if dyadic:    
        for translate, bin in enumerate(projection):
            if angleSign >= 0 and perp: #Reverse for perp
                translateMojette = translateOffset - translate
            else:
                translateMojette = translate - translateOffset
            translateFinite = (inv*translateMojette)%N
            if center:
                translateFinite = (translateFinite + shiftQ + m*(N-shiftP))%N
            finiteProj[translateFinite] += bin
    else:
        for translate, bin in enumerate(projection):
            if angleSign >= 0 and perp: #Reverse for perp
                translateMojette = int(translateOffset) - int(translate)
            else:
                translateMojette = int(translate) - int(translateOffset)
                
            if translateMojette < 0:
                translateFinite = ( N - ( inv*abs(translateMojette) )%N )%N
            else:         
                translateFinite = (inv*translateMojette)%N #has issues in C, may need checking
            if center:
                translateFinite = (translateFinite + shiftQ + m*(N-shiftP))%N
            finiteProj[translateFinite] += bin
        
    return finiteProj

#inversion methods
def toDRT(projections, angles, N, P, Q, center=False):
    '''
    Convert the Mojette (asymetric) projection data to DRT (symetric) projections.
    Use the iFRT to reconstruct the image. Requires N+1 or N+N/2 projections if N is prime or dyadic respectively.
    Returns the resulting DRT space as a 2D array
    '''
    size = int(N + N/2)
    dyadic = True
    if N % 2 == 1: # if odd, assume prime
        size = int(N+1)
        dyadic = False
        
    m = 0
    
    frtSpace = np.zeros( (size,N) )
    
    if dyadic:
        print("Dyadic size not tested yet.")
        #for each project
        '''for index, proj in enumerate(projections):
            p, q = farey.get_pq(angles[index])
            
            m, inv = farey.toFinite(angles[index], N)

            frtSpace[m][:] = finiteProjection(proj, angles[index], P, Q, N, center)'''
        
    else: #prime size
        for index, proj in enumerate(projections):
            p, q = farey.get_pq(angles[index])
            
            m, inv = farey.toFinite(angles[index], N)

            frtSpace[m][:] = finiteProjection(proj, angles[index], P, Q, N, center)
    
    return frtSpace

#helper functions
def discreteSliceSamples(angle, b, fftShape):
    '''
    Generate the b points along slice at angle of DFT space with shape.
    '''
    r, s = fftShape
    q = farey.getX(angle)
    p = farey.getY(angle)
    u = []
    v = []
    
    u.append(0 + r/2)
    v.append(0 + s/2)
    for m in range(1, b/4):
         u.append(p*m + r/2)
         v.append(q*m + s/2)
    for m in range(-b/4, 1):
         u.append(p*m + r/2)
         v.append(q*m + s/2)
#    print "u:",u
#    print "v:",v
    return u, v
    
def sliceSamples(angle, b, fftShape, center=False):
    '''
    Generate the b points along slice at angle of DFT space with shape.
    '''
    r, s = fftShape
    p, q = farey.get_pq(angle)
    u = []
    v = []
    offsetU = 0
    offsetV = 0
    if center:
        offsetU = r/2
        offsetV = s/2
#    increment = 1.0/math.sqrt(p**2+q**2)
    
    u.append(0 + offsetU)
    v.append(0 + offsetV)
    for m in range(1, (b-1)/2):
         u.append((1.0/p)*m + offsetU)
         v.append(-(1.0/q)*m + offsetV)
    for m in range(-(b-1)/2, 1):
#        print "m:", m, "delP:", -(1.0/p)*m + offsetU, "delQ:", (1.0/q)*m + offsetV
        u.append((1.0/p)*m + offsetU)
        v.append(-(1.0/q)*m + offsetV)
#    print "u:",u
#    print "v:",v
    return u, v

#angle sets
def angleSet_ProjectionLengths(angles, P, Q):
    '''
    Returns a matching list of projection lengths for each angle in set
    '''
    binLengthList = []
    for angle in angles:
        binLengthList.append(projectionLength(angle,P,Q))
        
    return binLengthList

def angleSet_Finite(p, quadrants=1, finiteList=False):
    '''
    Generate the minimal L1 angle set for the MT that has finite coverage.
    If quadrants is more than 1, two quadrants will be used.
    '''
    fareyVectors = farey.Farey()
    
    octants = 2
    if quadrants > 1:
        octants = 4
    if quadrants > 2:
        octants = 8
        
    fareyVectors.compactOn()
    fareyVectors.generate(p/2, octants)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2) #sort by L2 magnitude
    
    #compute corresponding m values
    finiteAngles = []
    for vector in sortedVectors:
        if vector.real == 0:
            m = 0
        elif vector.imag == 0:
            m = p
        else:
            m, inv = toFinite(vector, p)
        finiteAngles.append(m)
#        print("m:", m, "vector:", vector)
#    print("sortedVectors:", sortedVectors)
    #print(finiteAngles)
        
    #ensure coverage
    count = 0
    filled = [0]*(p+1) #list of zeros        
    finalVectors = []
    finalFiniteAngles = [] 
    for vector, m in zip(sortedVectors, finiteAngles):
        if filled[m] == 0:
            count += 1
            filled[m] = 1
            finalVectors.append(vector)
            finalFiniteAngles.append(m)
            
        if count == p+1:
            break
    
    if finiteList:
        return finalVectors, finalFiniteAngles
        
    return finalVectors
    
def angleSet_Symmetric(P, Q, octant=0, binLengths=False, K = 1):
    '''
    Generate the minimal L1 angle set for the MT.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is non-zero, full quadrant will be used. Octant schemes are as follows:
        If octant = -1, the opposing octant is also used.
        If octant = 0,1 (default), only use one octant.
        If octant = 2, octant will be mirrored from diagonal to form a quadrant.
        If octant = 4, 2 quadrants.
        If octant = 8, all quadrants.
    Function can also return bin lengths for each bin.
    '''
    angles = []
    fareyVectors = farey.Farey()
    maxPQ = max(P,Q)

    fareyVectors.compactOff()
    fareyVectors.generate(maxPQ-1, 1)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2) #sort by L2 magnitude
    
    index = 0
    binLengthList = []
    angles.append(sortedVectors[index])
    binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    while not isKatzCriterion(P, Q, angles, K) and index < len(sortedVectors): # check Katz
        index += 1
        angles.append(sortedVectors[index])
        p, q = farey.get_pq(sortedVectors[index]) # p = imag, q = real
        
        binLengthList.append(projectionLength(sortedVectors[index],P,Q))
        
#        if isKatzCriterion(P, Q, angles):
#            break
        
        if octant == 0:
            continue
        
        #add octants
        if octant == -1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 0 and p != q:
            nextOctantAngle = farey.farey(q, p) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
        if octant > 1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if p != q: #dont replicate
                nextOctantAngle = farey.farey(q, -p) #mirror from axis and swap to mirror from diagonal
                angles.append(nextOctantAngle)
                binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if octant > 1: #add the diagonal and column projections when symmetric (all quadrant are wanted)
        nextOctantAngle = farey.farey(1, 0) #mirror from axis
        angles.append(nextOctantAngle)
        binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if binLengths:
        return angles, binLengthList
    return angles
    
def angleSubSets_Symmetric(s, mode, P, Q, octant=0, binLengths=False, K = 1):
    '''
    Generate the minimal L1 angle set for the MT for s subsets.
    Parameter K controls the redundancy, K = 1 is minimal.
    If octant is non-zero, full quadrant will be used. Octant schemes are as follows:
        If octant = -1, the opposing octant is also used.
        If octant = 0,1 (default), only use one octant.
        If octant = 2, octant will be mirrored from diagonal to form a quadrant.
        If octant = 4, 2 quadrants.
        If octant = 8, all quadrants.
    Function can also return bin lengths for each bin.
    '''
    angles = []
    subsetAngles = []
    for i in range(s):
        subsetAngles.append([])
    fareyVectors = farey.Farey()
    maxPQ = max(P,Q)

    fareyVectors.compactOff()
    fareyVectors.generate(maxPQ-1, 1)
    vectors = fareyVectors.vectors
    sortedVectors = sorted(vectors, key=lambda x: x.real**2+x.imag**2) #sort by L2 magnitude
    
    index = 0
    subsetIndex = 0
    binLengthList = []
    angles.append(sortedVectors[index])
    subsetAngles[subsetIndex].append(sortedVectors[index])
    binLengthList.append(projectionLength(sortedVectors[index],P,Q))
    while not isKatzCriterion(P, Q, angles, K) and index < len(sortedVectors): # check Katz
        index += 1
        angles.append(sortedVectors[index])
        subsetAngles[subsetIndex].append(sortedVectors[index])
        p, q = farey.get_pq(sortedVectors[index]) # p = imag, q = real
        
        binLengthList.append(projectionLength(sortedVectors[index],P,Q))
        
#        if isKatzCriterion(P, Q, angles):
#            break
        
        if octant == 0:
            continue
        
        #add octants
        if octant == -1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s
        if octant > 0 and p != q:
            nextOctantAngle = farey.farey(q, p) #swap to mirror from diagonal
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s
        if octant > 1:
            nextOctantAngle = farey.farey(p, -q) #mirror from axis
            angles.append(nextOctantAngle)
            subsetAngles[subsetIndex].append(nextOctantAngle)
            binLengthList.append(projectionLength(nextOctantAngle,P,Q))
            if mode == 1:
                subsetIndex += 1
                subsetIndex %= s
            if p != q: #dont replicate
                nextOctantAngle = farey.farey(q, -p) #mirror from axis and swap to mirror from diagonal
                angles.append(nextOctantAngle)
                subsetAngles[subsetIndex].append(nextOctantAngle)
                binLengthList.append(projectionLength(nextOctantAngle,P,Q))
                if mode == 1:
                    subsetIndex += 1
                    subsetIndex %= s
                
        if mode == 0:
            subsetIndex += 1
            subsetIndex %= s
    
    if octant > 1: #add the diagonal and column projections when symmetric (all quadrant are wanted)
        nextOctantAngle = farey.farey(1, 0) #mirror from axis
        angles.append(nextOctantAngle)
        subsetAngles[0].append(nextOctantAngle)
        binLengthList.append(projectionLength(nextOctantAngle,P,Q))
    
    if binLengths:
        return angles, subsetAngles, binLengthList
    return angles, subsetAngles
    
def angleSetSliceCoordinates(angles, P, Q, N, center=False):
    '''
    Compute the 2D coordinates of each translate (in NxN DFT space) of every projection having angle in angles.
    Returns a list of u, v coordinate arrays [[u_0[...],v_0[...]], [u_1[...],v_1[...]], ...] per angle
    '''
    coords = []
    translateOffset = 0
    translateMojette = 0
    translateFinite = 0
    m = 0

    offset = 0.0
    if center:
        offset = N/2.0
    
    for index, angle in enumerate(angles):
        u = []
        v = []
        coordinateList = []
        p = int(angle.imag)
        q = int(angle.real)
        angleSign = p*q
        
        m, inv = farey.toFinite(angle, N)
        translateOffset, perp = farey.finiteTranslateOffset(angle, N)
        B = projectionLength(angle, P, Q)
        
        for translate in range(0, B):
            if angleSign >= 0 and perp: #Reverse for perp
                translateMojette = translateOffset - translate
            else:
                translateMojette = translate - translateOffset
            
            translateFinite = (inv*translateMojette)%N #has issues in C, may need checking
#            frtSpace[m][translateFinite] += bin
            u.append( (translateFinite+offset)%N )
            v.append( (m*translateFinite+offset)%N )
        
        coordinateList.append(u)
        coordinateList.append(v)
        coords.append(coordinateList)
    
    return coords
