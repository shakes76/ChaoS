#!/usr/bin/env python
'''
This is the Farey class. Generate Farey series as vectors (q/p).
The vector is stored as complex numbers with the imaginary part as p.
The cordinate system is assumed to be matrix system with p (and hence x) along the rows.

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
import math
import finitetransform.numbertheory as nt #local modules

def farey(p, q):
    '''
    Convenience member for creating a Farey vector from a Farey fraction p/q
    '''
    return complex( int(q), int(p) )
    
def getX(angle):
    '''
    Convenience function for extracting the consistent coordinate from a Farey vector
    This is based on matrix coordinate system, i.e. x is taken along rows and is normally q of p/q vector
    '''
    if not isinstance(angle, complex):
        print("Warning: Angle provided not of correct type. Use the farey() member.")
    
    return angle.real
    
def getY(angle):
    '''
    Convenience function for extracting the consistent coordinate from a Farey vector
    This is based on matrix coordinate system, i.e. y is taken along columns and is normally p of p/q vector
    '''
    if not isinstance(angle, complex):
        print("Warning: Angle provided not of correct type. Use the farey() member.")
    
    return angle.imag
    
def get_pq(angle):
    '''
    Return p, q tuple of the angle provided using module convention
    '''
    p = int(angle.imag)
    q = int(angle.real)
    
    return p, q
    
def projectionLength(angle, P, Q):
    '''
    Return the number of bins for projection at angle of a PxQ image.
    '''
    p, q = get_pq(angle)
    return (Q-1)*abs(p)+(P-1)*abs(q)+1 #no. of bins
    
def total(n):
    '''
    Return the approx total Farey vectors/angles possible for given n
    '''
    return int(0.304*n*n + 0.5)
    
def size(mu):
    '''
    Given number of projections mu, return the approx size n
    '''
    return int(math.sqrt(mu/0.304))

def angle(angle, radians=True):
    '''
    Given p and q, return the corresponding angle (in Radians by default)
    '''
    p, q = get_pq(angle)
    theta = 0
    if p != 0:
        theta = math.atan(q/float(p))
        
    if radians:
        return theta
    else:
        return 180/math.pi*theta
    
def toFinite(fareyVector, N):
    '''
    Return the finite vector corresponding to the Farey vector provided for a given modulus/length N
    and the multiplicative inverse of the relevant Farey angle
    '''
    p, q = get_pq(fareyVector)
    coprime = nt.is_coprime(abs(q), N)
    qNeg = q #important, if q < 0, preserve for minverse.
    if q < 0:
        q += N #necessary for quadrants other than the first
    if p < 0:
        p += N #necessary for quadrants other than the first
#    print("p:", p, "q:", q)

    mValue = 0
    inv = 1
    if coprime:
        inv = nt.minverse(qNeg, N)
#        identity = (inv*q)%N
        mValue = (p*inv)%N
#        print "vec:", fareyVector, "m:", mValue, "inv:", inv, "I:", identity
    else: #perp projection
        inv = nt.minverse(p, N)
        mValue = (q*inv)%N + N 
#        print "perp vec:", fareyVector, "m:", mValue, "inv:", inv

    return mValue, inv
    
def finiteTranslateOffset(fareyVector, N, P, Q):
    '''
    Translate offset required when mapping Farey vectors to finite angles
    Returns translate offset and perp Boolean flag pair
    '''
    p, q = get_pq(fareyVector)
    angleSign = p*q
    coprime = nt.is_coprime(abs(q), N)
    
    perp = True
    if coprime:
        perp = False
    
    translateOffset = 0
    if angleSign >= 0 and not perp: #If not negative slope
        translateOffset = abs(p)*(Q-1)
    elif angleSign >= 0: #If not negative slope and perp
        translateOffset = abs(p)*(Q-1)
        
    return translateOffset, perp

class Farey:
    '''
    Class for the Farey vectors. It uses Gaussian integers to represent them.
    
    Conventions used in theis class:
    Farey fraction p/q is represented as a vector (q, p) in (x, y) coordinates
    '''
    #static constants
    startVector = farey(0, 1)
    endVector = farey(1, 1)

    def __init__(self):
        #class variables
        self.vector = farey(0, 0)
        self.generated = False
        self.generatedFinite = False
        self.compact = False
        self.vectors = []
        self.finiteAngles = []
        
    def nextFarey(self, n, vec1, vec2):
        '''
        Generate and return the next Farey vector
        '''
        p1 = vec1.imag
        q1 = vec1.real
        p2 = vec2.imag
        q2 = vec2.real

        p3 = math.floor( (q1+n) / float(q2) )*p2 - p1
        q3 = math.floor( (q1+n) / float(q2) )*q2 - q1
        
        return farey(p3, q3)
       
    def nextCompactFarey(self, n, vec1, vec2):
        '''
        Generate and return the next compact (in terms of L1 norm) Farey vector 
        '''
        p1 = vec1.imag
        q1 = vec1.real
        p2 = vec2.imag
        q2 = vec2.real

        p3 = math.floor( (q1+p1+n) / float(q2+p2) )*p2 - p1
        q3 = math.floor( (q1+p1+n) / float(q2+p2) )*q2 - q1
        
        return farey(p3, q3)
        
    def generate(self, n, octants=1):
        '''
        Generate all the Farey vectors up to given n.
        Octants is the number of octants to produce, 1 is the first octant, 2 is the first two octants, 4 is first two quadrants and > 4 is all quadrants
        '''
        del self.vectors[:] #clear list
        
        nthVector = farey(1, n) # 1/n
        angle1 = self.startVector # 0/1
        angle2 = nthVector
        self.vectors.append(self.startVector)
        
        nextAngle = farey(0, 0)
        while nextAngle != self.endVector: # 1/1
            if self.compact:
                nextAngle = self.nextCompactFarey(n, angle1, angle2)
            else:
                nextAngle = self.nextFarey(n, angle1, angle2)

            self.vectors.append(nextAngle)
#            print nextAngle
            angle1 = angle2
            angle2 = nextAngle
            
        self.vectors.append(nthVector)
        
        secondOctantVectors = []
        if octants > 1:
            #first quadrant, second octant
            for nextAngle in self.vectors:
                if not nextAngle.imag == nextAngle.real:
                    nextOctantAngle = farey(nextAngle.real, nextAngle.imag) #mirror
                    secondOctantVectors.append(nextOctantAngle)
                
            self.vectors += secondOctantVectors #merge lists
            
        firstQuadrantVectors = self.vectors
        if octants > 2: 
            #second quadrant
            secondQuadrantVectors = []
            for nextAngle in firstQuadrantVectors:
                if nextAngle.imag == 0 or nextAngle.real == 0:
                    continue
                nextOctantAngle = farey(nextAngle.imag, -nextAngle.real) #reflect x
                secondQuadrantVectors.append(nextOctantAngle)
                
            self.vectors += secondQuadrantVectors #merge lists
            
        if octants > 4: #do all quadrants
            #third quadrant
            thirdQuadrantVectors = []
            for nextAngle in firstQuadrantVectors:
                nextOctantAngle = farey(-nextAngle.imag, -nextAngle.real) #reflect x, y
                thirdQuadrantVectors.append(nextOctantAngle)
                
            self.vectors += thirdQuadrantVectors #merge lists
            
            #forth quadrant
            forthQuadrantVectors = []
            for nextAngle in firstQuadrantVectors:
                nextOctantAngle = farey(-nextAngle.imag, nextAngle.real) #reflex y
                forthQuadrantVectors.append(nextOctantAngle)
                
            self.vectors += forthQuadrantVectors #merge lists
        
        self.generated = True
        
    def generate2(self, n, octants=1):
        '''
        Generate all the Farey vectors up to given n (exclusive). Tries to handle octants more compactly
        Octants is the number of octants to produce, 1 is the first octant, 2 is the first two octants. > 2 is all quadrants
        '''
        del self.vectors[:] #clear list
        
        nthVector = farey(1, n) # 1/n
        angle1 = self.startVector # 0/1
        angle2 = nthVector
        self.vectors.append(self.startVector)
        
#        if octants > 1:
#            nextOctantAngle = farey(self.startVector.real, self.startVector.imag) #mirror
#            self.vectors.append(nextOctantAngle)
#        if octants > 2:
#            nextOctantAngle = farey(self.startVector.imag, -self.startVector.real) #reflect x
#            self.vectors.append(nextOctantAngle)
        if octants > 3:
            nextOctantAngle = farey(-self.startVector.real, self.startVector.imag) #mirror, reflect x
            self.vectors.append(nextOctantAngle)
        if octants > 4:
            nextOctantAngle = farey(-self.startVector.imag, -self.startVector.real) #reflect x, y
            self.vectors.append(nextOctantAngle)
#        if octants > 5:
#            nextOctantAngle = farey(-self.startVector.real, -self.startVector.imag) #mirror, reflect x, y
#            self.vectors.append(nextOctantAngle)
#        if octants > 6:
#            nextOctantAngle = farey(-self.startVector.imag, self.startVector.real) #reflex y
#            self.vectors.append(nextOctantAngle)
        if octants > 7:
            nextOctantAngle = farey(self.startVector.real, -self.startVector.imag) #mirror, reflex y
            self.vectors.append(nextOctantAngle)
        
        nextAngle = farey(0, 0)
        while nextAngle != self.endVector: # 1/1
            if self.compact:
                nextAngle = self.nextCompactFarey(n, angle1, angle2)
            else:
                nextAngle = self.nextFarey(n, angle1, angle2)

            self.vectors.append(nextAngle)
            
            if octants > 1:
                nextOctantAngle = farey(nextAngle.real, nextAngle.imag) #mirror
                self.vectors.append(nextOctantAngle)
            if octants > 2:
                nextOctantAngle = farey(nextAngle.imag, -nextAngle.real) #reflect x
                self.vectors.append(nextOctantAngle)
            if octants > 3:
                nextOctantAngle = farey(-nextAngle.real, nextAngle.imag) #mirror, reflect x
                self.vectors.append(nextOctantAngle)
            if octants > 4:
                nextOctantAngle = farey(-nextAngle.imag, -nextAngle.real) #reflect x, y
                self.vectors.append(nextOctantAngle)
            if octants > 5:
                nextOctantAngle = farey(-nextAngle.real, -nextAngle.imag) #mirror, reflect x, y
                self.vectors.append(nextOctantAngle)
            if octants > 6:
                nextOctantAngle = farey(-nextAngle.imag, nextAngle.real) #reflex y
                self.vectors.append(nextOctantAngle)
            if octants > 7:
                nextOctantAngle = farey(nextAngle.real, -nextAngle.imag) #mirror, reflex y
                self.vectors.append(nextOctantAngle)
            
#            print nextAngle
            angle1 = angle2
            angle2 = nextAngle
        
    def generateRange(self, n, angleMin, angleMax, octants=1, radians=True):
        '''
        Generate all the Farey vectors up to given n within given angle range inclusive.
        Octants is the number of octants to produce, 1 is the first octant, 2 is the first two octants etc.
        '''
        if not self.generated:
            self.generate(n, octants)
            
        vectors = self.vectors[:] #copy list
        del self.vectors[:] #clear list
        for vector in vectors:
            vectorAngle = angle(vector,radians)
            if vectorAngle <= angleMax and vectorAngle >= angleMin:
                self.vectors.append(vector)
        
    def generateFinite(self, N):
        '''
        Generate a list of finite vectors for the corresponding Farey vectors
        '''
        if not self.generated:
            self.generate(N, 4)
            
        del self.finiteAngles[:] #clear list
        for vector in self.vectors:
            if vector.real == 0:
                m = 0
            else:
                m, inv = toFinite(vector, N)
            self.finiteAngles.append(m)
            
        self.generatedFinite = True
    
    def generateFiniteWithCoverage(self, N, L1Norm=True):
        '''
        Generate Farey set and corresponding m values then select vectors that cover
        all of DFT space. Internal lists are updated to match coverage.
        If L1Norm is true, sort the angles based on norm first to minimise length.
        '''
        if not self.generatedFinite:
            self.generateFinite(N)
            
        if L1Norm:
            self.finiteAngles, self.vectors = self.sort('length')
            
        count = 0
        vectors = self.vectors[:] #copy list
        finiteAngles = self.finiteAngles[:] #copy list
        filled = [0]*(N+1) #list of zeros        
        del self.vectors[:] #clear list
        del self.finiteAngles[:] #clear list
#        print vectors
        for vector, m in zip(vectors, finiteAngles):
            if filled[m] == 0:
                count += 1
                filled[m] = 1
                self.vectors.append(vector)
                self.finiteAngles.append(m)
            else:
                continue
                
            if count == N+1:
                break
    
    def compactOn(self):
        '''
        Generate the shortest vectors (in terms of L1)
        '''
        self.compact = True
        
    def compactOff(self):
        '''
        Do not generate the shortest vectors (in terms of L1)
        '''
        self.compact = False
    
    def sort(self, type='length'):
        '''
        Returns sorted vectors and finite angles (if finite vectors have been computed).
        
        Type:
        'length':
        Return sorted angles based on L1 norm
        'finite':
        Return sorted angles based on finite angles and the sorted finite angles (as a pair)
        '''
        if type == 'finite':
            return [y for (y,x) in sorted(zip(self.finiteAngles,self.vectors), key=lambda pair: pair[0])], [x for (y,x) in sorted(zip(self.finiteAngles,self.vectors), key=lambda pair: pair[0])]
        else:
            norms = []
            for vector in self.vectors:
                p, q = get_pq(vector)
                norm = abs(p) + abs(q)
                norms.append(norm)
            
            if not self.finiteAngles:
                return [x for (n,x) in sorted(zip(norms,self.vectors), key=lambda pair: pair[0])]
            else:
                return [y for (n,y,x) in sorted(zip(norms,self.finiteAngles,self.vectors), key=lambda pair: pair[0])], [x for (n,y,x) in sorted(zip(norms,self.finiteAngles,self.vectors), key=lambda pair: pair[0])]
