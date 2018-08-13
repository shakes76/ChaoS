# -*- coding: utf-8 -*-
"""
Tomography related functions for image reconstruction from analog projections.
'simple example implementation for standard parallel beam projection/back-projection routines'

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
import numpy
import math
    
def sliceSamples(theta, b, N, fftShape):
    '''
    Generate the b points along slice of length N at angle theta of DFT space with shape.
    '''
    r, s = fftShape
    phi = math.pi/2.0-theta
    projLengthAdjacent = float(N)*math.cos(phi)
    projLengthOpposite = float(N)*math.sin(phi)
    #ensure origin is included
    u2 = numpy.linspace(0, -projLengthAdjacent/2.0, b/2.0, endpoint=False) + r/2.0
    u1 = numpy.linspace(projLengthAdjacent/2.0, 0, b/2.0, endpoint=False) + r/2.0
    u = numpy.concatenate((u1,u2),axis=0)
    v2 = numpy.linspace(0, projLengthOpposite/2.0, b/2.0, endpoint=False) + s/2.0
    v1 = numpy.linspace(-projLengthOpposite/2.0, 0, b/2.0, endpoint=False) + s/2.0
    v = numpy.concatenate((v1,v2),axis=0)
    #print "u:",u
    #print "v:",v
    return u, v
    
def randomSamples(row, b, fftShape):
    '''
    Generate random b points along row of DFT space with shape.
    '''
    r, s = fftShape
    u = numpy.full(int(b), row, dtype=numpy.int32)
    #~ v = numpy.random.randint(0, high=s, size=b)
    v = numpy.random.choice(s, int(b), replace=False)
    #print "u:",u
    #print "v:",v
    return u, v
    
def randomLines(b, fftShape):
    '''
    Generate random b lines of DFT space with shape.
    '''
    r, s = fftShape
    #~ randLinesRows = numpy.random.randint(0, high=r, size=b)
    randLinesRows = numpy.random.choice(s, int(b), replace=False)
    randLines = []
    for randRow in randLinesRows:
        u = numpy.full(s, randRow, dtype=numpy.int32)
        v = numpy.arange(0, s)
        line = u, v
        randLines.append(line)
        #print "u:",u
        #print "v:",v
    return randLines

#reconstruction algorithms
#FBP OMITTED
