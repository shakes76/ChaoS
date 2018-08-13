# -*- coding: utf-8 -*-
"""
Create package paths context

Created on Fri May 20 22:21:46 2016

@author: shakes
"""
import os
import sys

thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../..')

if libdir not in sys.path:
  sys.path.insert(0, libdir)
