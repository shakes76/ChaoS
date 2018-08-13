# -*- coding: utf-8 -*-
"""
Chaotic Sensing (ChaoS) is a collection of algorithms for image processing and sparse reconstruction.

There are a number of sub-modules:

radon - this houses the finite Radon transform algorithms
mojette - this houses the aperiodic Radon transform algorithms
tomo - this houses the traditional reconstruction algorithms

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
import os.path as osp

pkg_dir = osp.abspath(osp.dirname(__file__))
data_dir = osp.join(pkg_dir, '../data')

__version__ = '0.1'

del osp
