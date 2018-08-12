# Chaotic Sensing (ChaoS)
This library houses the algorithms for sparse signal/image recovery via fractal sampling found in the open-access publication
```
Chandra, S. S.; Ruben, G.; Jin, J.; Li, M.; Kingston, A.; Svalbe, I. & Crozier, S.
Chaotic Sensing,
IEEE Transactions on Image Processing, 2018, 1-1
```
See [pblication page](https://doi.org/10.1109/TIP.2018.2864918) for more details.

The newly discovered finite fractal is also presented in this work
![Finite Fractal](projects/finite_fractal/farey_image_1031_1.png)

## ChaoS Library
This is a mostly Python library for implementing ChaoS algorithms

There are a number of sub-modules:

* radon - this houses the finite Radon transform algorithms
* mojette - this houses the aperiodic Radon transform algorithms
* tomo - this houses the traditional reconstruction sampling

The library has been developed under Python 3.5

## License
The license is outlined in the license.txt file.
```
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
```