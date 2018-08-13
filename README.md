# Chaotic Sensing (ChaoS)
**Under Construction**

This library houses the algorithms for sparse signal/image recovery via fractal sampling found in the open-access publication
```
Chandra, S. S.; Ruben, G.; Jin, J.; Li, M.; Kingston, A.; Svalbe, I. & Crozier, S.
Chaotic Sensing,
IEEE Transactions on Image Processing, 2018, 1-1
```
See [IEEE publication page](https://doi.org/10.1109/TIP.2018.2864918) for more details.
DOI: [https://doi.org/10.1109/TIP.2018.2864918](https://doi.org/10.1109/TIP.2018.2864918)

The newly discovered finite fractal is also presented in this work
![Finite Fractal](projects/finite_fractal/farey_image_1031_1.png)

## Setup/Dependencies
The ChaoS library is dependent on the usual Numpy, Scipy, Matplotlib and scikit-image libraries. You may need pyFFTW installed as well.
The best way to set this up in Windows is using the [WinPython scientific distribution](https://sourceforge.net/projects/winpython/files/WinPython_2.7/2.7.13.1/) for Python 2.7.13.
WinPython sets up a Python distribution with minimal libraries, but in a compact stand-alone directory without the need to install.
In the interest of reproducibility, I have uploaded my WinPython distribution. Simply download, extract, launch Spyder from the distribution, open the script you wish to run and run it.

## ChaoS Library
This is a mostly Python library for implementing ChaoS algorithms.

**Warning: Although compatible for Python 3, the library has been developed under Python 2.7.13 and all results in the paper correspond only to this version.**

There are a number of sub-modules:

* radon - this houses the finite Radon transform algorithms
* mojette - this houses the aperiodic Radon transform algorithms
* tomo - this houses the traditional reconstruction sampling

The main scripts for generating the results of the [publication](https://doi.org/10.1109/TIP.2018.2864918) can be found in
* projects/finite_fractal - To generate the fractal see test_finite_farey_fractal.py
* projects/simulation - To run simulations of the reconstruction algorithms see test_finite_slices_osem_plot.py and test_finite_slices_ossirt_plot.py. Then run the test_compute_metrikz.py to generate the relevant metrics and figures.

## Known Issues
* Python 3.5 has issues running the SSIM metric.

## License
Nearly all parts of the library is licensed under the [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0) and is outlined in the license.txt file.
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
The exception is the generation of fractals and figures via this library. All figures and code pertaining to the display, saving and generation of fractals, are covered under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
For publication and commercial use of this content, please obtain a suitable license from the author.
