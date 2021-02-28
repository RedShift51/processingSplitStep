Few python scripts for numerical results processing

py_scripts/IExpAllN.py
collecting local speckle maximums, parallel screens processing
Writing values and its coordinates to csv file

py_scripts/collect_iexp.py
Processing obtained csvs, then for regions of interest 
we collect statistics from all random seeds and calculate histograms
of local maximums

py_scripts/gpu_akde.py
To increase accuracy of histogram we use modified adaptive kernel density
estimation (akde) of probability density function (PDF) by gaussian kernels.
To speed up this operation TensorFlow library is used

py_scripts/profile_avg_big_mean.py
Script for calculating the dependence of the average intensity on the radius

py_scripts/profile_avg_8192.py
Script for PDF calculation in axial rings, angle averaging

jupyter files
fit_law.ipynb
Few scales comparison of obtained data and its approximations and fits

read_dict.ipynb
Transform data obtained from HPC server calculations to json-like storage

iexp.ipynb
Further HPC data processing
