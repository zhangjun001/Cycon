# Unsupervised Deformable Image Registration Using Cycle-Consistent CNN

Pytorch implementation for a MICCAI 2019 paper, entitled "Unsupervised Deformable Image Registration Using Cycle-Consistent CNN".

I only implemented the algorithm according to the paper from Kim et al. 2019 (https://link.springer.com/chapter/10.1007/978-3-030-32226-7_19), and it is not guaranteed to be the optimal implementation.

The code was written by Jun Zhang, Tencent AI Healthcare. The implementation of NCC loss was borrowed from Voxelmorph (https://github.com/voxelmorph/voxelmorph).

# Applications

3D deformable image registration

# Prerequisites

Install pytorch (0.4.0 or 0.4.1) and dependencies from http://pytorch.org/ 

Install SimpleITK with `pip install SimpleITK` 

Install numpy with `pip install numpy`

# Apply
```
cd Code
```
Apply our Pre-trained Model (Note that the image pairs must be linearly aligned before using our code).
It is better to perform the histogram matching according to any one of our provided images. 
```
python Test.py --fixed ../Dataset/image_A.nii.gz --moving ../Dataset/image_B.nii.gz
```

# Train 
If you want to train a model using your own dataset, please perform the following script.
```
python Train.py --datapath yourdatapath
```
You need to perform the histogram matching for you dataset since we emply the `MSE loss` for measuring similarity.
You may need to adjust the parameters of `--lambda`, `--alpha`, and  `--beta` for your dataset to get better registration performance. 