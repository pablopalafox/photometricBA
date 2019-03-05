# Photometric Bundle Adjustment

<p align="center">
	<img src="/assets/images/photo.gif" alt="photo_gif" width="600">	
</p>

This is the code for the project "Photometric Bundle Adjustment" developed as part of the practical course "Vision-based Navigation" (IN2106) taught at the Technical University of Munich.

The authors of the original version are [Vladyslav Usenko](https://vision.in.tum.de/members/usenko) and [Nikolaus Demmel](https://vision.in.tum.de/members/demmeln).

The author of the extension presented in this repository is [Pablo R. Palafox](https://pablorpalafox.github.io/).

## 1. Prerequisites and Installation

We have tested this code in **Ubuntu 16.04** and **MacOS High Sierra**.

After cloning the repository, and only if you're on Ubuntu, run the following to install all required dependencies and to build all submodules. 

```bash
$ cd visnav_ws18
$ ./install_dependencies.sh
$ ./build_submodules.sh
```

**For MacOS**: Instead of running `./install_dependencies`, install the following packages using brew:

`brew install clang-format ceres-solver tbb glew eigen ccache`

Now, inside the root directory of the project, run the following to build the code:

<a name="build"></a>
```bash
$ mkdir build
$ cd build
$ ..cmake
$ make
```


## 2. EuRoC Dataset

For convenience of the user, this code ships with a subset of the __Vicon room 1 “easy”__ sequence from the [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and a pre-computed file ([gt_poses_timestamps.txt](data/mav0/state_groundtruth_estimate0/gt_poses_timestamps.txt)) that contains the closest groundtruth measurements to the selected set of images.

Note that we are not the owners of this data. The dataset was published in:

M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. Achtelik and R. Siegwart, The EuRoC micro aerial vehicle datasets, International Journal of Robotic Research, DOI: 10.1177/0278364915620033, early 2016. [bibtex](https://projects.asl.ethz.ch/datasets/doku.php?id=bibtex:euroc_datasets), [Publisher Link](https://journals.sagepub.com/doi/abs/10.1177/0278364915620033)


## 3. Run the project

An executable with the name **photo_sfm** must have been created inside the *build* directory after having ran [this](#build). If you're inside the **build** directory, go back using to the root folder and run the executable from there:

```bash
$ cd .. # after this, we should be at the root folder of the project
$ build/photo_sfm --dataset-path data/euroc_V1 --groundtruth-path data/mav0
```

The following Qt window should appear:

<p align="center">
	<img src="/assets/images/clear.png">	
</p>

The code ships with precomputed [maps](maps) that you can easily load by clicking the corresponding buttons in the Qt GUI:

- **load_map_geom**: map obtained after feature-based Structure-from-Motion.

<p align="center">
	<img src="/assets/images/geom.png">	
</p>

- **load_map_photo**: map obtained after running photometric BA on a map obtained previously using feature-based SfM.

<p align="center">
	<img src="/assets/images/photo.png">	
</p>

- **load_map_added_lm**: map obtained adding new points to the map using the concept of epipolar line search. This point addition is done on top of a "photometric" map, i.e., a map where 3D features have one _host frame_ (where they "live") and a bunch of _observations_ in other frames.

<p align="center">
	<img src="/assets/images/photolm.png">	
</p>

## 4. Results

The following table shows an ablation study comparing 3 different camera calibrations, namely **kb4** ([Kannala Brandt 4 Camera](http://www.ee.oulu.fi/mvg/files/pdf/pdf_697.pdf)), **ds** ([Double Sphere](https://vision.in.tum.de/research/vslam/double-sphere)) and **eucm** (Extended Unified Camera Model]). 

We compute the RMSE (Root Mean Square Error) of the ATE (Absolute Trajectory Error) for the "geometric" map (second column), for the "photometric" map (third column) and for the "new landmarks" map (fourth column), that is, a photometric map to which we have added new landmarks (using epipolar line search) and which we have then optimized again with photometric BA.

	| Camera calibration | ATE geometric | ATE photometric | ATE photo + new + photo |
	|:------------------:|:-------------:|:---------------:|:-----------------------:|
	|         kb4        |   0.0437577   |    0.0406419    |        **0.0404956**        |
	|         ds         |   0.0841705   |    0.0751423    |        0.0730535        |
	|        eucm        |   0.0442077   |    0.0444199    |         0.043117        |

By using the kb4 camera model and by running photometric BA after feature-based SfM + adding new points to the map + running photometric BA once again, we achieve the smallest ATE. 



## 5. License

The code for this practical course is provided under a BSD 3-clause license. See the LICENSE file for details.

Parts of the code (`include/tracks.h`, `include/union_find.h`) are adapted from OpenMVG and distributed under an MPL 2.0 licence.

Parts of the code (`include/local_parameterization_se3.hpp`, `src/test_ceres_se3.cpp`) are adapted from Sophus and distributed under an MIT license.

Note also the different licenses of thirdparty submodules.
