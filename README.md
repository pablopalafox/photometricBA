# Photometric Bundle Adjustment

<p align="center">
	<img src="/assets/images/photo.gif" alt="photo_gif" width="994" height="795">	
</p>

This is the code for the project "Photometric Bundle Adjustment" developed as part of the practical course "Vision-based Navigation" (IN2106) taught at the Technical University of Munich.

The authors of the original version are Vladyslav Usenko and Nikolaus Demmel.

The author of the extension presented in this repository is Pablo R. Palafox

## 1. Prerequisites and Installation

We have tested this code in **Ubuntu 16.04** and **MacOS High Sierra**.

After cloning the repository, and only if you're on Ubuntu, run the following to install all required dependencies and to build all submodules. 

```bash
cd visnav_ws18
./install_dependencies.sh
./build_submodules.sh
```

**For MacOS**: Instead of running `./install_dependencies`, install the following packages using brew:

`brew install clang-format ceres-solver tbb glew eigen ccache`

Now, inside the root directory of the project, run the following to build the code:

<a name="build"></a>
```bash
mkdir build
cd build
..cmake
make
```


## 2. EuRoC Dataset

For convenience of the user, this code ships with a subset of the __Vicon room 1 “easy”__ sequence from the [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and a pre-computed file ([gt_poses_timestamps.txt](data/mav0/state_groundtruth_estimate0/gt_poses_timestamps.txt)) that contains the closest groundtruth measurements to the selected set of images.

Note that we are not the owners of this data. The dataset was published in:

M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. Achtelik and R. Siegwart, The EuRoC micro aerial vehicle datasets, International Journal of Robotic Research, DOI: 10.1177/0278364915620033, early 2016. [bibtex](https://projects.asl.ethz.ch/datasets/doku.php?id=bibtex:euroc_datasets), [Publisher Link](https://journals.sagepub.com/doi/abs/10.1177/0278364915620033)


## 3. Run the project

An executable with the name **photo_sfm** must have been created inside the *build* directory after having ran [this](#build). If you're inside the **build** directory, go back using to the root folder and run the executable from there:

```bash
cd .. # after this, we should be at the root folder of the project
build/photo_sfm --dataset-path data/euroc_V1 --groundtruth-path data/mav0
```

The following Qt window should appear:

<p align="center">
	<img src="/assets/images/clear">	
</p>




## License

The code for this practical course is provided under a BSD 3-clause license. See the LICENSE file for details.

Parts of the code (`include/tracks.h`, `include/union_find.h`) are adapted from OpenMVG and distributed under an MPL 2.0 licence.

Parts of the code (`include/local_parameterization_se3.hpp`, `src/test_ceres_se3.cpp`) are adapted from Sophus and distributed under an MIT license.

Note also the different licenses of thirdparty submodules.
