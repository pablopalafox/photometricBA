# Photometric Bundle Adjustment



This is the code for the project "Photometric Bundle Adjustment" developed as part of the practical course "Vision-based Navigation" (IN2106) taught at the Technical University of Munich.

The authors of the original version are Vladyslav Usenko and Nikolaus Demmel.
The author of the extension presented in this repository is Pablo R. Palafox

## Prerequisites and Installation

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

```bash
mkdir build
cd build
..cmake
make
```

An executable with the name **photo_sfm** must have been created inside the build directory. Run `./photo_sfm --dataset_path=`


## License

The code for this practical course is provided under a BSD 3-clause license. See the LICENSE file for details.

Parts of the code (`include/tracks.h`, `include/union_find.h`) are adapted from OpenMVG and distributed under an MPL 2.0 licence.

Parts of the code (`include/local_parameterization_se3.hpp`, `src/test_ceres_se3.cpp`) are adapted from Sophus and distributed under an MIT license.

Note also the different licenses of thirdparty submodules.
