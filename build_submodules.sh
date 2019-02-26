#!/usr/bin/env bash

MYPWD=$(pwd)

set -x
set -e

GCC=gcc
GXX=g++

BUILD_TYPE=RelWithDebInfo

if [ -n "$1" ]; then
BUILD_TYPE=$1
fi

NUM_PARALLEL_BUILDS=4

# Important note on Eigen alignment and the arch flag. TLDR: Passing
# arch=native for all build types is currently the only viable option
# to avoid suble bugs with Eigen.
#
# Eigen uses 16 byte alignemnt by default, but switches to 32 byte
# alignment if AVX instructions are enabled. This is the case on
# modern Intel hardware if we pass arch=native. It is vital to ensure
# that all translation units, including all thirdparty libraries, use
# the same value for EIGEN_MAX_ALIGN_BYTES (see
# https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html),
# since the aliged-malloc functions provided by Eigen might not be
# inlined and thus be taken from any of the translation units. Our
# current approach is to ensure arch=native everywhere. A possible
# alternative would be to explicitly pass -DEIGEN_MAX_ALIGN_BYTES=32
# everywhere, then 32 bytes would be used regardless of whether avx is
# enabled or not.
#
# Note that Ceres might override with arch=native in Release mode no
# matter what we pass it and OpenGV overwrites to always build in
# release mode and use arch=native, so any other value for CXX_MARCH
# might not work as expected. Also, even though we explicitly pass
# -O3, it might be overwritten e.g. with -O2 if we don't also set the
# build type to Release.

CXX_MARCH=native

EIGEN_DIR="$MYPWD/thirdparty/eigen"

COMMON_CMAKE_ARGS=(-DCMAKE_C_COMPILER=${GCC} -DCMAKE_CXX_COMPILER=${GXX} -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CXX_FLAGS="-march=$CXX_MARCH -O3")

BUILD_CERES=thirdparty/build-ceres-solver
BUILD_PANGOLIN=thirdparty/build-Pangolin
BUILD_OPENGV=thirdparty/build-opengv

git submodule sync --recursive
git submodule update --init --recursive

rm -rf "$BUILD_CERES"
rm -rf "$BUILD_PANGOLIN"
rm -rf "$BUILD_OPENGV"

mkdir -p "$BUILD_CERES"
pushd "$BUILD_CERES"
cmake ../ceres-solver "${COMMON_CMAKE_ARGS[@]}" \
    "-DEIGEN_INCLUDE_DIR_HINTS=$EIGEN_DIR" \
    -DCXX11=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DEXPORT_BUILD_DIR=ON \
    -DTBB=ON \
    -DOPENMP=OFF
make -j$NUM_PARALLEL_BUILDS
popd

mkdir -p "$BUILD_PANGOLIN"
pushd "$BUILD_PANGOLIN"
cmake ../Pangolin "${COMMON_CMAKE_ARGS[@]}" \
    -DEXPORT_Pangolin=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF \
    "-DEIGEN_INCLUDE_DIR=$EIGEN_DIR"
make -j$NUM_PARALLEL_BUILDS
popd

mkdir -p "$BUILD_OPENGV"
pushd "$BUILD_OPENGV"
cmake ../opengv "${COMMON_CMAKE_ARGS[@]}" \
    "-DEIGEN_INCLUDE_DIR=$EIGEN_DIR" \
    -DBUILD_TESTS=OFF
make -j$NUM_PARALLEL_BUILDS
popd
