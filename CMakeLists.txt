cmake_minimum_required(VERSION 3.4)
project(visnav)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
set(CERES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-ceres-solver")
set(PANGOLIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-Pangolin/src")
set(OPENGV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-opengv")

# Set default build type if not specified otherwise.
# See https://cmake.org/pipermail/cmake/2012-May/050243.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# See comment in ./build_submodules.sh on the relation of arch=native
# and Eigen's alignment. TLDR: Passing arch=native for all build types
# is currently the only viable option to avoid suble bugs with Eigen.
SET(CXX_MARCH native)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# for some reason clang doesn't include them in Wall whereas gcc does, so include them explicitly.
SET(EXTRA_WARNING_FLAGS "-Wsign-compare")

SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -ftree-vectorize")
SET(CMAKE_CXX_FLAGS " -ftemplate-backtrace-limit=0 -Wall ${EXTRA_WARNING_FLAGS} -march=${CXX_MARCH} ${CMAKE_CXX_FLAGS}")

set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

include_directories(thirdparty/Sophus)
include_directories(thirdparty/cereal/include)
include_directories(thirdparty/CLI11/include)
include_directories(include)

# CERES
find_package(Ceres 1.14 REQUIRED HINTS "${CERES_ROOT}")
include_directories(${CERES_INCLUDE_DIRS})

# PANGOLIN
set(Pangolin_DIR ${PANGOLIN_ROOT})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIR})
message(STATUS "Found Pangolin headers in: ${Pangolin_INCLUDE_DIR}")

# OpenGV
find_package(opengv REQUIRED HINTS ${OPENGV_ROOT})
message(STATUS "Found OpenGV headers in: ${opengv_DIR}")

# TBB
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})
message(STATUS "Found TBB headers in: ${TBB_INCLUDE_DIR}")

find_package( OpenCV 4 QUIET COMPONENTS core features2d)
if (OpenCV_FOUND)
  message(STATUS "Found OpenCV 4 headers in: ${OpenCV_INCLUDE_DIRS}")
else()
  message(STATUS "Didn't find OpenCV 3, looking for 2")
  # HINTS needed for macos homebrew (install with `brew install opencv@2`)
  find_package( OpenCV 2 REQUIRED COMPONENTS core features2d
      HINTS /usr/local/Cellar/opencv@2 )
  message(STATUS "Found OpenCV 2 headers in: ${OpenCV_INCLUDE_DIRS}")
endif()



#add_executable(test_ceres_se3 src/test_ceres_se3.cpp)
#target_link_libraries(test_ceres_se3 ceres)

#add_executable(calibration src/calibration.cpp)
#target_link_libraries(calibration ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES})

#add_executable(sfm src/sfm.cpp)
#target_link_libraries(sfm ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS} opengv)

#add_executable(odometry src/odometry.cpp)
#target_link_libraries(odometry ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS} opengv)

add_executable(photo_sfm src/photo_sfm.cpp)
target_link_libraries(photo_sfm ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS} opengv)


#add_subdirectory(test)
