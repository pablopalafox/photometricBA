/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <Eigen/Dense>

#include <cereal/cereal.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

#include "calibration.h"
#include "common_types.h"

namespace cereal {

template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
void serialize(
    Archive& archive,
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  static_assert(_Rows > 0, "matrix should be static size");
  static_assert(_Cols > 0, "matrix should be static size");
  for (size_t i = 0; i < _Rows; i++)
    for (size_t j = 0; j < _Cols; j++) archive(m(i, j));
}

template <class Archive>
void save(Archive& ar, const DoubleSphereCamera<double>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("xi", cam.getParam()[4]),
     cereal::make_nvp("alpha", cam.getParam()[5]));
}

template <class Archive>
void load(Archive& ar, DoubleSphereCamera<double>& cam) {
  Eigen::Matrix<double, 8, 1> intr;

  intr.setZero();

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]),
     cereal::make_nvp("xi", intr[4]), cereal::make_nvp("alpha", intr[5]));

  cam = DoubleSphereCamera<double>(intr);
}

template <class Archive>
void save(Archive& ar, const std::shared_ptr<AbstractCamera<double>>& cam) {
  const std::string cam_type = cam->name();
  const Eigen::Matrix<double, 8, 1> intr = cam->getParam();
  ar(cereal::make_nvp("cam_type", cam_type), cereal::make_nvp("fx", intr[0]),
     cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("p1", intr[4]),
     cereal::make_nvp("p2", intr[5]), cereal::make_nvp("p3", intr[6]),
     cereal::make_nvp("p4", intr[7]));
}

template <class Archive>
void load(Archive& ar, std::shared_ptr<AbstractCamera<double>>& cam) {
  Eigen::Matrix<double, 8, 1> intr;
  std::string cam_type;

  ar(cereal::make_nvp("cam_type", cam_type), cereal::make_nvp("fx", intr[0]),
     cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("p1", intr[4]),
     cereal::make_nvp("p2", intr[5]), cereal::make_nvp("p3", intr[6]),
     cereal::make_nvp("p4", intr[7]));

  cam = AbstractCamera<double>::from_data(cam_type, intr.data());
}

template <class Archive>
void serialize(Archive& ar, CalibCornerData& c) {
  ar(c.corners, c.corner_ids);
}

template <class Archive>
void serialize(Archive& ar, CalibInitPoseData& c) {
  ar(c.T_a_c, c.num_inliers, c.reprojected_corners);
}

template <class Archive>
void serialize(Archive& ar, Sophus::SE3d& p) {
  ar(cereal::make_nvp("px", p.translation()[0]),
     cereal::make_nvp("py", p.translation()[1]),
     cereal::make_nvp("pz", p.translation()[2]),
     cereal::make_nvp("qx", p.so3().data()[0]),
     cereal::make_nvp("qy", p.so3().data()[1]),
     cereal::make_nvp("qz", p.so3().data()[2]),
     cereal::make_nvp("qw", p.so3().data()[3]));
}

template <class Archive>
void serialize(Archive& ar, Calibration& cam) {
  ar(CEREAL_NVP(cam.T_i_c), CEREAL_NVP(cam.intrinsics));
}

template <class Archive, class Scalar, class CamT>
void serialize(Archive& ar, LoadCalibration<Scalar, CamT>& cam) {
  ar(CEREAL_NVP(cam.T_i_c), CEREAL_NVP(cam.intrinsics));
}

template <class Archive>
void serialize(Archive& ar, MatchData& m) {
  ar(CEREAL_NVP(m.T_i_j), CEREAL_NVP(m.inliers), CEREAL_NVP(m.matches));
}

template <class Archive>
void serialize(Archive& ar, KeypointsData& m) {
  ar(CEREAL_NVP(m.corners), CEREAL_NVP(m.corner_angles),
     CEREAL_NVP(m.corner_descriptors));
}

template <class Archive>
void serialize(Archive& ar, PhotoCandidatePointsData& m) {
  ar(CEREAL_NVP(m.selected_points), CEREAL_NVP(m.matched_points));
}

template <class Archive>
void serialize(Archive& ar, Camera& c) {
  ar(CEREAL_NVP(c.T_w_c), CEREAL_NVP(c.affine_ab),
     CEREAL_NVP(c.min_inv_distance), CEREAL_NVP(c.max_inv_distance));
}

template <class Archive>
void serialize(Archive& ar, Landmark& lm) {
  ar(CEREAL_NVP(lm.p), CEREAL_NVP(lm.obs), CEREAL_NVP(lm.outlier_obs));
}

template <class Archive>
void serialize(Archive& ar, Patch& p) {
  ar(CEREAL_NVP(p.positions), CEREAL_NVP(p.intensities));
}

template <class Archive>
void serialize(Archive& ar, PhotoLandmark& photolm) {
  ar(CEREAL_NVP(photolm.selected), CEREAL_NVP(photolm.d),
     CEREAL_NVP(photolm.patch), CEREAL_NVP(photolm.host),
     CEREAL_NVP(photolm.obs));
}

}  // namespace cereal

void write_vector_to_file(const std::vector<double>& myVector,
                          std::string filename) {
  std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
  std::ostream_iterator<char> osi{ofs};
  const char* beginByte = (char*)myVector.data();
  const char* endByte = (char*)&myVector.back() + sizeof(double);
  std::copy(beginByte, endByte, osi);
}
