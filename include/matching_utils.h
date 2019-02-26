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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include "camera_models.h"
#include "common_types.h"

using namespace opengv;

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t = T_0_1.translation();
  const Eigen::Matrix3d R = T_0_1.rotationMatrix();

  Eigen::Vector3d t_norm;
  t_norm = t.normalized();

  Eigen::Matrix3d T_hat;
  T_hat << 0, -t_norm[2], t_norm[1], t_norm[2], 0, -t_norm[0], -t_norm[1],
      t_norm[0], 0;

  E = T_hat * R;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);

    // So I manage to pass the test with the code below, but I was wondering why
    // double epipolar_constraint = p1_2d.transpose().homogeneous() * E *
    // p0_2d.homogeneous() does not work,
    // i.e., using the 2d coordinates of the points as Prof. Cremmers describes
    // in the youtube lectures. In those slides (Lecture 8 minute 36:30), he
    // shows equation x2.transpose * T^ * R * x1 = 0

    double epipolar_constraint = p0_3d.transpose() * E * p1_3d;
    //    std::cout << "epip const " << epipolar_constraint << std::endl;

    if (fabs(epipolar_constraint) < epipolar_error_threshold) {
      md.inliers.push_back(md.matches[j]);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       double ransac_thresh, int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();

  // TODO: run RANSAC with using opengv's CentralRelativePose and
  // store in md.inliers. If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty.

  // create the central relative adapter
  bearingVectors_t bearingVectors1;
  bearingVectors_t bearingVectors2;

  for (size_t j = 0; j < md.matches.size(); j++) {
    bearingVectors1.push_back(
        cam1->unproject(kd1.corners[md.matches[j].first]).normalized());
    bearingVectors2.push_back(
        cam2->unproject(kd2.corners[md.matches[j].second]).normalized());
  }

  relative_pose::CentralRelativeAdapter adapter(bearingVectors1,
                                                bearingVectors2);

  // create a RANSAC object
  sac::Ransac<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;

  // create a CentralRelativePoseSacProblem
  std::shared_ptr<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new sac_problems::relative_pose::CentralRelativePoseSacProblem(
              adapter, sac_problems::relative_pose::
                           CentralRelativePoseSacProblem::NISTER));

  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.max_iterations_ = 300;
  ransac.probability_ = 0.99;
  ransac.computeModel();

  transformation_t bestModel =
      relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  Eigen::Vector3d t_refined;
  t_refined = bestModel.block<3, 1>(0, 3);

  Eigen::Matrix3d R_refined;
  R_refined = bestModel.block<3, 3>(0, 0);

  /////////////////////////////////////////////////////////////////////////////
  // UPDATE SET OF INLIERS using the refined pose
  transformation_t T_refined;
  T_refined.block<3, 3>(0, 0) = R_refined;
  T_refined.block<3, 1>(0, 3) = t_refined;
  ransac.model_coefficients_ = T_refined;
  ransac.sac_model_->selectWithinDistance(ransac.model_coefficients_,
                                          ransac.threshold_, ransac.inliers_);
  /////////////////////////////////////////////////////////////////////////////

  if (ransac.inliers_.size() >= size_t(ransac_min_inliers)) {
    for (size_t i = 0; i < ransac.inliers_.size(); i++) {
      size_t index = ransac.inliers_[i];
      md.inliers.push_back(md.matches[index]);
    }
  }

  md.T_i_j.setRotationMatrix(R_refined);
  md.T_i_j.translation() = t_refined;
}
