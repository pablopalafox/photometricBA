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

#include <fstream>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include "common_types.h"
#include "serialization.h"

#include "local_parameterization_se3.hpp"
#include "reprojection.h"

#include "tracks.h"

using namespace opengv;

// save map with all features and matches
void save_map_file(const std::string &map_path, const Corners &feature_corners,
                   const Matches &feature_matches,
                   const FeatureTracks &feature_tracks,
                   const FeatureTracks &outlier_tracks,
                   const Cameras &geometric_cameras, const Cameras &cameras,
                   const Landmarks &landmarks,
                   const PhotoLandmarks &photolandmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(geometric_cameras);
      archive(cameras);
      archive(landmarks);
      archive(photolandmarks);

      size_t num_obs_lm = 0;
      for (const auto &kv : landmarks) {
        num_obs_lm += kv.second.obs.size();
      }
      size_t num_obs_photolm = 0;
      for (const auto &kv : photolandmarks) {
        num_obs_photolm += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, "
                << num_obs_lm << " observations, " << photolandmarks.size()
                << " photolandmarks, " << num_obs_photolm << " observations)"
                << std::endl;

    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string &map_path, Corners &feature_corners,
                   Matches &feature_matches, FeatureTracks &feature_tracks,
                   FeatureTracks &outlier_tracks, Cameras &geometric_cameras,
                   Cameras &cameras, Landmarks &landmarks,
                   PhotoLandmarks &photolandmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(geometric_cameras);
      archive(cameras);
      archive(landmarks);
      archive(photolandmarks);

      size_t num_obs_lm = 0;
      for (const auto &kv : landmarks) {
        num_obs_lm += kv.second.obs.size();
      }
      size_t num_obs_photolm = 0;
      for (const auto &kv : photolandmarks) {
        num_obs_photolm += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, "
                << num_obs_lm << " observations, " << photolandmarks.size()
                << " photolandmarks, " << num_obs_photolm << " observations)"
                << std::endl;

    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const TimeCamId &tcid0,
                                   const TimeCamId &tcid1,
                                   const Calibration &calib_cam,
                                   const Corners &feature_corners,
                                   const FeatureTracks &feature_tracks,
                                   const Cameras &cameras,
                                   Landmarks &landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<TimeCamId> tcids = {tcid0, tcid1};
  if (!GetTracksInImages(tcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // GOAL: Triangulate all new features and add to the map

  bearingVectors_t bearingVectors0;
  bearingVectors_t bearingVectors1;

  // Go over all shared feature tracks
  for (const TrackId &shared_track_id : shared_track_ids) {
    // If shared track is not already in landmarks...
    if (landmarks.find(shared_track_id) == landmarks.end()) {
      // Add id of track to new_track_ids
      new_track_ids.push_back(shared_track_id);

      // GOAL: Add shared track to landmarks

      // Get the 2d features corresponding to the current shared_track_id
      const FeatureId featureId0 = feature_tracks.at(shared_track_id).at(tcid0);
      const FeatureId featureId1 = feature_tracks.at(shared_track_id).at(tcid1);

      const Eigen::Vector2d p0_2d =
          feature_corners.at(tcid0).corners[featureId0];
      const Eigen::Vector2d p1_2d =
          feature_corners.at(tcid1).corners[featureId1];

      // Add 2d coordinates to bearing vectors
      bearingVectors0.push_back(
          calib_cam.intrinsics[tcid0.second]->unproject(p0_2d).normalized());
      bearingVectors1.push_back(
          calib_cam.intrinsics[tcid1.second]->unproject(p1_2d).normalized());
    }
  }

  Sophus::SE3d T;
  T = cameras.at(tcid0).T_w_c.inverse() * cameras.at(tcid1).T_w_c;

  // create a central relative adapter
  // (immediately pass translation and rotation)
  relative_pose::CentralRelativeAdapter adapter(
      bearingVectors0, bearingVectors1, T.translation(), T.rotationMatrix());

  for (uint i = 0; i < new_track_ids.size(); ++i) {
    // Compute point
    point_t point = triangulation::triangulate(adapter, i);

    // Create new landmark
    Landmark landmark;
    landmark.p = cameras.at(tcid0).T_w_c * point;

    for (auto it = feature_tracks.at(new_track_ids[i]).begin();
         it != feature_tracks.at(new_track_ids[i]).end(); ++it) {
      auto find = cameras.find(it->first);
      if (find != cameras.end()) {
        landmark.obs[it->first] = it->second;
      }
    }
    landmarks[new_track_ids[i]] = landmark;
  }

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const TimeCamId &tcid0,
                                       const TimeCamId &tcid1,
                                       const Calibration &calib_cam,
                                       const Corners &feature_corners,
                                       const FeatureTracks &feature_tracks,
                                       Cameras &cameras, Landmarks &landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(tcid0.first == tcid1.first && tcid0.second != tcid1.second)) {
    std::cerr << "Images " << tcid0 << " and " << tcid1
              << " don't for a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // Initialize scene (add initial cameras and landmarks)

  // 1.1 Set left camera pose to identity
  Eigen::Matrix3d R_identity = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_zero = Eigen::Vector3d::Zero();
  cameras[tcid0].T_w_c = Sophus::SE3d(R_identity, t_zero);

  // 1.2 Set right camera pose to relative pose according to the extrinsic
  // camera calibration (note that cam0 is aligned with the IMU (i))
  cameras[tcid1].T_w_c = cameras[tcid0].T_w_c * calib_cam.T_i_c[tcid1.second];

  add_new_landmarks_between_cams(tcid0, tcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const TimeCamId &tcid, const std::vector<TrackId> &shared_track_ids,
    const Calibration &calib_cam, const Corners &feature_corners,
    const FeatureTracks &feature_tracks, const Landmarks &landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d &T_w_c, std::vector<TrackId> &inlier_track_ids) {
  inlier_track_ids.clear();

  // Localize a new image in a given map

  bearingVectors_t bearingVectors;
  points_t points;

  // BEARING VECTORS & POINTS
  // Go through all tracks shared between the image and the landmarks
  for (TrackId shared_track_id : shared_track_ids) {
    // Get the 3D point corresponding to the landmark with id ''trackId''
    point_t p = landmarks.at(shared_track_id).p;
    points.push_back(p);

    // Get the 2D coordinate corresponding to the track with id ''trackId''
    const FeatureId featureId = feature_tracks.at(shared_track_id).at(tcid);
    const Eigen::Vector2d p_2d = feature_corners.at(tcid).corners[featureId];
    // Add 2d coordinates to bearing vectors
    bearingVectors.push_back(
        calib_cam.intrinsics[tcid.second]->unproject(p_2d).normalized());
  }

  // DEFINE the OpenGV stuff
  // create the central adapter
  absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  // create a Ransac object
  sac::Ransac<sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter,
              sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));

  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.max_iterations_ = 500;
  ransac.computeModel();
  // get the result
  transformation_t best_transformation = ransac.model_coefficients_;

  const Eigen::Matrix3d R = best_transformation.block<3, 3>(0, 0);
  const Eigen::Vector3d t = best_transformation.block<3, 1>(0, 3);

  // Run non-linear optimization until number of inliers does not increase any
  // more

  int prev_num_inliers = 0;

  // Define the refined pose
  Eigen::Matrix3d R_refined = R;
  Eigen::Vector3d t_refined = t;

  while (ransac.inliers_.size() > prev_num_inliers) {
    /////////////////////////////////////////////////////////////////////////////
    // NON-LINEAR OPTIMIZATION (using all available correspondences)
    adapter.setR(R_refined);
    adapter.sett(t_refined);
    transformation_t nonlinear_transformation =
        absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

    R_refined = nonlinear_transformation.block<3, 3>(0, 0);
    t_refined = nonlinear_transformation.block<3, 1>(0, 3);
    /////////////////////////////////////////////////////////////////////////////

    // Get previous number of inliers
    prev_num_inliers = ransac.inliers_.size();

    /////////////////////////////////////////////////////////////////////////////
    // UPDATE SET OF INLIERS using the refined pose
    ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                            ransac.threshold_, ransac.inliers_);
    /////////////////////////////////////////////////////////////////////////////
  }

  // This is the pose we return
  T_w_c = Sophus::SE3d(R_refined, t_refined);

  // Fill in the inlier_track_ids vector that we return
  for (auto inlier_idx : ransac.inliers_) {
    TrackId inlier_track_id = shared_track_ids[inlier_idx];
    inlier_track_ids.push_back(inlier_track_id);
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners &feature_corners,
                       const BundleAdjustmentOptions &options,
                       const std::set<TimeCamId> &fixed_cameras,
                       Calibration &calib_cam, Cameras &cameras,
                       Landmarks &landmarks) {
  ceres::Problem problem;

  // Get name of camera model
  std::string name =
      calib_cam.intrinsics.at(fixed_cameras.begin()->second)->name();

  std::cout << "Name of camera model is " << name << std::endl;

  // Fix intrinsics for left and right camera
  for (int i = 0; i < 2; ++i) {
    problem.AddParameterBlock(calib_cam.intrinsics[i]->data(), 8);
    problem.SetParameterBlockConstant(calib_cam.intrinsics[i]->data());
  }

  // ALL Camera poses
  for (auto &camera : cameras) {
    problem.AddParameterBlock(camera.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
    // Fix first two cameras
    if (fixed_cameras.count(camera.first) == 1) {
      problem.SetParameterBlockConstant(camera.second.T_w_c.data());
    }
  }

  std::cout << "Let's optimize" << std::endl;

  // SET UP OPTIMIZATION PROBLEM

  // Go through every LANDMARK
  for (auto &kl : landmarks) {
    TrackId trackId = kl.first;
    Landmark &landmark = kl.second;

    // Go through every OBSERVATION of the current landmard
    // (through every camera observing the landmark)
    for (auto &obs : landmark.obs) {
      TimeCamId tcid = obs.first;
      FeatureId featureId = obs.second;

      // Get corresponding 2d
      Eigen::Vector2d kp = feature_corners.at(tcid).corners.at(featureId);

      ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, 3, 8>(
          new BundleAdjustmentReprojectionCostFunctor(kp, name));

      ceres::LossFunction *loss_function;
      if (options.use_huber)
        loss_function = new ceres::HuberLoss(options.huber_parameter);
      else
        loss_function = NULL;

      problem.AddResidualBlock(cost_function, loss_function,
                               cameras.at(tcid).T_w_c.data(), landmark.p.data(),
                               calib_cam.intrinsics[tcid.second]->data());
    }
  }

  std::cout << "Solving ceres problem..." << std::endl;

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}
