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

#include <set>

#include "common_types.h"

#include "calibration.h"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

using namespace opengv;

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.

  for (const auto& kv : landmarks) {
    const TrackId track_id = kv.first;
    const Eigen::Vector3d p_3d_w = kv.second.p;

    // Transform from world space to camera space
    Eigen::Vector3d p_3d_c = current_pose.inverse() * p_3d_w;

    // Ignore points behind the camera
    if (p_3d_c.z() < cam_z_threshold) continue;

    // Project onto image plane
    Eigen::Vector2d p_2d = cam->project(p_3d_c);

    // Ignore points that project outside the image plane
    if (p_2d.x() < 0 || p_2d.x() > 751 || p_2d.y() < 0 || p_2d.y() > 479)
      continue;

    projected_points.emplace_back(p_2d);
    projected_track_ids.emplace_back(track_id);
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_max_dist,
    const double feature_match_test_next_best, MatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_max_dist and feature_match_test_next_best
  // should be used to filter outliers the same way as in exercise 3.

  // Go through all KEYPOINTS in the current image
  for (size_t i = 0; i < kdl.corners.size(); ++i) {
    int best_trackId = -1;
    int smallest_dist = 500;
    int second_smallest_dist = 500;

    const Eigen::Vector2d kp_2d = kdl.corners[i];
    const Descriptor descriptor = kdl.corner_descriptors.at(i);

    // Go through all PROJECTED POINTS (which correspond to different
    // LANDMARKS) and compute the landmark's descriptor
    for (size_t j = 0; j < projected_points.size(); ++j) {
      const Eigen::Vector2d p_proj = projected_points[j];
      const TrackId track_id = projected_track_ids[j];

      // Search for matches inside a circle around the detected keypoint
      if ((p_proj - kp_2d).squaredNorm() <
          (match_max_dist_2d * match_max_dist_2d)) {
        const Landmark lm = landmarks.at(track_id);

        // Var to store the min distance between descriptors
        int smallest_landmark_dist = 500;
        // Go through all OBSERVATIONS
        for (const auto& obs : lm.obs) {
          TimeCamId tcid = obs.first;
          FeatureId featureid = obs.second;
          KeypointsData kp_lm = feature_corners.at(tcid);
          Descriptor descriptor_lm = kp_lm.corner_descriptors.at(featureid);

          // Compute distance between descriptor_lm and the current descriptor
          int dist = (descriptor ^ descriptor_lm).count();

          // Get the descriptor with the smallest distance to our descriptor
          smallest_landmark_dist = std::min(smallest_landmark_dist, dist);
        }

        if (smallest_landmark_dist < smallest_dist) {
          second_smallest_dist = smallest_dist;
          smallest_dist = smallest_landmark_dist;
          best_trackId = projected_track_ids[j];
        } else if (smallest_landmark_dist < second_smallest_dist) {
          second_smallest_dist = smallest_landmark_dist;
        }
      }
    }

    // Now, some checks:
    if (best_trackId == -1) continue;
    if (smallest_dist >= feature_match_max_dist) continue;
    if (second_smallest_dist < (feature_match_test_next_best * smallest_dist))
      continue;

    md.matches.emplace_back(i, best_trackId);
  }
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     const MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  if (md.matches.size() == 0) {
    T_w_c = Sophus::SE3d();
    return;
  }

  // TODO SHEET 5: Find the pose (T_w_c) and the inliers using the landmark to
  // keypoints matches and PnP. This should be similar to the localize_camera in
  // exercise 4 but in this execise we don't explicitlly have tracks.

  bearingVectors_t bearingVectors;
  points_t points;

  // BEARING VECTORS & POINTS
  // Go through all tracks shared between the image and the landmarks
  for (const auto& match : md.matches) {
    FeatureId featureid = match.first;
    TrackId trackid = match.second;

    point_t p = landmarks.at(trackid).p;
    points.push_back(p);

    const Eigen::Vector2d p_2d = kdl.corners[featureid];
    bearingVectors.push_back(cam->unproject(p_2d).normalized());
  }

  absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  sac::Ransac<sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  std::shared_ptr<sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter,
              sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));

  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.max_iterations_ = 250;
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

  inliers = ransac.inliers_;
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       Landmarks& landmarks, TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains landmark to map
  // matches for the left camera (camera 0). The inliers vector contains all
  // inliers in md that were used to compute the pose T_w_c0. For all inliers
  // add the observations to the existing landmarks (if the left point is in
  // md_stereo.inliers then add both observations). For all stereo
  // observations
  // that were not added to the existing landmarks triangulate and add new
  // landmarks. Here next_landmark_id is a running index of the landmarks,
  // so
  // after adding a new landmark you should always increase next_landmark_id
  // by 1.

  // Convert stereo matches from vector to a map
  std::unordered_map<FeatureId, FeatureId> md_stereo_unorderedmap;
  for (const auto& kv : md_stereo.inliers) {
    md_stereo_unorderedmap[kv.first] = kv.second;
  }

  // Adding OBSERVATIONS to alredy present landmarks
  for (const int& inlier_idx : inliers) {
    FeatureId featureid_left_cam = md.matches[inlier_idx].first;
    TrackId trackid = md.matches[inlier_idx].second;

    Landmark& landmark = landmarks.at(trackid);
    landmark.obs[tcidl] = featureid_left_cam;

    auto it = md_stereo_unorderedmap.find(featureid_left_cam);
    if (it != md_stereo_unorderedmap.end()) {
      landmark.obs[tcidr] = it->second;
    }
    md_stereo_unorderedmap.erase(featureid_left_cam);
  }

  // Adding NEW LANDMARKS by triangulating position
  {
    bearingVectors_t bearingVectors0;
    bearingVectors_t bearingVectors1;

    std::vector<std::pair<FeatureId, FeatureId>> md_stereo_vector;

    for (const auto& kv : md_stereo_unorderedmap) {
      const FeatureId featureidl = kv.first;
      const FeatureId featureidr = kv.second;

      const Eigen::Vector2d p0_2d = kdl.corners[featureidl];
      const Eigen::Vector2d p1_2d = kdr.corners[featureidr];

      bearingVectors0.push_back(
          calib_cam.intrinsics[tcidl.second]->unproject(p0_2d).normalized());
      bearingVectors1.push_back(
          calib_cam.intrinsics[tcidr.second]->unproject(p1_2d).normalized());

      md_stereo_vector.emplace_back(kv.first, kv.second);
    }

    // create a central relative adapter
    relative_pose::CentralRelativeAdapter adapter(
        bearingVectors0, bearingVectors1, t_0_1, R_0_1);

    for (uint i = 0; i < bearingVectors0.size(); ++i) {
      // Compute 3d coord in camera space of left camera
      point_t point = triangulation::triangulate(adapter, i);

      // Create new landmark (we want 3d coord to be in world space)
      Landmark landmark;
      landmark.p = T_w_c0 * point;
      landmark.obs[tcidl] = md_stereo_vector.at(i).first;
      landmark.obs[tcidr] = md_stereo_vector.at(i).second;

      landmarks[next_landmark_id] = landmark;
      next_landmark_id++;
    }
  }
}

void remove_old_keyframes(const TimeCamId tcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(tcidl.first);

  // TODO SHEET 5: Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.

  while (kf_frames.size() > max_num_kfs) {
    // kf_frames has as first element the oldest keyframe added
    FrameId kfid_to_remove = *(kf_frames.begin());
    kf_frames.erase(kfid_to_remove);

    // Remove camera from cameras
    TimeCamId left_cam_to_remove = TimeCamId(kfid_to_remove, 0);
    TimeCamId right_cam_to_remove = TimeCamId(kfid_to_remove, 1);
    cameras.erase(left_cam_to_remove);
    cameras.erase(right_cam_to_remove);

    std::set<TrackId> remove_landmarks;

    // Remove observations
    for (auto& kv : landmarks) {
      Landmark& lm = kv.second;

      auto lit = lm.obs.find(left_cam_to_remove);
      if (lit != lm.obs.end()) lm.obs.erase(lit);

      auto rit = lm.obs.find(right_cam_to_remove);
      if (rit != lm.obs.end()) lm.obs.erase(rit);

      if (lm.obs.empty()) {
        remove_landmarks.emplace(kv.first);
      }
    }
    for (const auto lm_trackid : remove_landmarks) {
      old_landmarks[lm_trackid] = landmarks[lm_trackid];
      landmarks.erase(lm_trackid);
    }
  }
}
