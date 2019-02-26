#pragma once

#include <iostream>

#include "Eigen/Dense"

#include "common_types.h"

using std::sqrt;

Eigen::VectorXd align_trajectories(const Cameras &target_cameras,
                                   const Cameras &source_cameras,
                                   Cameras &aligned_source_cameras,
                                   Eigen::Matrix3d &R_align,
                                   Eigen::Vector3d &t_align) {
  // Align two trajectories using the method of Horn (closed-form).
  // Find rotation and traslation to align source_cameras to target_cameras

  int num_cams_used = 0;

  // TARGET (Y) TRAJECTORY //
  Eigen::Matrix3Xd target;
  for (const auto &cam : target_cameras) {
    if (cam.first.first == 0) continue;
    num_cams_used++;
    // Get camera position and add to target matrix
    Eigen::Vector3d cam_position = cam.second.T_w_c.translation();
    target.conservativeResize(Eigen::NoChange, target.cols() + 1);
    target.col(target.cols() - 1) = cam_position;
  }
  Eigen::Vector3d target_mean = target.rowwise().mean();
  Eigen::Matrix3Xd target_covariance = target.colwise() - target_mean;

  // SOURCE (X) TRAJECTORY //
  Eigen::Matrix3Xd source;
  for (const auto &cam : source_cameras) {
    if (cam.first.first == 0) continue;
    // Get camera position and add to source matrix
    Eigen::Vector3d cam_position = cam.second.T_w_c.translation();
    source.conservativeResize(Eigen::NoChange, source.cols() + 1);
    source.col(source.cols() - 1) = cam_position;
  }
  Eigen::Vector3d source_mean = source.rowwise().mean();
  Eigen::Matrix3Xd source_covariance = source.colwise() - source_mean;

  // HORN METHOD //
  Eigen::Matrix3d cross_covariance =
      target_covariance * source_covariance.transpose();

  // We now compute the SVD of the cross_variance (Y X^t = U S V^t)
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      cross_covariance, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
  if ((svd.matrixU().determinant() * svd.matrixV().determinant()) < 0) {
    S(2, 2) = -1.0;
  }
  // R = U V^t
  Eigen::Matrix3d R = svd.matrixU() * S * svd.matrixV().transpose();
  Eigen::Vector3d t = target_mean - R * source_mean;

  R_align = R;
  t_align = t;

  // Align source to target
  Eigen::Matrix3Xd source_aligned;
  for (const auto &cam : source_cameras) {
    TimeCamId tcid = cam.first;
    Sophus::SE3d T_w_c = cam.second.T_w_c;

    // Update translation and rotation of source pose
    T_w_c.translation() = R * T_w_c.translation() + t;
    T_w_c.so3() = Sophus::SO3d(R) * T_w_c.so3();
    aligned_source_cameras[tcid].T_w_c = T_w_c;

    if (cam.first.first == 0) continue;
    // Compute matrix of alignment camera centers
    source_aligned.conservativeResize(Eigen::NoChange,
                                      source_aligned.cols() + 1);
    source_aligned.col(source_aligned.cols() - 1) = T_w_c.translation();
  }

  // Compute aligment error
  Eigen::Matrix3Xd alignment_error = source_aligned - target;
  alignment_error = alignment_error.array().square();
  Eigen::VectorXd sum_of_squares = alignment_error.colwise().sum();
  Eigen::VectorXd trans_error = sum_of_squares.array().sqrt();

  std::cout << "We used " << num_cams_used << " cameras to compute alignment."
            << std::endl;

  return trans_error;
}
