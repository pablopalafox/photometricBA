#pragma once

#include <fstream>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include "calibration.h"
#include "common_types.h"
#include "local_parameterization_se3.hpp"
#include "reprojection.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Pattern
Eigen::Vector2d pattern[8] = {Eigen::Vector2d(0, 0),   Eigen::Vector2d(0, 2),
                              Eigen::Vector2d(-1, 1),  Eigen::Vector2d(-2, 0),
                              Eigen::Vector2d(-1, -1), Eigen::Vector2d(0, -2),
                              Eigen::Vector2d(1, -1),  Eigen::Vector2d(2, 0)};

struct PhotoBundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 2;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// update affine transformation parameters
  bool optimize_affine = false;

  /// use huber robust norm or squared norm
  bool use_huber = false;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void photo_bundle_adjustment(
    const PhotoBundleAdjustmentOptions &options,
    const std::set<TimeCamId> &fixed_cameras, const int NUM_CAMS,
    const Interpolators &intensities, Calibration &calib_cam, Cameras &cameras,
    PhotoLandmarks &photolandmarks, const Corners &corners,
    const PhotoCandidatePoints &candidates, std::vector<double> &residuals) {
  ceres::Problem problem;

  // Get name of camera model
  std::string cam_model =
      calib_cam.intrinsics.at(fixed_cameras.begin()->second)->name();
  std::cout << "Name of camera model is " << cam_model << std::endl;

  //////////////////////////////////////////////////////////////////
  /// ADDING PARAMETER BLOCKS
  //////////////////////////////////////////////////////////////////
  // INTRINSICS (left and right cameras)
  for (int i = 0; i < NUM_CAMS; ++i) {
    problem.AddParameterBlock(calib_cam.intrinsics[i]->data(), 8);
    if (!options.optimize_intrinsics) {
      // Set intrinsics constant
      problem.SetParameterBlockConstant(calib_cam.intrinsics[i]->data());
    }
  }

  // CAMERAS
  for (auto &camera : cameras) {
    /// Poses
    problem.AddParameterBlock(camera.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);

    // Fix pose for first two cameras
    if (fixed_cameras.count(camera.first) == 1)
      problem.SetParameterBlockConstant(camera.second.T_w_c.data());

    /// Affine transformation params
    problem.AddParameterBlock(camera.second.affine_ab.data(), 2);

    // fix ALL affine params if we don't want to optimize for them
    if (!options.optimize_affine) {
      std::cout << "Setting ALL cameras' affine params as constant!"
                << std::endl;
      problem.SetParameterBlockConstant(camera.second.affine_ab.data());
    }
    // fixe only first affine params in case we do want to optimize for them
    TimeCamId first(0, 0);
    if (options.optimize_affine && camera.first == first)
      problem.SetParameterBlockConstant(camera.second.affine_ab.data());
  }

  //////////////////////////////////////////////////////////////////
  /// OPTIMIZING
  //////////////////////////////////////////////////////////////////
  // SET UP OPTIMIZATION PROBLEM
  std::cout << "Let's optimize" << std::endl;

  ceres::LossFunction *loss_function;
  if (options.use_huber)
    loss_function = new ceres::HuberLoss(options.huber_parameter);
  else
    loss_function = NULL;

  int count = 0;

  // Go through every PHOTOLANDMARK
  for (auto &kl : photolandmarks) {
    TrackId trackId = kl.first;
    PhotoLandmark &photolm = kl.second;
    bool selected = photolm.selected;
    TimeCamId host_tcid = photolm.host.first;
    FeatureId host_featureid = photolm.host.second;

    if (!selected) {
      // Go through every OBSERVATION of the current photolandmard
      for (const auto &obs : photolm.obs) {
        TimeCamId obs_tcid = obs.first;
        FeatureId obs_featureid = obs.second;

        count++;

        if (host_tcid.second == obs_tcid.second) {
          ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<
              PhotoBundleAdjustmentPhotometricCostFunctor, PIXELS_IN_PATCH,
              Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 1, 2,
              2, 8>(new PhotoBundleAdjustmentPhotometricCostFunctor(
              photolm.patch, intensities.at(obs_tcid), cam_model,
              corners.at(host_tcid).corners[host_featureid],
              corners.at(obs_tcid).corners[obs_featureid]));

          problem.AddResidualBlock(
              cost_function, loss_function, cameras.at(host_tcid).T_w_c.data(),
              cameras.at(obs_tcid).T_w_c.data(), &photolm.d,
              cameras.at(host_tcid).affine_ab.data(),
              cameras.at(obs_tcid).affine_ab.data(),
              calib_cam.intrinsics[host_tcid.second]->data());
        } else {
          ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<
              PhotoBundleAdjustmentPhotometricCostFunctor, PIXELS_IN_PATCH,
              Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 1, 2,
              2, 8, 8>(new PhotoBundleAdjustmentPhotometricCostFunctor(
              photolm.patch, intensities.at(obs_tcid), cam_model,
              corners.at(host_tcid).corners[host_featureid],
              corners.at(obs_tcid).corners[obs_featureid]));

          problem.AddResidualBlock(
              cost_function, loss_function, cameras.at(host_tcid).T_w_c.data(),
              cameras.at(obs_tcid).T_w_c.data(), &photolm.d,
              cameras.at(host_tcid).affine_ab.data(),
              cameras.at(obs_tcid).affine_ab.data(),
              calib_cam.intrinsics[host_tcid.second]->data(),
              calib_cam.intrinsics[obs_tcid.second]->data());
        }
      }
    } else {
      // Go through every OBSERVATION of the current SELECTED photolandmard
      for (const auto &obs : photolm.obs) {
        TimeCamId obs_tcid = obs.first;
        FeatureId obs_featureid = obs.second;

        count++;

        if (host_tcid.second == obs_tcid.second) {
          ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<
              PhotoBundleAdjustmentPhotometricCostFunctor, PIXELS_IN_PATCH,
              Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 1, 2,
              2, 8>(new PhotoBundleAdjustmentPhotometricCostFunctor(
              photolm.patch, intensities.at(obs_tcid), cam_model,
              candidates.at(host_tcid).selected_points[host_featureid],
              candidates.at(obs_tcid).matched_points[obs_featureid]));

          problem.AddResidualBlock(
              cost_function, loss_function, cameras.at(host_tcid).T_w_c.data(),
              cameras.at(obs_tcid).T_w_c.data(), &photolm.d,
              cameras.at(host_tcid).affine_ab.data(),
              cameras.at(obs_tcid).affine_ab.data(),
              calib_cam.intrinsics[host_tcid.second]->data());
        } else {
          ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<
              PhotoBundleAdjustmentPhotometricCostFunctor, PIXELS_IN_PATCH,
              Sophus::SE3d::num_parameters, Sophus::SE3d::num_parameters, 1, 2,
              2, 8, 8>(new PhotoBundleAdjustmentPhotometricCostFunctor(
              photolm.patch, intensities.at(obs_tcid), cam_model,
              candidates.at(host_tcid).selected_points[host_featureid],
              candidates.at(obs_tcid).matched_points[obs_featureid]));

          problem.AddResidualBlock(
              cost_function, loss_function, cameras.at(host_tcid).T_w_c.data(),
              cameras.at(obs_tcid).T_w_c.data(), &photolm.d,
              cameras.at(host_tcid).affine_ab.data(),
              cameras.at(obs_tcid).affine_ab.data(),
              calib_cam.intrinsics[host_tcid.second]->data(),
              calib_cam.intrinsics[obs_tcid.second]->data());
        }
      }
    }
  }

  // std::cout << "Count: " << count << std::endl;
  std::cout << "Solving ceres problem..." << std::endl;

  /// Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  // ceres_options.minimizer_progress_to_stdout = 1;
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;  // summary.BriefReport()

  // Getting the vector of Residuals
  //  ceres::Problem::EvaluateOptions evaluateOptions;
  //  evaluateOptions.apply_loss_function = false;
  //  if (problem.Evaluate(evaluateOptions, NULL, &residuals, NULL, NULL)) {
  //    std::cout << "Evaluating residuals w/o Huber Norm" << std::endl;
  //    std::cout << "Number of residuals " << residuals.size() << std::endl;
  //    std::cout << "Number of residual blocks " << residuals.size() / 8
  //              << std::endl;
  //    for (auto i = residuals.begin(); i != residuals.end(); ++i) {
  //      std::cout << "\nResidual " << *i << std::endl;
  //    }
  //  }
}
