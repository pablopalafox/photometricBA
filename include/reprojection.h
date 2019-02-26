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

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <ceres/ceres.h>
#include "ceres/cubic_interpolation.h"

#include "common_types.h"

template <class T>
class AbstractCamera;

struct ReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                          const Eigen::Vector3d& p_3d,
                          const std::string& cam_model)
      : p_2d(p_2d), p_3d(p_3d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_i_c,
                  T const* const sIntr, T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_i_c(sT_i_c);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);

    Eigen::Matrix<T, 3, 1> p_3d_cam;
    p_3d_cam = T_i_c.inverse() * T_w_i.inverse() * p_3d.cast<T>();
    residuals = p_2d.cast<T>() - cam->project(p_3d_cam);

    return true;
  }

  Eigen::Vector2d p_2d;
  Eigen::Vector3d p_3d;
  std::string cam_model;
};

struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                                          const std::string& cam_model)
      : p_2d(p_2d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_c, T const* const sp_3d_w,
                  T const* const sIntr, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T_w_c(sT_w_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3d_w(sp_3d_w);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // Compute reprojection error
    Eigen::Matrix<T, 3, 1> p_3d_c;
    p_3d_c = T_w_c.inverse() * p_3d_w;

    // if (p_3d_c[2] < 0.1) return false;

    residuals = p_2d.cast<T>() - cam->project(p_3d_c);

    return true;
  }

  Eigen::Vector2d p_2d;
  std::string cam_model;
};

///////////////////////////////////////////////////////////////////////

struct PhotoBundleAdjustmentPhotometricCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PhotoBundleAdjustmentPhotometricCostFunctor(
      const Patch& patch_i,
      const std::shared_ptr<
          ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>>&
          compute_intensity_j,
      const std::string& cam_model, const Eigen::Vector2d& gt_p_i,
      const Eigen::Vector2d& gt_p_j)
      : patch_i(patch_i),
        compute_intensity_j(compute_intensity_j),
        cam_model(cam_model),
        gt_p_i(gt_p_i),
        gt_p_j(gt_p_j) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_w_j,
                  T const* const d_i, T const* const ab_i, T const* const ab_j,
                  T const* const sIntr, T* sResiduals) const {
    ////////////////////////////////
    // map inputs
    ////////////////////////////////
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_w_j(sT_w_j);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);
    Eigen::Map<Eigen::Matrix<T, PIXELS_IN_PATCH, 1>> residuals(sResiduals);
    const T a_i = ab_i[0];
    const T b_i = ab_i[1];
    const T a_j = ab_j[0];
    const T b_j = ab_j[1];
    ////////////////////////////////
    // map inputs
    ////////////////////////////////

    const Sophus::SE3<T> T_j_i = T_w_j.inverse() * T_w_i;

    const Eigen::Matrix<T, 3, PIXELS_IN_PATCH> P_i_3d =
        cam->unproject_many(patch_i.positions.cast<T>()) / T(*d_i);

    Eigen::Matrix<T, 3, PIXELS_IN_PATCH> P_j_3d;
    for (size_t p = 0; p < PIXELS_IN_PATCH; ++p) {
      P_j_3d.col(p) = T_j_i * P_i_3d.col(p);
    }

    const Eigen::Matrix<T, 2, PIXELS_IN_PATCH> P_j = cam->project_many(P_j_3d);

    ////////////////////////
    /// CHECK PROJECTION ///
    ////////////////////////
    for (size_t pid = 0; pid < PIXELS_IN_PATCH; ++pid) {
      const Eigen::Matrix<T, 2, 1>& p_2d = P_j.col(pid);
      if (p_2d.x() < 0.0 || p_2d.x() > 751.0 || p_2d.y() < 0.0 ||
          p_2d.y() > 479.0) {
        // std::cout << "OUTSIDE PLANE (A)!!!!" << std::endl;
        // std::cout << P_j << std::endl;
        residuals = Eigen::Matrix<T, PIXELS_IN_PATCH, 1>::Zero();
        return true;
      }
    }

    Eigen::Matrix<T, PIXELS_IN_PATCH, 1> I_j;
    for (size_t p = 0; p < PIXELS_IN_PATCH; ++p) {
      T i_j;
      compute_intensity_j->Evaluate(P_j.col(p).y(), P_j.col(p).x(), &i_j);
      I_j.row(p) << i_j;
    }

    residuals =
        (I_j.array() - b_j) -
        (exp(a_j) / exp(a_i)) * (patch_i.intensities.cast<T>().array() - b_i);

    return true;
  }

  // ------------------------------------------------- //

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_w_j,
                  T const* const d_i, T const* const ab_i, T const* const ab_j,
                  T const* const sIntr_i, T const* const sIntr_j,
                  T* sResiduals) const {
    ////////////////////////////////
    // map inputs
    ////////////////////////////////
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_w_j(sT_w_j);
    const std::shared_ptr<AbstractCamera<T>> cam_i =
        AbstractCamera<T>::from_data(cam_model, sIntr_i);
    const std::shared_ptr<AbstractCamera<T>> cam_j =
        AbstractCamera<T>::from_data(cam_model, sIntr_j);
    Eigen::Map<Eigen::Matrix<T, PIXELS_IN_PATCH, 1>> residuals(sResiduals);
    const T a_i = ab_i[0];
    const T b_i = ab_i[1];
    const T a_j = ab_j[0];
    const T b_j = ab_j[1];
    ////////////////////////////////
    // map inputs
    ////////////////////////////////

    const Sophus::SE3<T> T_j_i = T_w_j.inverse() * T_w_i;

    const Eigen::Matrix<T, 3, PIXELS_IN_PATCH> P_i_3d =
        cam_i->unproject_many(patch_i.positions.cast<T>()) / T(*d_i);

    Eigen::Matrix<T, 3, PIXELS_IN_PATCH> P_j_3d;
    for (size_t p = 0; p < PIXELS_IN_PATCH; ++p) {
      P_j_3d.col(p) = T_j_i * P_i_3d.col(p);
    }

    const Eigen::Matrix<T, 2, PIXELS_IN_PATCH> P_j =
        cam_j->project_many(P_j_3d);

    ////////////////////////
    /// CHECK PROJECTION ///
    ////////////////////////
    for (size_t pid = 0; pid < PIXELS_IN_PATCH; ++pid) {
      const Eigen::Matrix<T, 2, 1>& p_2d = P_j.col(pid);
      if (p_2d.x() < 0.0 || p_2d.x() > 751.0 || p_2d.y() < 0.0 ||
          p_2d.y() > 479.0) {
        // std::cerr << "OUTSIDE PLANE (B)!!!!" << std::endl;
        // std::cerr << P_j << std::endl << std::endl;
        residuals = Eigen::Matrix<T, PIXELS_IN_PATCH, 1>::Zero();
        return true;
      }
    }
    ////////////////////////
    /// CHECK PROJECTION ///
    ////////////////////////

    Eigen::Matrix<T, PIXELS_IN_PATCH, 1> I_j;
    for (size_t p = 0; p < PIXELS_IN_PATCH; ++p) {
      T i_j;
      compute_intensity_j->Evaluate(P_j.col(p).y(), P_j.col(p).x(), &i_j);
      I_j.row(p) << i_j;
    }

    residuals =
        (I_j.array() - b_j) -
        (exp(a_j) / exp(a_i)) * (patch_i.intensities.cast<T>().array() - b_i);

    return true;
  }

  std::string cam_model;
  Patch patch_i;
  std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>>
      compute_intensity_j;
  Eigen::Vector2d gt_p_i;
  Eigen::Vector2d gt_p_j;
};
