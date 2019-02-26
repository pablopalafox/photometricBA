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

#include <chrono>
#include <iostream>
#include <thread>

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include "local_parameterization_se3.hpp"

#include "aprilgrid.h"
#include "calibration.h"
#include "reprojection.h"

#include "serialization.h"

#include <CLI/CLI.hpp>

void drawImageOverlay(pangolin::View& v, size_t cam_id);
void load_data(const std::string& path);
void compute_projections();
void save_calib();
void optimize();

AprilGrid aprilgrid;

Calibration calib_cam;
std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vec_T_w_i;

tbb::concurrent_unordered_map<TimeCamId, CalibCornerData> calib_corners;
tbb::concurrent_unordered_map<TimeCamId, CalibInitPoseData> calib_init_poses;
tbb::concurrent_unordered_map<TimeCamId, pangolin::TypedImage> calib_images;

tbb::concurrent_unordered_map<TimeCamId, CalibCornerData> opt_corners;

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 1500);
pangolin::Var<bool> show_detected("ui.show_detected", true, false, true);
pangolin::Var<bool> show_opt("ui.show_opt", true, false, true);

std::string dataset_path;
std::string cam_model = "ds";

int main(int argc, char** argv) {
  const int UI_WIDTH = 200;
  const int NUM_CAMS = 2;

  bool show_gui = false;

  std::cout << "show " << show_gui << std::endl;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path, "Dataset path")->required();
  //  app.add_option("--dataset-path", dataset_path, "Dataset path");
  app.add_option(
      "--cam-model", cam_model,
      "Camera model. Possible values: pinhole, ds, eucm, kb4. Default: ds.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(dataset_path);
  compute_projections();

  if (show_gui) {
    //    pangolin::CreateWindowAndBind("Main", 1800, 1000);
    pangolin::CreateWindowAndBind("Main", 900, 500);

    pangolin::View* img_view_display;

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;

    img_view_display =
        &pangolin::CreateDisplay()
             .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
             .SetLayout(pangolin::LayoutEqual);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);
      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display->AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&drawImageOverlay, std::placeholders::_1, idx);
    }

    pangolin::Var<std::function<void(void)>> optimize_btn("ui.optimize",
                                                          std::bind(&optimize));
    pangolin::Var<std::function<void(void)>> save_calib_btn(
        "ui.save_calib", std::bind(&save_calib));

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (show_frame.GuiChanged()) {
        size_t frame_id = show_frame;

        for (size_t cam_id = 0; cam_id < NUM_CAMS; cam_id++) {
          TimeCamId tcid;
          tcid.first = frame_id;
          tcid.second = cam_id;
          if (calib_images.find(tcid) != calib_images.end()) {
            img_view[cam_id]->SetImage(calib_images[tcid]);
          } else {
            std::cout << "NO IMAGE" << std::endl;
            img_view[cam_id]->Clear();
          }
        }
      }

      pangolin::FinishFrame();

      // prevent the GUI from burning too much CPU when just idling
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  } else {
    std::cout << "Not showing GUI - only optim and save" << std::endl;
    optimize();
    save_calib();
  }

  return 0;
}

void drawImageOverlay(pangolin::View& v, size_t cam_id) {
  size_t frame_id = show_frame;

  TimeCamId tcid = std::make_pair(frame_id, cam_id);

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (calib_corners.find(tcid) != calib_corners.end()) {
      const CalibCornerData& cr = calib_corners.at(tcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, 20);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, 20);
    }
  }

  if (show_opt) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 1.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (opt_corners.find(tcid) != opt_corners.end()) {
      const CalibCornerData& cr = opt_corners.at(tcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(400, 20);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Too few corners detected").Draw(400, 20);
    }
  }
}

void load_data(const std::string& dataset_path) {
  std::string poses_path = dataset_path + "/init_poses.json";
  std::string corners_path = dataset_path + "/detected_corners.json";
  std::string calib_path = dataset_path + "/calibration-double-sphere.json";

  {
    std::ifstream os(poses_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_init_poses);
      std::cout << "Loaded " << calib_init_poses.size() << " poses "
                << std::endl;
    } else {
      std::cerr << "could not load poses from " << poses_path << std::endl;
    }
  }

  {
    std::ifstream os(corners_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_corners);
      std::cout << "Loaded " << calib_corners.size() << " corners "
                << std::endl;
    } else {
      std::cerr << "could not load corners from " << poses_path << std::endl;
    }
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    LoadCalibration<double, DoubleSphereCamera<double>> loaded_cam_calib;
    loaded_cam_calib.T_i_c.resize(2);
    loaded_cam_calib.intrinsics.resize(2);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(loaded_cam_calib);
      std::cout << "Loaded camera" << std::endl;

      calib_cam.T_i_c = loaded_cam_calib.T_i_c;

      calib_cam.intrinsics.clear();
      for (size_t i = 0; i < loaded_cam_calib.intrinsics.size(); i++) {
        calib_cam.intrinsics.push_back(AbstractCamera<double>::initialize(
            cam_model, loaded_cam_calib.intrinsics[i].data()));
      }

    } else {
      std::cerr << "could not load camera " << poses_path << std::endl;
    }
  }

  vec_T_w_i.resize(calib_corners.size() / 2);

  for (const auto& kv : calib_corners) {
    std::stringstream ss;
    ss << dataset_path << "/" << kv.first.first << "_" << kv.first.second
       << ".jpg";

    pangolin::TypedImage img = pangolin::LoadImage(ss.str());
    calib_images[kv.first] = std::move(img);

    if (calib_init_poses.find(kv.first) != calib_init_poses.end() &&
        kv.first.second == 0) {
      Sophus::SE3d pose = calib_init_poses[kv.first].T_a_c;
      vec_T_w_i[kv.first.first] = pose;
    }
  }

  show_frame.Meta().range[1] = calib_images.size() / 2 - 1;
  show_frame.Meta().gui_changed = true;
}

void compute_projections() {
  opt_corners.clear();

  for (const auto& kv : calib_corners) {
    CalibCornerData ccd;

    for (size_t i = 0; i < aprilgrid.aprilgrid_corner_pos_3d.size(); i++) {
      // Transformation from body (IMU) frame to world frame
      Sophus::SE3d T_w_i = vec_T_w_i[kv.first.first];

      // Transformation from camera to body (IMU) frame
      Sophus::SE3d T_i_c = calib_cam.T_i_c[kv.first.second];

      // 3D coordinates of the aprilgrid corner in the world frame
      Eigen::Vector3d p_3d = aprilgrid.aprilgrid_corner_pos_3d[i];

      // Bring point to camera space
      Eigen::Vector3d p_3d_cam = T_i_c.inverse() * T_w_i.inverse() * p_3d;

      // Project onto image plane
      Eigen::Vector2d p_2d;
      p_2d = calib_cam.intrinsics[kv.first.second]->project(p_3d_cam);

      ccd.corners.push_back(p_2d);
    }

    opt_corners[kv.first] = ccd;
  }
}

void optimize() {
  // Build the problem.
  ceres::Problem problem;

  // Specify local update rule for our parameter
  for (int i = 0; i < calib_cam.T_i_c.size(); ++i) {
    problem.AddParameterBlock(calib_cam.T_i_c[i].data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
  }

  problem.SetParameterBlockConstant(calib_cam.T_i_c[0].data());

  for (int i = 0; i < vec_T_w_i.size(); ++i) {
    problem.AddParameterBlock(vec_T_w_i[i].data(), Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
  }

  for (const auto& kv : calib_corners) {
    for (size_t i = 0; i < kv.second.corners.size(); i++) {
      int corner_idx = kv.second.corner_ids[i];
      Eigen::Vector3d p_3d = aprilgrid.aprilgrid_corner_pos_3d[corner_idx];

      ceres::CostFunction* cost_function =
          new ceres::AutoDiffCostFunction<ReprojectionCostFunctor, 2,
                                          Sophus::SE3d::num_parameters,
                                          Sophus::SE3d::num_parameters, 8>(
              new ReprojectionCostFunctor(kv.second.corners[i], p_3d,
                                          cam_model));

      problem.AddResidualBlock(cost_function, NULL,
                               vec_T_w_i[kv.first.first].data(),
                               calib_cam.T_i_c[kv.first.second].data(),
                               calib_cam.intrinsics[kv.first.second]->data());
    }
  }

  ceres::Solver::Options options;
  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  // Solve
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  {
    cereal::JSONOutputArchive archive(std::cout);
    archive(calib_cam);
    std::cout << std::endl;
  }

  compute_projections();
}

void save_calib() {
  std::ofstream os("opt_calib.json");

  if (os.is_open()) {
    cereal::JSONOutputArchive archive(os);
    archive(calib_cam);
    std::cout << "Saved camera calibration" << std::endl;
  }
  os.close();
}
