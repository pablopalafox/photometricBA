#pragma once

#include "vector"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_types.h"

void compute_image_gradients(const std::vector<double>& img_raw, const size_t h,
                             const size_t w, std::vector<double>& grad) {
  for (int y = 1; y < h - 1; y++)
    for (int x = 1; x < w - 1; x++) {
      int idx = x + y * w;
      double dx = 0.5 * (img_raw[idx + 1] - img_raw[idx - 1]);
      double dy = 0.5 * (img_raw[idx + w] - img_raw[idx - w]);

      if (!std::isfinite(dx)) dx = 0;
      if (!std::isfinite(dy)) dy = 0;

      grad[idx] = sqrt(dx * dx + dy * dy);
    }
}

double medianMat(const cv::Mat& Input, int nVals) {
  // COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
  float range[] = {0, float(nVals)};
  const float* histRange = {range};
  cv::Mat hist;
  cv::calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange);

  // COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
  cv::Mat cdf;
  hist.copyTo(cdf);
  for (int i = 1; i <= nVals - 1; i++) {
    cdf.at<float>(i) += cdf.at<float>(i - 1);
  }
  cdf /= Input.total();

  // COMPUTE MEDIAN
  double medianVal = 0.0;
  for (int i = 0; i <= nVals - 1; i++) {
    if (cdf.at<float>(i) >= 0.5) {
      medianVal = i;
      break;
    }
  }
  return medianVal;
}

void select_photo_candidates(const std::vector<double>& img_raw, const size_t h,
                             const size_t w, PhotoCandidatePointsData& pcp) {
  // Compute image gradients
  std::vector<double> squared_grad(w * h, 0.0);
  compute_image_gradients(img_raw, h, w, squared_grad);

  // Find maximum gradient of image
  auto it = std::max_element(std::begin(squared_grad), std::end(squared_grad));
  double max_grad = *it;

  // Convert gradient image from double to uint8_t (unsigned char)
  std::vector<uint8_t> img_uint;
  for (size_t i = 0; i < squared_grad.size(); i++) {
    img_uint.push_back(
        static_cast<uint8_t>(round(squared_grad[i] / max_grad * 255)));
  }

  // Create opencv image with the uint8_t vector of gradients
  cv::Mat grad_image(h, w, CV_8U, img_uint.data());

  double min, max;
  cv::minMaxLoc(grad_image, &min, &max);

  // Subdivide image into a total of 16 x 16 subregions
  int blocks = 8;  // 34 will be right for the current params
  int w16 = (w - 4) / blocks;
  int h16 = (h - 4) / blocks;

  // Go through all 16 x 16 blocks
  for (int j = 0; j < blocks; j++)
    for (int i = 0; i < blocks; i++) {
      int x = i * w16 + 2;
      int y = j * h16 + 2;

      // Extract current patch (or block)
      cv::Rect rect = cv::Rect(x, y, w16, h16);
      cv::Mat patch = cv::Mat(grad_image, rect);

      // Compute median absolute gradient over all pixels of current block
      double median = medianMat(patch, max);

      // Find maximum value and its location w.r.t. the current patch
      double max_in_block;
      cv::Point max_loc_in_block, max_loc_in_image;
      cv::minMaxLoc(patch, 0, &max_in_block, 0, &max_loc_in_block);

      // Compute location w.r.t. the whole image's coordinate system
      max_loc_in_image = cv::Point(x, y) + max_loc_in_block;

      if (max_in_block > (median + 7)) {
        // cv::circle(grad_image, max_loc_in_image, 2, cv::Scalar(255, 255, 0));

        pcp.selected_points.push_back(Eigen::Vector2d(
            double(max_loc_in_image.x), double(max_loc_in_image.y)));
      }

      // cv::rectangle(grad_image, rect, cv::Scalar(255, 0, 0), 1, 8, 0);
    }

  //  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  //  cv::imshow("Display window", grad_image);
  //  cv::waitKey();
}
