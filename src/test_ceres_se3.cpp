/**
File adapted from Sophus

Copyright 2011-2017 Hauke Strasdat
          2012-2017 Steven Lovegrove

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/

#include <ceres/ceres.h>
#include <iostream>
#include <sophus/se3.hpp>

#include "local_parameterization_se3.hpp"

struct TestCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TestCostFunctor(Sophus::SE3d T_aw) : T_aw(T_aw) {}

  template <class T>
  bool operator()(T const* const sT_wa, T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_wa(sT_wa);
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(sResiduals);

    residuals = (T_aw.cast<T>() * T_wa).log();
    return true;
  }

  Sophus::SE3d T_aw;
};

bool test(Sophus::SE3d const& T_w_targ, Sophus::SE3d const& T_w_init) {
  // Optimisation parameter
  Sophus::SE3d T_wr = T_w_init;

  // Build the problem.
  ceres::Problem problem;

  // Specify local update rule for our parameter
  problem.AddParameterBlock(T_wr.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  // Create and add cost function. Derivatives will be evaluated via
  // automatic differentiation

  TestCostFunctor* c = new TestCostFunctor(T_w_targ.inverse());
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<TestCostFunctor, Sophus::SE3d::DoF,
                                      Sophus::SE3d::num_parameters>(c);
  problem.AddResidualBlock(cost_function, NULL, T_wr.data());

  // Set solver options (precision / method)
  ceres::Solver::Options options;
  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::DENSE_QR;

  // Solve
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // Difference between target and parameter
  double const mse = (T_w_targ.inverse() * T_wr).log().squaredNorm();
  bool const passed = mse < 10. * Sophus::Constants<double>::epsilon();
  return passed;
}

template <typename Scalar>
bool CreateSE3FromMatrix(Eigen::Matrix<Scalar, 4, 4> mat) {
  auto se3 = Sophus::SE3<Scalar>(mat);
  se3 = se3;
  return true;
}

int main(int, char**) {
  using SE3Type = Sophus::SE3<double>;
  using SO3Type = Sophus::SO3<double>;
  using Point = SE3Type::Point;
  double const kPi = Sophus::Constants<double>::pi();

  std::vector<SE3Type> se3_vec;
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)));

  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.2, 0.5, -1.0)), Point(10, 0, 0)));

  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.)), Point(0, 100, 5)));

  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0., 0., 0.00001)), Point(0, 0, 0)));

  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                            Point(0, -0.00000001, 0.0000000001)));

  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0., 0., 0.00001)), Point(0.01, 0, 0)));

  se3_vec.push_back(SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(4, -5, 0)));

  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(-0.2, -0.5, -0.0)), Point(0, 0, 0)));

  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.3, 0.5, 0.1)), Point(2, 0, -7)) *
      SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(-0.3, -0.5, -0.1)), Point(0, 6, 0)));

  for (size_t i = 0; i < se3_vec.size(); ++i) {
    std::cout << i << " " << (i + 3) % se3_vec.size() << std::endl;
    bool const passed = test(se3_vec[i], se3_vec[(i + 3) % se3_vec.size()]);
    if (!passed) {
      std::cerr << "failed!" << std::endl << std::endl;
      exit(-1);
    }
  }

  Eigen::Matrix<ceres::Jet<double, 28>, 4, 4> mat;
  mat.setIdentity();
  std::cout << CreateSE3FromMatrix(mat) << std::endl;

  return 0;
}
