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

#include <sophus/se3.hpp>
#include "common_types.h"

////////////////////////////////////////////////////////////////////////////////////
template <class T>
Eigen::Matrix<T, 3, 3> get_skew_symmetric_matrix(const Eigen::Matrix<T, 3, 1>& a) {
    Eigen::Matrix<T, 3, 3> a_;
    a_ <<    T(0), -a(2,0),  a(1,0),
           a(2,0),    T(0), -a(0,0),
          -a(1,0),  a(0,0),    T(0);
    return a_;
}
////////////////////////////////////////////////////////////////////////////////////

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(const Eigen::Matrix<T, 3, 1>& xi)
{
    const T theta = xi.norm();
    const Eigen::Matrix<T, 3, 1> a = xi.normalized();
    Eigen::Matrix<T, 3, 3> res;
    res = cos(theta) * Eigen::Matrix<T, 3, 3>::Identity() + (1 - cos(theta)) * a * a.transpose() + sin(theta) * get_skew_symmetric_matrix(a);
    return res;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(const Eigen::Matrix<T, 3, 3>& mat)
{
    T theta = acos((mat.trace() - T(1)) / T(2));

    Eigen::Matrix<T, 3, 1> a;
    a << mat(2,1) - mat(1,2),
         mat(0,2) - mat(2,0),
         mat(1,0) - mat(0,1);
    if (theta != T(0)) {
        a = (T(1) / (T(2) * sin(theta))) * a;
    }
    else {
        return a;
    }

    Eigen::Matrix<T, 3, 1> res;
    res = a * theta;
    return res;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(const Eigen::Matrix<T, 6, 1>& xi)
{
    if (xi.isZero(0)) {
        return Eigen::Matrix<T, 4, 4>::Identity();
    }

    const Eigen::Matrix<T, 3, 1> rho = xi.block(0, 0, 3, 1);
    const Eigen::Matrix<T, 3, 1> phi = xi.block(3, 0, 3, 1);

//    std::cout << "rho " << rho << std::endl;
//    std::cout << "phi " << phi << std::endl;

    const T theta = phi.norm();
    const Eigen::Matrix<T, 3, 1> a = phi.normalized();

    // Get an identity matrix (just to simplify stuff)
    Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();

    // Rotation part
    Eigen::Matrix<T, 3, 3> phi_;
    phi_ = cos(theta) * I + (1 - cos(theta)) * a * a.transpose() + sin(theta) * get_skew_symmetric_matrix(a);

//    std::cout << "phi_: " << phi_ << std::endl;

    // Translation part
    Eigen::Matrix<T, 3, 3> J = (sin(theta) / theta) * I + (1 - (sin(theta) / theta)) * a * a.transpose() + ((1 - cos(theta)) / theta) * get_skew_symmetric_matrix(a);
    Eigen::Matrix<T, 3, 1> Jrho = J * rho;
//    std::cout << "Jrho: " << Jrho << std::endl;

    Eigen::Matrix<T, 4, 4> res;
    res.setZero();
    res.block(0, 0, 3, 3) = phi_;
    res.block(0, 3, 3, 1) = J * rho;
    res(3, 3) = T(1);
    return res;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(const Eigen::Matrix<T, 4, 4>& mat)
{
    Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
    Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);

    T theta = acos((R.trace() - T(1)) / T(2));

    Eigen::Matrix<T, 3, 1> a;
    a << R(2,1) - R(1,2),
         R(0,2) - R(2,0),
         R(1,0) - R(0,1);
    if (theta != T(0)) {
        a = (T(1) / (T(2) * sin(theta))) * a;
    }
    else {
        return Eigen::Matrix<T, 6, 1>::Zero();
    }


    // Get an identity matrix (just to simplify stuff)
    Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();

    // Translation part
    Eigen::Matrix<T, 3, 3> J = (sin(theta) / theta) * I + (T(1) - (sin(theta) / theta)) * a * a.transpose() + ((T(1) - cos(theta)) / theta) * get_skew_symmetric_matrix(a);

    Eigen::Matrix<T, 6, 1> res;
    res.block(0, 0, 3, 1) = J.inverse() * t;
    res.block(3, 0, 3, 1) = a * theta;
    return res;
}
