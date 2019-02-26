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

#include <Eigen/Dense>

struct AprilGrid {
  AprilGrid() {
    const int tagCols = 6;          // number of apriltags
    const int tagRows = 6;          // number of apriltags
    const double tagSize = 0.088;   // size of apriltag, edge to edge [m]
    const double tagSpacing = 0.3;  // ratio of space between tags to tagSize

    double x_corner_offsets[4] = {0, tagSize, tagSize, 0};
    double y_corner_offsets[4] = {0, 0, tagSize, tagSize};

    aprilgrid_corner_pos_3d.resize(tagCols * tagRows * 4);

    for (int y = 0; y < tagCols; y++) {
      for (int x = 0; x < tagRows; x++) {
        int tag_id = tagRows * y + x;
        double x_offset = x * tagSize * (1 + tagSpacing);
        double y_offset = y * tagSize * (1 + tagSpacing);

        for (int i = 0; i < 4; i++) {
          int corner_id = (tag_id << 2) + i;

          // WATCH OUT! pos_3d is a REFERENCE!!!!
          Eigen::Vector3d& pos_3d = aprilgrid_corner_pos_3d[corner_id];

          pos_3d[0] = x_offset + x_corner_offsets[i];
          pos_3d[1] = y_offset + y_corner_offsets[i];
          pos_3d[2] = 0;
        }
      }
    }
  }

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      aprilgrid_corner_pos_3d;
};
