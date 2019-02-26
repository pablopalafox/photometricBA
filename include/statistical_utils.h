#pragma once

#include "vector"

#include <math.h>

#include "common_types.h"

struct Bucket {
  int freq = 0;
  std::unordered_map<TimeCamId, invDistAndPoint> best_matches;
};

double max_inverse_distance(
    const std::unordered_map<TimeCamId, invDistAndPoint>& matches) {
  double max_inv_distance = -1;
  for (const auto& match : matches) {
    if (match.second.first > max_inv_distance)
      max_inv_distance = match.second.first;
  }
  return max_inv_distance;
}

double compute_most_frequent(
    std::unordered_map<TimeCamId, invDistAndPoint>& matches) {
  if (matches.size() == 0) return std::numeric_limits<double>::quiet_NaN();

  double max_inv_dist = max_inverse_distance(matches);

  const double bucket_size = 0.001;
  int number_of_buckets = (int)ceil(max_inv_dist / bucket_size);
  std::vector<Bucket> histogram(number_of_buckets);

  for (auto& match : matches) {
    int bucket = static_cast<int>(floor(match.second.first / bucket_size));
    histogram[bucket].freq += 1;
    histogram[bucket].best_matches[match.first] =
        std::make_pair(match.second.first, match.second.second);
  }

  // We will reuse this vector to finally store the matched 2d points
  matches.clear();

  double most_frequent = std::numeric_limits<double>::quiet_NaN();
  int freq_of_most_frequent = -1;
  int best_bucket = -1;
  for (int bucket = 0; bucket < number_of_buckets; ++bucket) {
    if (histogram[bucket].freq > freq_of_most_frequent) {
      freq_of_most_frequent = histogram[bucket].freq;
      best_bucket = bucket;
    }
  }

  most_frequent = double(best_bucket) * bucket_size;
  matches = histogram[best_bucket].best_matches;

  if (freq_of_most_frequent < 6)
    most_frequent = std::numeric_limits<double>::quiet_NaN();

  return most_frequent;
}
