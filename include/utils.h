#pragma once

#include "common_types.h"

TrackId get_biggest_trackid(const PhotoLandmarks& photolandmarks) {
  // Get biggest track index
  std::map<TrackId, PhotoLandmark> photolm_ordered(photolandmarks.begin(),
                                                   photolandmarks.end());
  return photolm_ordered.rbegin()->first + 1;
}

template <class T>
int print_unordered_map(const T& unordered_map) {
  int size = unordered_map.size();
  for (const auto& kv : unordered_map) {
    std::cout << kv.first << " - " << kv.second << std::endl;
  }
  // std::cout << "Total: ";
  return size;
}
