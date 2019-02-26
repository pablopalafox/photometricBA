/**
Pablo Rodriguez-Palafox
*/

#pragma once

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

/*
 * A class to read data from a csv file of the euroc dataset
 */
class CSVReader {
  std::string fileName;
  std::string delimeter;

 public:
  CSVReader(const std::string filename, std::string delm = ",")
      : fileName(filename), delimeter(delm) {}

  // Helper function to convert a vector of strings to vector of doubles
  std::vector<double> stringVector_to_doubleVector(
      const std::vector<std::string> &stringVector) {
    std::vector<double> doubleVector(stringVector.size());
    std::transform(stringVector.begin(), stringVector.end(),
                   doubleVector.begin(),
                   [](const std::string &val) { return std::stod(val); });
    return doubleVector;
  }

  // Parses through csv file line by line and returns the data
  // in vector of vector of strings.
  std::map<int64_t, std::vector<double>> get_data() {
    std::ifstream file(fileName);

    std::map<int64_t, std::vector<double>> dataList;

    std::string header = "";
    getline(file, header);

    std::string line = "";
    while (getline(file, line)) {
      std::vector<std::string> stringVec;
      boost::algorithm::split(stringVec, line, boost::is_any_of(delimeter));

      int64_t timestamp = int64_t(std::stol(stringVec[0]));
      stringVec.erase(stringVec.begin());    // remove timestamp
      stringVec.erase(stringVec.begin() + 7, /*getting only poses*/
                      stringVec.end());      // remove extra data

      std::vector<double> doubleVec = stringVector_to_doubleVector(stringVec);
      dataList[timestamp] = doubleVec;
    }

    file.close();

    return dataList;
  }
};
