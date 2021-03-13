#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H


// define the commonly included file to avoid a long include list
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;

// for Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
using SO3 = Sophus::SO3d;
using SE3 = Sophus::SE3d;

// for cv
#include <opencv2/core/core.hpp>
using cv::Mat;

// std
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using std::cout;
using std::cerr;

#endif