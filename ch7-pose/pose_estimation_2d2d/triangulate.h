#ifndef __TRIANGULATE_H__
#define __TRIANGULATE_H__

#include <opencv2/core.hpp>
#include <vector>

void Triangulation(const std::vector<cv::KeyPoint>& keypoint_1,
                   const std::vector<cv::KeyPoint>& keypoint_2,
                   const std::vector<cv::DMatch>& matches, const cv::Mat& K,
                   const cv::Mat& R, const cv::Mat& t,
                   std::vector<cv::Point3d>& points);

#endif  // !__TRIANGULATE_H__
