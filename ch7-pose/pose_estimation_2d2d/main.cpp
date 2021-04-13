#include <filesystem>
#include <fstream>
#include <iostream>
namespace fs = std::filesystem;
using namespace std;

#include <boost/format.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#include "TwoViewReconstruction.h"
#include "triangulate.h"

/* 3d car instance dataset...
camera.fx: 576.136966392455
camera.fy: 576.4689170155
camera.cx: 423.0
camera.cy: 339.0
*/

const cv::Mat1d& K() {
  static cv::Mat1d k(3, 3);
  static bool initialized = false;
  if (!initialized) {
    // k << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
    // K << 400, 0, 400, 0, 400, 400, 0, 0, 1;
    // k << 576.136966392455, 0, 423.0, 0, 576.4689170155, 339.0, 0, 0, 1;
    k << 2300.39065314361, 0, 1713.21615190657, 0, 2301.31478860597,
        1342.91100799715, 0, 0, 1;
    initialized = true;
  }
  return k;
}

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches(const Mat& img_1, const Mat& img_2, const Mat& mask1,
                          const Mat& mask2, std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches) {
  auto orb = ORB::create(2000);
  orb->detect(img_1, keypoints_1, mask1);
  orb->detect(img_2, keypoints_2, mask2);
  cv::Mat desp1, desp2;
  orb->compute(img_1, keypoints_1, desp1);
  orb->compute(img_2, keypoints_2, desp2);
  BFMatcher matcher(NORM_HAMMING);
  matcher.match(desp1, desp2, matches);
  // 排除一些不必要的匹配
  float min_dist = 1e6f, max_dist = 0.f;
  for (auto& dm : matches) {
    min_dist = min(min_dist, dm.distance);
    max_dist = max(max_dist, dm.distance);
  }
  vector<DMatch> good_match;
  for (auto& dm : matches) {
    if (dm.distance <= max(2 * min_dist, 0.5f * (min_dist + max_dist)))
      good_match.push_back(dm);
  }
  swap(good_match, matches);
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches, Mat& R, Mat& t) {
  vector<Point2f> pts1, pts2;
  for (auto& dm : matches) {
    pts1.emplace_back(keypoints_1[dm.queryIdx].pt);
    pts2.emplace_back(keypoints_2[dm.trainIdx].pt);
  }
  // compute F
  Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT);
  cout << "fundamental_matrix is " << endl << F << endl;
  // compute E
  Mat E = cv::findEssentialMat(pts1, pts2, K());
  cout << "essential_matrix is " << endl << E << endl;
  // compute H
  Mat H = cv::findHomography(pts1, pts2, RANSAC);
  cout << "homography_matrix is " << endl << H << endl;
  // 从 E 中得到旋转和平移
  cv::recoverPose(E, pts1, pts2, K(), R, t);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;
  //TODO：
  //1. 位姿估计时添加BA优化，以减小误差
}

// 像素坐标 => 归一化坐标
Mat1d pixel2cam(const Point2d& p, const Mat& K) {
  Mat1d mat(3, 1);
  mat << (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1), 1;
  return mat;
}

/****************************************************
 * 三角化生成点云数据
 * **************************************************/
void triangulation(const vector<KeyPoint>& keypoint_1,
                   const vector<KeyPoint>& keypoint_2,
                   const std::vector<DMatch>& matches, const Mat& R,
                   const Mat& t, vector<Point3d>& points) {
  Mat1f T1(3, 4), T2(3, 4);
  T1 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
  T2 << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      t.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(1, 1),
      R.at<double>(1, 2), t.at<double>(1, 0), R.at<double>(2, 0),
      R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0);
  vector<Point2f> pts1, pts2;  // 存归一化坐标
  for (DMatch m : matches) {
    auto cam1_pt = pixel2cam(keypoint_1[m.queryIdx].pt, K()),
         cam2_pt = pixel2cam(keypoint_2[m.trainIdx].pt, K());
    pts1.emplace_back((float)cam1_pt[0][0], (float)cam1_pt[1][0]);
    pts2.emplace_back((float)cam2_pt[0][0], (float)cam2_pt[1][0]);
  }

  // 4XN 矩阵，存 3D 几次坐标
  Mat pts;
  cv::triangulatePoints(T1, T2, pts1, pts2, pts);  // 输入矩阵必须是浮点类型
  // 转换成非齐次坐标
  for (int i = 0; i < pts.cols; i++) {
    Mat x = pts.col(i);
    x /= x.at<float>(3, 0);  // 归一化
    Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}

/****************************************************
 * 采用 ORB-SLAM 进行求解位姿和三角化！
 *****************************************************/
bool Reconstruct(const std::vector<cv::KeyPoint>& keypoints_1,
                 const std::vector<cv::KeyPoint>& keypoints_2,
                 const std::vector<cv::DMatch>& matches, cv::Mat& R21,
                 cv::Mat& t21, std::vector<cv::Point3f>& points3D,
                 std::vector<bool>& is_triangulated) {
  using namespace std;
  using namespace cv;
  vector<int> total_matches(keypoints_1.size(), -1);
  for (auto& m : matches) {
    total_matches[m.queryIdx] = m.trainIdx;
  }
  Mat1f k = K();
  ORB_SLAM3::TwoViewReconstruction r(k);
  return r.Reconstruct(keypoints_1, keypoints_2, total_matches, R21, t21,
                       points3D, is_triangulated);
}

void Test_Reconstruct() {
  //-- 读取图像
  auto img_name1 =
      R"(E:\datasets\3d_car_instance_sample\3d_car_instance_sample\images\180116_064519300_Camera_6.jpg)"s;
  auto img_name2 =
      R"(E:\datasets\3d_car_instance_sample\3d_car_instance_sample\images\180116_064519424_Camera_6.jpg)"s;
  Mat img_1 = imread(img_name1);
  Mat img_2 = imread(img_name2);
  Mat mask1 = imread((boost::format("%s_bg_mask.tiff") % img_name1).str(),
                     IMREAD_GRAYSCALE);
  Mat mask2 = imread((boost::format("%s_bg_mask.tiff") % img_name2).str(),
                     IMREAD_GRAYSCALE);
  //-- 提取特征点
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, mask1, mask2, keypoints_1, keypoints_2,
                       matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;
  //-- 重建和求解位姿
  Mat R21, t21;
  vector<Point3f> points;
  vector<bool> is_triangulated;
  bool ok = ::Reconstruct(keypoints_1, keypoints_2, matches, R21, t21, points,
                          is_triangulated);
  cout << "R is\n" << R21.t() << "\nt is\n" << -1 * t21 << endl;
  //-- 输出点云数据
  ofstream out("./cloud_points.txt");
  for (auto& pt : points) {
    out << boost::format("%f;%f;%f\n") % pt.x % pt.y % pt.z;
  }
}

int main1(int argc, char** argv) {
  //-- 读取图像
  // Mat img_1 = imread(R"(E:\git_repos\slambook\ch7\1.png)");
  // Mat img_2 = imread(R"(E:\git_repos\slambook\ch7\2.png)");
  // Mat img_1 = imread(R"(E:\Py_Projects\airsim_usage\scenes\scene_0.png)");
  // Mat img_2 = imread(R"(E:\Py_Projects\airsim_usage\scenes\scene_1.png)");
  auto img_name1 =
      R"(E:\datasets\3d_car_instance_sample\3d_car_instance_sample\images\180116_064519300_Camera_6.jpg)"s;
  auto img_name2 =
      R"(E:\datasets\3d_car_instance_sample\3d_car_instance_sample\images\180116_064519424_Camera_6.jpg)"s;
  Mat img_1 = imread(img_name1);
  Mat img_2 = imread(img_name2);
  Mat mask1 = imread((boost::format("%s_bg_mask.tiff") % img_name1).str(),
                     IMREAD_GRAYSCALE);
  Mat mask2 = imread((boost::format("%s_bg_mask.tiff") % img_name2).str(),
                     IMREAD_GRAYSCALE);
  // Mat1b mask1(480, 640, 255), mask2(480, 640, 255);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, mask1, mask2, keypoints_1, keypoints_2,
                       matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  //-- 估计两张图像间运动
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //-- 验证E=t^R*scale
  Mat t_x = (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
             t.at<double>(2, 0), 0, -t.at<double>(0, 0), -t.at<double>(1, 0),
             t.at<double>(0, 0), 0);

  cout << "t^R=" << endl << t_x * R << endl;

  //-- 验证对极约束
  vector<DMatch> good_matches;
  for (DMatch m : matches) {
    Mat1d y1 = pixel2cam(keypoints_1[m.queryIdx].pt, K()),
          y2 = pixel2cam(keypoints_2[m.trainIdx].pt, K());
    Mat d = y2.t() * t_x * R * y1;
    cout << "epipolar constraint = " << d << endl;
    if (d.at<double>(0) <= 1e-3) {
      good_matches.push_back(m);
    }
  }
  cout << "good matches size is " << good_matches.size() << endl;

  //-- 三角化！
  vector<Point3d> points;
  triangulation(keypoints_1, keypoints_2, good_matches, R, t, points);
  // Triangulation(keypoints_1, keypoints_2, good_matches, K(), R, t, points);

  //-- 验证三角化点和特征点的重投影关系！
  ofstream df("out_2d.txt");
  vector<Point3d> good_points;
  for (int i = 0; i < good_matches.size(); i++) {
    auto pt1_cam_ = pixel2cam(keypoints_1[good_matches[i].queryIdx].pt, K());
    Point2d pt1_cam(pt1_cam_[0][0], pt1_cam_[1][0]);
    Point2d pt1_cam_reproj(points[i].x / points[i].z,
                           points[i].y / points[i].z);

    cout << "point in the first camera frame: " << pt1_cam << endl;
    df << boost::format("%ld;%ld;%ld") % pt1_cam.x % pt1_cam.y % 0 << endl;
    cout << "point projected from 3D " << pt1_cam_reproj
         << ", d=" << points[i].z << endl;
    auto diff = pt1_cam - pt1_cam_reproj;
    if (diff.dot(diff) <= 1e-6) {
      good_points.emplace_back(points[i].x, points[i].y, points[i].z);
    }

    // 第二个图
    auto pt2_cam_ = pixel2cam(keypoints_2[good_matches[i].trainIdx].pt, K());
    Point2f pt2_cam(pt2_cam_[0][0], pt2_cam_[1][0]);
    Mat pt2_trans =
        R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    pt2_trans /= pt2_trans.at<double>(2, 0);
    cout << "point in the second camera frame: " << pt2_cam << endl;
    cout << "point reprojected from second frame: " << pt2_trans.t() << endl;
    cout << endl;
  }
  cout << "good points size is " << good_points.size() << endl;

  //-- 输出三角点
  // 计算深度的一半
  double depth = 0.;
  for (auto& p : good_points) {
    depth = max(depth, p.z);
  }
  depth /= 2;
  cout << "half depth maximum is" << depth << endl;
  ofstream f("out.txt");
  for (auto& p : good_points) {
    f << boost::format("%ld;%ld;%ld") % (p.x * depth) % (p.y * depth) %
             (p.z * depth)
      << endl;
  }

  cv::Mat show;
  cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, show);
  imshow("www", show);
  cv::waitKey(0);

  return 0;
}

/****************************************************
 *                    M A I N
 * **************************************************/
int main(int argc, char** argv) {
  Test_Reconstruct();
  main1(argc, argv);
  return 0;
}