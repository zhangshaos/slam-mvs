#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;
using namespace std;

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches) {
  auto orb = ORB::create();
  orb->detect(img_1, keypoints_1);
  orb->detect(img_2, keypoints_2);
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
  Mat1d K(3, 3);
  K << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
  vector<Point2f> pts1, pts2;
  for (auto& dm : matches) {
    pts1.emplace_back(keypoints_1[dm.queryIdx].pt);
    pts2.emplace_back(keypoints_2[dm.trainIdx].pt);
  }
  // compute F
  Mat F = cv::findFundamentalMat(pts1, pts2, CV_FM_8POINT);
  cout << "fundamental_matrix is " << endl << F << endl;
  // compute E
  Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
  double focal_length = 521;              //相机焦距, TUM dataset标定值
  Mat E = cv::findEssentialMat(pts1, pts2, focal_length, principal_point);
  cout << "essential_matrix is " << endl << E << endl;
  // compute H
  Mat H = cv::findHomography(pts1, pts2, RANSAC);
  cout << "homography_matrix is " << endl << H << endl;
  // 从 E 中得到旋转和平移
  cv::recoverPose(E, pts1, pts2, R, t, focal_length, principal_point);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;
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
  T1 << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0;
  T2 << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0);
  Mat1d K(3, 3); K << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
  vector<Point2f> pts1, pts2; // 存归一化坐标
  for (DMatch m : matches) {
    auto cam1_pt = pixel2cam(keypoint_1[m.queryIdx].pt, K),
         cam2_pt = pixel2cam(keypoint_2[m.trainIdx].pt, K);
    pts1.emplace_back((float)cam1_pt[0][0], (float)cam1_pt[1][0]);
    pts2.emplace_back((float)cam2_pt[0][0], (float)cam2_pt[1][0]);
  }

  // 4XN 矩阵，存 3D 几次坐标
  Mat pts;
  cv::triangulatePoints(T1, T2, pts1, pts2, pts); // 输入矩阵必须是浮点类型
  // 转换成非齐次坐标
  for (int i = 0; i < pts.cols; i++) {
    Mat x = pts.col(i);
    x /= x.at<float>(3, 0);  // 归一化
    Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}


/****************************************************
 *                    M A I N
 * **************************************************/
int main(int argc, char** argv) {
  //-- 读取图像
  Mat img_1 = imread(R"(E:\git_repos\slambook\ch7\1.png)", CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(R"(E:\git_repos\slambook\ch7\2.png)", CV_LOAD_IMAGE_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
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
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  for (DMatch m : matches) {
    Mat1d y1 = pixel2cam(keypoints_1[m.queryIdx].pt, K),
        y2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    Mat d = y2.t() * t_x * R * y1;
    cout << "epipolar constraint = " << d << endl;
  }

  //-- 三角化！
  vector<Point3d> points;
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);

  //-- 验证三角化点和特征点的重投影关系！
  for (int i = 0; i < matches.size(); i++) {
    auto pt1_cam_ = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
    Point2d pt1_cam(pt1_cam_[0][0], pt1_cam_[1][0]);
    Point2d pt1_cam_reproj(points[i].x / points[i].z, points[i].y / points[i].z);

    cout << "point in the first camera frame: " << pt1_cam << endl;
    cout << "point projected from 3D " << pt1_cam_reproj << ", d=" << points[i].z
         << endl;

    // 第二个图
    auto pt2_cam_ = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
    Point2f pt2_cam(pt2_cam_[0][0], pt2_cam_[1][0]);
    Mat pt2_trans =
        R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    pt2_trans /= pt2_trans.at<double>(2, 0);
    cout << "point in the second camera frame: " << pt2_cam << endl;
    cout << "point reprojected from second frame: " << pt2_trans.t() << endl;
    cout << endl;
  }

  return 0;
}