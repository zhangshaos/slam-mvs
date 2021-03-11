#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <iostream>
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

// 像素坐标 => 归一化坐标
Mat1d pixel2cam(const Point2d& p, const Mat& K) {
  Mat1d mat(3, 1);
  mat << (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1), 1;
  return mat;
}

// BA adjustment 使用 g2o 目前还不会！！！略过！
void bundleAdjustment(const vector<Point3f> points_3d,
                      const vector<Point2f> points_2d, const Mat& K, Mat& R,
                      Mat& t) {
  //-- 初始化g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>
      Block;  // pose 维度为 6, landmark 维度为 3
  Block::LinearSolverType* linearSolver =
      new g2o::LinearSolverCSparse<Block::PoseMatrixType>();  // 线性方程求解器
  Block* solver_ptr = new Block(
      std::unique_ptr<Block::LinearSolverType>(linearSolver));  // 矩阵块求解器
  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(
          std::unique_ptr<Block>(solver_ptr));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  // vertex
  g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();  // camera pose
  Eigen::Matrix3d R_mat;
  R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
      R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat(
      R_mat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0),
                             t.at<double>(2, 0))));
  optimizer.addVertex(pose);

  int index = 1;
  for (const Point3f p : points_3d)  // landmarks
  {
    g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
    point->setId(index++);
    point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
    point->setMarginalized(true);  // g2o 中必须设置 marg 参见第十讲内容
    optimizer.addVertex(point);
  }

  // parameter: camera intrinsics
  g2o::CameraParameters* camera = new g2o::CameraParameters(
      K.at<double>(0, 0),
      Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
  camera->setId(0);
  optimizer.addParameter(camera);

  // edges
  index = 1;
  for (const Point2f p : points_2d) {
    g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    edge->setId(index);
    edge->setVertex(
        0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
    edge->setVertex(1, pose);
    edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
    edge->setParameterId(0, 0);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(100);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds."
       << endl;

  cout << endl << "after optimization:" << endl;
  cout << "T=" << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}

/****************************************************
 *                    M A I N
 * **************************************************/
int main(int argc, char** argv) {
  //-- 读取图像
  Mat img_1 = imread(R"(E:\git_repos\slambook\ch7\1.png)");
  Mat img_2 = imread(R"(E:\git_repos\slambook\ch7\2.png)");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  //-- 创建 3D 点
  Mat d1 = imread(
      R"(E:\git_repos\slambook\ch7\1_depth.png)");  // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (DMatch& m : matches) {
    uint16_t d = d1.ptr<unsigned short>(
        int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)  // bad depth
      continue;
    float dd = d / 5000.0;
    auto p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.emplace_back(float(dd * p1[0][0]), float(dd * p1[1][0]),
                        float(dd * p1[2][0]));
    pts_2d.emplace_back(keypoints_2[m.trainIdx].pt);
  }
  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  //-- PnP 求解 R t
  Mat rot_vec, t, R;
  cv::solvePnP(pts_3d, pts_2d, K, Mat(), rot_vec, t);
  cv::Rodrigues(rot_vec, R);
  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;

  cout << "calling bundle adjustment" << endl;
  bundleAdjustment(pts_3d, pts_2d, K, R, t);
  return 0;
}