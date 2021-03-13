#define _USE_MATH_DEFINES
#include "../include/visual_odometry.h"

#include <algorithm>
#include <boost/format.hpp>
#include <boost/timer.hpp>
#include <cmath>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/config.h"
#include "../include/g2o_types.h"

namespace myslam {

VisualOdometry::VisualOdometry()
    : state_(INITIALIZING),
      ref_(nullptr),
      curr_(nullptr),
      map_(new Map),
      num_lost_(0),
      num_inliers_(0),
      matcher_flann_(new cv::flann::LshIndexParams(5, 10, 2)) {
  num_of_features_ = Config::get<int>("number_of_features");
  scale_factor_ = Config::get<double>("scale_factor");
  level_pyramid_ = Config::get<int>("level_pyramid");
  match_ratio_ = Config::get<float>("match_ratio");
  max_num_lost_ = Config::get<float>("max_num_lost");
  min_inliers_ = Config::get<int>("min_inliers");
  key_frame_min_rot = Config::get<double>("keyframe_rotation");
  key_frame_min_trans = Config::get<double>("keyframe_translation");
  map_point_erase_ratio_ = Config::get<double>("map_point_erase_ratio");
  orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
}

VisualOdometry::~VisualOdometry() {}

//**************************************//
//--------------- M-A-I-N --------------//
//**************************************//
void VisualOdometry::run(cv::Mat img) {
  Frame::Ptr pFrame = myslam::Frame::createFrame();
  pFrame->camera_ = Camera::Ptr(new myslam::Camera);
  pFrame->color_ = img;
  //pFrame->depth_ = depth;
  //pFrame->time_stamp_ = rgb_times[i];
  addFrame(pFrame);
}

void VisualOdometry::compute_kp_match(cv::Mat& desp1, cv::Mat& desp2,
                                      std::vector<cv::DMatch>& matches) {
  using namespace std;
  boost::timer timer;
  matcher_flann_.match(desp1, desp2, matches);
  cout << "compute flann matching cost time: " << timer.elapsed() << endl;
  // 排除一些不必要的匹配
  float min_dist = 1e8f, max_dist = 0.f;
  for (auto& dm : matches) {
    min_dist = min(min_dist, dm.distance);
    max_dist = max(max_dist, dm.distance);
  }
  vector<cv::DMatch> good_match;
  for (auto& dm : matches) {
    if (dm.distance <= max(match_ratio_ * min_dist,
                           1.0f / match_ratio_ * (min_dist + max_dist)))
      good_match.push_back(dm);
  }
  swap(good_match, matches);
}

// 对静态物体增加场景 3D 点
void VisualOdometry::addMapPoints(std::vector<cv::DMatch>& matches,
                                  std::vector<cv::Point3d>& points) {
  assert(matches.size() == points.size());
  for (int i = 0; i < points.size(); ++i) {
    Vector3d p_world(points[i].x, points[i].y, points[i].z);
    p_world = ref_->camera_->camera2world(p_world, ref_->T_c_w_);
    Vector3d n = p_world - ref_->getCamCenter();
    n.normalize();
    int j = matches[i].trainIdx;  // j == curr_ 's descriptor idx
    MapPoint::Ptr map_point = MapPoint::createMapPoint(
        p_world, n, descriptors_curr_.row(j).clone(), curr_.get());
    map_->insertMapPoint(map_point);
  }
}

void VisualOdometry::find_feature_matches(
    cv::InputArray& mask1, cv::InputArray& mask2,
    std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2, cv::Mat& desp1, cv::Mat& desp2,
    std::vector<cv::DMatch>& matches) {
  using namespace std;
  using namespace cv;
  boost::timer timer1;
  orb_->detect(ref_->color_, keypoints_1, mask1);
  orb_->detect(curr_->color_, keypoints_2, mask2);
  cout << "extract keypoints cost time: " << timer1.elapsed() << endl;

  boost::timer timer2;
  orb_->compute(ref_->color_, keypoints_1, desp1);
  orb_->compute(curr_->color_, keypoints_2, desp2);
  cout << "descriptor computation cost time: " << timer2.elapsed() << endl;

  compute_kp_match(desp1, desp2, matches);
}

void VisualOdometry::pose_estimation_2d2d(
    std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches,
    Mat& R, Mat& t) {
  using namespace std;
  using namespace cv;
  Mat K = (cv::Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_,
           0, ref_->camera_->fy_, ref_->camera_->cy_, 0, 0, 1);
  vector<Point2f> pts1, pts2;
  for (auto& dm : matches) {
    pts1.emplace_back(keypoints_1[dm.queryIdx].pt);
    pts2.emplace_back(keypoints_2[dm.trainIdx].pt);
  }
  // compute E
  boost::timer timer;
  Mat E = cv::findEssentialMat(pts1, pts2, K);
  // 从 E 中得到旋转和平移
  cv::recoverPose(E, pts1, pts2, K, R, t);
  cout << "compute E and recover Pose cost time: " << timer.elapsed() << endl;
  std::cout << "R is " << endl << R << endl;
  std::cout << "t is " << endl << t << endl;
}

void VisualOdometry::triangulation(const std::vector<cv::KeyPoint>& keypoint_1,
                                   const std::vector<cv::KeyPoint>& keypoint_2,
                                   const std::vector<cv::DMatch>& matches,
                                   const Mat& R, const Mat& t,
                                   std::vector<cv::Point3d>& points) {
  using namespace std;
  using namespace cv;
  Mat1f T1(3, 4), T2(3, 4);
  T1 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
  T2 << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      t.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(1, 1),
      R.at<double>(1, 2), t.at<double>(1, 0), R.at<double>(2, 0),
      R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0);
  Mat K = (cv::Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_,
           0, ref_->camera_->fy_, ref_->camera_->cy_, 0, 0, 1);
  vector<Point2f> pts1, pts2;  // 存归一化坐标
  for (DMatch m : matches) {
    Eigen::Vector2d kp1(keypoint_1[m.queryIdx].pt.x,
                        keypoint_1[m.queryIdx].pt.y),
        kp2(keypoint_2[m.trainIdx].pt.x, keypoint_2[m.trainIdx].pt.y);
    auto cam1_pt = ref_->camera_->pixel2camera(kp1),
         cam2_pt = curr_->camera_->pixel2camera(kp2);
    pts1.emplace_back((float)cam1_pt.x(), (float)cam1_pt.y());
    pts2.emplace_back((float)cam2_pt.x(), (float)cam2_pt.y());
  }

  // 4XN 矩阵，存 3D 几次坐标
  boost::timer timer;
  Mat pts;
  cv::triangulatePoints(T1, T2, pts1, pts2, pts);  // 输入矩阵必须是浮点类型
  cout << "triangulate points cost time: " << timer.elapsed() << endl;
  // 转换成非齐次坐标
  for (int i = 0; i < pts.cols; i++) {
    Mat x = pts.col(i);
    x /= x.at<float>(3, 0);  // 归一化
    Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}

// 使用 ref_ 和 curr_ 两帧静态物体来三角化和初始化点云！
// TODO: 对初始三角化的点云就行 BA 来优化
bool VisualOdometry::initialize() {
  using namespace std;
  boost::timer timer;
  // 计算背景的位姿变化，并进行三角化
  std::vector<cv::DMatch> matches;
  find_feature_matches(ref_->static_background_mask,
                       curr_->static_background_mask, keypoints_ref_,
                       keypoints_curr_, descriptors_ref_, descriptors_curr_,
                       matches);
  Mat R, t;
  pose_estimation_2d2d(keypoints_ref_, keypoints_curr_, matches, R, t);
  // 设置 T_c_w_estimated_
  Eigen::Matrix3d rot;
  Eigen::Vector3d trans;
  cv::cv2eigen(R, rot);
  cv::cv2eigen(t, trans);
  T_c_w_estimated_ = SE3(rot, trans);
  std::vector<cv::Point3d> points;  // 第一帧的相机坐标系
  triangulation(keypoints_ref_, keypoints_curr_, matches, R, t, points);
  // 背景点云放入地图中
  addMapPoints(matches, points);
  cout << "VO initializing const time: " << timer.elapsed() << endl;
  return true;
}

// MAIN FUNCTION doing all......
bool VisualOdometry::addFrame(Frame::Ptr frame) {
  using namespace std;
  switch (state_) {
    case BEFORE_INITIALIZING: {
      curr_ = frame;
      // 提取体征点，计算描述符
      extractKeyPoints();
      computeDescriptors();
      curr_->T_c_w_ = SE3(Eigen::Matrix4d::Identity());
      compute_dynamic_feature();
      passed();
      state_ = INITIALIZING;
      break;
    }
    case INITIALIZING: {
      curr_ = frame;
      // extract features from first frame and add them into map
      // 此阶段和上一阶段都不处理 dynamic object！
      // 使用两帧初始化！
      if (initialize()) {
        curr_->T_c_w_ = T_c_w_estimated_;
        addKeyFrame();  // the first frame is a key-frame
        state_ = OK;
        // 为了 OK 时对动态物体三角化
        // 从 frame 中读取动态物体的所有 mask，然后计算特征点和描述符
      }
      compute_dynamic_feature();
      passed();
      break;
    }
    case OK: {
      curr_ = frame;
      extractKeyPoints();    // 提取背景特征点
      computeDescriptors();  // 计算背景特征点描述符
      featureMatching();  // 计算背景的 3D(ref)-2D(cur) 点的匹配关系
      // TODO: 计算 [动态物体] 特征点和描述符
      // TODO: 三角化 [动态物体]，并生成当前相机坐标系下的物体点云
      compute_dynamic_mappoints();
      poseEstimationPnP();               // 计算当前帧的 Tcw
      if (checkEstimatedPose() == true)  // a good estimation
      {
        curr_->T_c_w_ = T_c_w_estimated_;
        // TODO: 背景三角化和填充地图点
        //  由于我们已经得到 curr_->T_c_w_ 了，只需要得到 curr_ 下的 3D 点就行
        //  将像素 => 归一化坐标，深度？
        //  额，算了。直接三角化吧
        // <求解思路>：计算出特征点匹配情况 + Tcr（R,t）=> map point in ref_ =>
        // 左乘 ref_->T_w_c_ 即可
        {
          std::vector<cv::DMatch> matches;
          compute_kp_match(descriptors_ref_, descriptors_curr_, matches);
          auto ref_T_w_c = ref_->T_c_w_.inverse();
          auto T_c_r = curr_->T_c_w_ * ref_T_w_c;
          cv::Mat R, t;
          cv::eigen2cv(T_c_r.rotationMatrix(), R);
          cv::eigen2cv(T_c_r.translation(), t);
          std::vector<cv::Point3d> points;
          triangulation(keypoints_ref_, keypoints_curr_, matches, R, t, points);
          addMapPoints(matches, points);
        }
        optimizeMap();  // 优化背景点
        num_lost_ = 0;
        if (checkKeyFrame() == true)  // is a key-frame
        {
          addKeyFrame();
        }
        passed();
      } else  // bad estimation due to various reasons
      {
        num_lost_++;
        if (num_lost_ > max_num_lost_) {
          state_ = LOST;
        }
        return false;
      }
      break;
    }
    case LOST: {
      cout << "vo has lost." << endl;
      break;
    }
  }

  return true;
}

void VisualOdometry::extractKeyPoints() {
  using namespace std;
  boost::timer timer;
  orb_->detect(curr_->color_, keypoints_curr_, curr_->static_background_mask);
  cout << "extract keypoints cost time: " << timer.elapsed() << endl;
}

void VisualOdometry::computeDescriptors() {
  using namespace std;
  boost::timer timer;
  orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
  cout << "descriptor computation cost time: " << timer.elapsed() << endl;
}

void VisualOdometry::featureMatching() {
  using namespace std;
  // 通过此函数，计算出 3D(ref)-2D(cur) 点的匹配关系
  boost::timer timer;
  // select the candidates in map
  Mat desp_map;
  vector<MapPoint::Ptr> candidate;
  for (auto& allpoints : map_->map_points_) {
    MapPoint::Ptr& p = allpoints.second;
    // check if p in curr frame image
    if (curr_->isInFrame(p->pos_)) {
      // add to candidate
      p->visible_times_++;
      candidate.push_back(p);
      desp_map.push_back(p->descriptor_);
    }
  }
  // 初步匹配
  vector<cv::DMatch> matches;
  matcher_flann_.match(desp_map, descriptors_curr_, matches);
  // select the best matches
  float min_dist = 1e8f, max_dist = 0.f;
  for (auto& dm : matches) {
    min_dist = min(min_dist, dm.distance);
    max_dist = max(max_dist, dm.distance);
  }

  match_3dpts_.clear();
  match_2dkp_index_.clear();
  for (cv::DMatch& m : matches) {
    if (m.distance < max(match_ratio_ * min_dist,
                         1.0f / match_ratio_ * (min_dist + max_dist))) {
      match_3dpts_.push_back(candidate[m.queryIdx]);
      match_2dkp_index_.push_back(m.trainIdx);
    }
  }
  cout << "good matches: " << match_3dpts_.size() << endl;
  cout << "match cost time: " << timer.elapsed() << endl;
}

void VisualOdometry::poseEstimationPnP() {
  using namespace std;
  // construct the 3d 2d observations
  vector<cv::Point3f> pts3d;  // 世界坐标系下的 3D 点
  vector<cv::Point2f> pts2d;  // curr_ 相机下的像素坐标

  for (int index : match_2dkp_index_) {
    pts2d.push_back(keypoints_curr_[index].pt);
  }
  for (MapPoint::Ptr pt : match_3dpts_) {
    pts3d.push_back(pt->getPositionCV());
  }

  Mat K = (cv::Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_,
           0, ref_->camera_->fy_, ref_->camera_->cy_, 0, 0, 1);
  Mat rvec, tvec, inliers;
  cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99,
                     inliers);
  num_inliers_ = inliers.rows;
  cout << "pnp inliers: " << num_inliers_ << endl;
  Mat rot_mat;
  cv::Rodrigues(rvec, rot_mat);
  Eigen::Matrix3d m;
  cv::cv2eigen(rot_mat, m);
  Eigen::Vector3d v;
  cv::cv2eigen(tvec, v);
  // 由于 3D 点是在世界坐标系下的（通常对应第一帧），所以相对于第一帧的位姿变化
  // == Tcw!
  T_c_w_estimated_ = SE3(m, v);

  // using bundle adjustment to optimize the pose
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
  Block::LinearSolverType* linearSolver =
      new g2o::LinearSolverDense<Block::PoseMatrixType>();
  Block* solver_ptr =
      new Block(unique_ptr<Block::LinearSolverType>(linearSolver));
  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat(T_c_w_estimated_.rotationMatrix(),
                                 T_c_w_estimated_.translation()));
  optimizer.addVertex(pose);

  // edges
  for (int i = 0; i < inliers.rows; i++) {
    int index = inliers.at<int>(i, 0);
    // 3D -> 2D projection
    EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
    edge->setId(i);
    edge->setVertex(0, pose);
    edge->camera_ = curr_->camera_.get();
    edge->point_ = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
    edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y));
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    // set the inlier map points
    match_3dpts_[index]->matched_times_++;
  }

  optimizer.initializeOptimization();
  optimizer.optimize(10);

  T_c_w_estimated_ =
      SE3(pose->estimate().rotation(), pose->estimate().translation());

  cout << "T_c_w_estimated_: " << endl << T_c_w_estimated_.matrix() << endl;
}

bool VisualOdometry::checkEstimatedPose() {
  using namespace std;
  // check if the estimated pose is good
  if (num_inliers_ < min_inliers_) {
    cout << "reject because inlier is too small: " << num_inliers_ << endl;
    return false;
  }
  // if the motion is too large, it is probably wrong
  SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
  Sophus::Vector6d d = T_r_c.log();
  if (d.norm() > 5.0) {
    cout << "reject because motion is too large: " << d.norm() << endl;
    return false;
  }
  return true;
}

bool VisualOdometry::checkKeyFrame() {
  SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
  Sophus::Vector6d d = T_r_c.log();
  Vector3d trans = d.head<3>();
  Vector3d rot = d.tail<3>();
  if (rot.norm() > key_frame_min_rot || trans.norm() > key_frame_min_trans)
    return true;
  return false;
}

void VisualOdometry::addKeyFrame() { map_->insertKeyFrame(curr_); }

// 需要由深度信息！
// void VisualOdometry::addMapPoints() {
//  using namespace std;
//  // add the new map points into map
//  vector<bool> matched(keypoints_curr_.size(), false);
//  for (int index : match_2dkp_index_) matched[index] = true;
//  for (int i = 0; i < keypoints_curr_.size(); i++) {
//    if (matched[i] == true) continue;
//    // 没有匹配到的像素点，
//    double d = ref_->findDepth(keypoints_curr_[i]);
//    if (d < 0) continue;
//    Vector3d p_world = ref_->camera_->pixel2world(
//        Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y),
//        curr_->T_c_w_, d);
//    Vector3d n = p_world - ref_->getCamCenter();
//    n.normalize();
//    MapPoint::Ptr map_point = MapPoint::createMapPoint(
//        p_world, n, descriptors_curr_.row(i).clone(), curr_.get());
//    map_->insertMapPoint(map_point);
//  }
//}

void VisualOdometry::optimizeMap() {
  using namespace std;
  // remove the hardly seen and no visible points
  for (auto iter = map_->map_points_.begin();
       iter != map_->map_points_.end();) {
    // 删掉不在当前帧的点！
    if (!curr_->isInFrame(iter->second->pos_)) {
      iter = map_->map_points_.erase(iter);
      continue;
    }
    // 删掉被多次看到，但是缺无法匹配的点
    float match_ratio =
        float(iter->second->matched_times_) / iter->second->visible_times_;
    if (match_ratio < map_point_erase_ratio_) {
      iter = map_->map_points_.erase(iter);
      continue;
    }
    // 从当前点被第一次看见，到当前帧，视角已经转换了 30 度，删掉这样的点
    double angle = getViewAngle(curr_, iter->second);
    if (angle > M_PI / 6.) {
      iter = map_->map_points_.erase(iter);
      continue;
    }
    if (iter->second->good_ == false) {
      // TODO try triangulate this map point
      // 目前所有的点都是 good 的点！
    }
    iter++;
  }

  if (match_2dkp_index_.size() < 100) {
    // addMapPoints();
    cout << "the curr_'s keypoints corresponded by 3D points in map is less"
            "TODO: add some 3D points in map.\n";
  }
  if (map_->map_points_.size() > 1000) {
    // TODO map is too large, remove some one
    map_point_erase_ratio_ += 0.05;
  } else
    map_point_erase_ratio_ = 0.1;
  cout << "map points: " << map_->map_points_.size() << endl;
}

double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point) {
  Vector3d n = point->pos_ - frame->getCamCenter();
  n.normalize();
  return acos(n.transpose() * point->norm_);
}

// 从 frame 中读取动态物体的所有 mask，然后计算特征点和描述符
void VisualOdometry::compute_dynamic_feature() {
  using namespace std;
  using namespace cv;
  boost::timer timer;
  for (auto& [id, mask] : curr_->dynamic_obj_masks) {
    orb_->detect(curr_->color_, dy_kpts_cur_[id], mask);
    orb_->compute(curr_->color_, dy_kpts_cur_[id], dy_desp_cur_[id]);
  }
  cout
      << boost::format(
             "compute %lld dynamic objects' keypoints and descriptor cost time "
             "%lf") %
             curr_->dynamic_obj_masks.size() % timer.elapsed();
}

void VisualOdometry::compute_dynamic_mappoints() {
  using namespace std;
  boost::timer timer;
  // <求解思路>：
  //  1.从 frame 中读取动态物体的所有 mask，然后计算特征点和描述符
  //  2.位姿估计，求出复合变换 R* 和 t*
  //  3.三角化求得第一帧下的3D坐标 => R*,t* => 第二帧下的3D坐标
  compute_dynamic_feature();
  for (auto obj : curr_->dynamic_obj_masks) {
    auto id = obj.first;
    std::vector<cv::DMatch> matches;
    compute_kp_match(dy_desp_ref_[id], dy_desp_cur_[id], matches);
    cv::Mat R, t;
    pose_estimation_2d2d(dy_kpts_ref_[id], dy_kpts_cur_[id], matches, R, t);
    std::vector<cv::Point3d> pts;
    triangulation(dy_kpts_ref_[id], dy_kpts_cur_[id], matches, R, t, pts);
    Eigen::Matrix3d R_s;
    Eigen::Vector3d t_s;
    cv::cv2eigen(R, R_s);
    cv::cv2eigen(t, t_s);
    auto T_s = Eigen::Isometry3d::Identity();
    T_s.rotate(R);
    T_s.pretranslate(t_s);
    for (auto& p : pts) {
      Eigen::Vector3f pt(p.x, p.y, p.z);
      curr_->dynamic_map_points[id].emplace_back(T_s * pt);
    } // get all mappoints
  } // get all objects
  cout << "compute all dynamic objs cost time " << timer.elapsed() << endl;
}  // namespace myslam
