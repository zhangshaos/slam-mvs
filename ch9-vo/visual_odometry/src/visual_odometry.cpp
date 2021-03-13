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
  // �ų�һЩ����Ҫ��ƥ��
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

// �Ծ�̬�������ӳ��� 3D ��
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
  // �� E �еõ���ת��ƽ��
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
  vector<Point2f> pts1, pts2;  // ���һ������
  for (DMatch m : matches) {
    Eigen::Vector2d kp1(keypoint_1[m.queryIdx].pt.x,
                        keypoint_1[m.queryIdx].pt.y),
        kp2(keypoint_2[m.trainIdx].pt.x, keypoint_2[m.trainIdx].pt.y);
    auto cam1_pt = ref_->camera_->pixel2camera(kp1),
         cam2_pt = curr_->camera_->pixel2camera(kp2);
    pts1.emplace_back((float)cam1_pt.x(), (float)cam1_pt.y());
    pts2.emplace_back((float)cam2_pt.x(), (float)cam2_pt.y());
  }

  // 4XN ���󣬴� 3D ��������
  boost::timer timer;
  Mat pts;
  cv::triangulatePoints(T1, T2, pts1, pts2, pts);  // �����������Ǹ�������
  cout << "triangulate points cost time: " << timer.elapsed() << endl;
  // ת���ɷ��������
  for (int i = 0; i < pts.cols; i++) {
    Mat x = pts.col(i);
    x /= x.at<float>(3, 0);  // ��һ��
    Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    points.push_back(p);
  }
}

// ʹ�� ref_ �� curr_ ��֡��̬���������ǻ��ͳ�ʼ�����ƣ�
// TODO: �Գ�ʼ���ǻ��ĵ��ƾ��� BA ���Ż�
bool VisualOdometry::initialize() {
  using namespace std;
  boost::timer timer;
  // ���㱳����λ�˱仯�����������ǻ�
  std::vector<cv::DMatch> matches;
  find_feature_matches(ref_->static_background_mask,
                       curr_->static_background_mask, keypoints_ref_,
                       keypoints_curr_, descriptors_ref_, descriptors_curr_,
                       matches);
  Mat R, t;
  pose_estimation_2d2d(keypoints_ref_, keypoints_curr_, matches, R, t);
  // ���� T_c_w_estimated_
  Eigen::Matrix3d rot;
  Eigen::Vector3d trans;
  cv::cv2eigen(R, rot);
  cv::cv2eigen(t, trans);
  T_c_w_estimated_ = SE3(rot, trans);
  std::vector<cv::Point3d> points;  // ��һ֡���������ϵ
  triangulation(keypoints_ref_, keypoints_curr_, matches, R, t, points);
  // �������Ʒ����ͼ��
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
      // ��ȡ�����㣬����������
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
      // �˽׶κ���һ�׶ζ������� dynamic object��
      // ʹ����֡��ʼ����
      if (initialize()) {
        curr_->T_c_w_ = T_c_w_estimated_;
        addKeyFrame();  // the first frame is a key-frame
        state_ = OK;
        // Ϊ�� OK ʱ�Զ�̬�������ǻ�
        // �� frame �ж�ȡ��̬��������� mask��Ȼ������������������
      }
      compute_dynamic_feature();
      passed();
      break;
    }
    case OK: {
      curr_ = frame;
      extractKeyPoints();    // ��ȡ����������
      computeDescriptors();  // ���㱳��������������
      featureMatching();  // ���㱳���� 3D(ref)-2D(cur) ���ƥ���ϵ
      // TODO: ���� [��̬����] �������������
      // TODO: ���ǻ� [��̬����]�������ɵ�ǰ�������ϵ�µ��������
      compute_dynamic_mappoints();
      poseEstimationPnP();               // ���㵱ǰ֡�� Tcw
      if (checkEstimatedPose() == true)  // a good estimation
      {
        curr_->T_c_w_ = T_c_w_estimated_;
        // TODO: �������ǻ�������ͼ��
        //  ���������Ѿ��õ� curr_->T_c_w_ �ˣ�ֻ��Ҫ�õ� curr_ �µ� 3D �����
        //  ������ => ��һ�����꣬��ȣ�
        //  ����ˡ�ֱ�����ǻ���
        // <���˼·>�������������ƥ����� + Tcr��R,t��=> map point in ref_ =>
        // ��� ref_->T_w_c_ ����
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
        optimizeMap();  // �Ż�������
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
  // ͨ���˺���������� 3D(ref)-2D(cur) ���ƥ���ϵ
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
  // ����ƥ��
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
  vector<cv::Point3f> pts3d;  // ��������ϵ�µ� 3D ��
  vector<cv::Point2f> pts2d;  // curr_ ����µ���������

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
  // ���� 3D ��������������ϵ�µģ�ͨ����Ӧ��һ֡������������ڵ�һ֡��λ�˱仯
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

// ��Ҫ�������Ϣ��
// void VisualOdometry::addMapPoints() {
//  using namespace std;
//  // add the new map points into map
//  vector<bool> matched(keypoints_curr_.size(), false);
//  for (int index : match_2dkp_index_) matched[index] = true;
//  for (int i = 0; i < keypoints_curr_.size(); i++) {
//    if (matched[i] == true) continue;
//    // û��ƥ�䵽�����ص㣬
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
    // ɾ�����ڵ�ǰ֡�ĵ㣡
    if (!curr_->isInFrame(iter->second->pos_)) {
      iter = map_->map_points_.erase(iter);
      continue;
    }
    // ɾ������ο���������ȱ�޷�ƥ��ĵ�
    float match_ratio =
        float(iter->second->matched_times_) / iter->second->visible_times_;
    if (match_ratio < map_point_erase_ratio_) {
      iter = map_->map_points_.erase(iter);
      continue;
    }
    // �ӵ�ǰ�㱻��һ�ο���������ǰ֡���ӽ��Ѿ�ת���� 30 �ȣ�ɾ�������ĵ�
    double angle = getViewAngle(curr_, iter->second);
    if (angle > M_PI / 6.) {
      iter = map_->map_points_.erase(iter);
      continue;
    }
    if (iter->second->good_ == false) {
      // TODO try triangulate this map point
      // Ŀǰ���еĵ㶼�� good �ĵ㣡
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

// �� frame �ж�ȡ��̬��������� mask��Ȼ������������������
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
  // <���˼·>��
  //  1.�� frame �ж�ȡ��̬��������� mask��Ȼ������������������
  //  2.λ�˹��ƣ�������ϱ任 R* �� t*
  //  3.���ǻ���õ�һ֡�µ�3D���� => R*,t* => �ڶ�֡�µ�3D����
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
