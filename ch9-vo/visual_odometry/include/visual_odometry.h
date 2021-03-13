#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <opencv2/features2d/features2d.hpp>

#include "common_include.h"
#include "map.h"

namespace myslam {
class VisualOdometry {
 public:
  typedef std::shared_ptr<VisualOdometry> Ptr;
  enum VOState { BEFORE_INITIALIZING = -2, INITIALIZING = -1, OK = 0, LOST };

  VOState state_;  // current VO status
  Map::Ptr map_;   // map with all frames and map points

  Frame::Ptr ref_;   // reference key-frame
  Frame::Ptr curr_;  // current frame

  cv::Ptr<cv::ORB> orb_;  // orb detector and computer
  // 下面的是背景的特征点和描述符
  std::vector<cv::KeyPoint> keypoints_ref_,
      keypoints_curr_;                      // keypoints in current frame
  Mat descriptors_ref_, descriptors_curr_;  // descriptor in current frame
  // 下面的是动态物体的特征点和描述符
  std::unordered_map<unsigned long, std::vector<cv::KeyPoint>> dy_kpts_ref_,
      dy_kpts_cur_;  // ObjID, keypoints
  std::unordered_map<unsigned long, Mat> dy_desp_ref_,
      dy_desp_cur_;  // ObjId, Descriptor

  cv::FlannBasedMatcher matcher_flann_;     // flann matcher
  std::vector<MapPoint::Ptr> match_3dpts_;  // matched 3d points
  std::vector<int> match_2dkp_index_;  // matched 2d pixels (index of kp_curr)

  SE3 T_c_w_estimated_;  // the estimated pose of current frame
  int num_inliers_;      // number of inlier features in icp
  int num_lost_;         // number of lost times

  // parameters
  int num_of_features_;           // number of features
  double scale_factor_;           // scale in image pyramid
  int level_pyramid_;             // number of pyramid levels
  float match_ratio_;             // ratio for selecting  good matches
  int max_num_lost_;              // max number of continuous lost times
  int min_inliers_;               // minimum inliers
  double key_frame_min_rot;       // minimal rotation of two key-frames
  double key_frame_min_trans;     // minimal translation of two key-frames
  double map_point_erase_ratio_;  // remove map point ratio

 public:  // functions
  VisualOdometry();
  ~VisualOdometry();

  bool addFrame(Frame::Ptr frame);  // add a new frame

  // <用处>：通过 frame 来获取动态物体的数量，然后 resize
  // 所有动态物体的特征点和描述符
  void run(cv::Mat img);

 protected:
  // inner operation
  // 下面的操作一般针对静态背景的初始化（只有 2D-2D 估计位姿时）
  void VisualOdometry::find_feature_matches(
      cv::InputArray& mask1, cv::InputArray& mask2,
      std::vector<cv::KeyPoint>& keypoints_1,
      std::vector<cv::KeyPoint>& keypoints_2, cv::Mat& desp1, cv::Mat& desp2,
      std::vector<cv::DMatch>& matches);
  void pose_estimation_2d2d(std::vector<cv::KeyPoint>& keypoints_1,
                            std::vector<cv::KeyPoint>& keypoints_2,
                            std::vector<cv::DMatch>& matches, Mat& R, Mat& t);
  void triangulation(const std::vector<cv::KeyPoint>& keypoint_1,
                     const std::vector<cv::KeyPoint>& keypoint_2,
                     const std::vector<cv::DMatch>& matches, const Mat& R,
                     const Mat& t, std::vector<cv::Point3d>& points);
  void compute_kp_match(cv::Mat& desp1, cv::Mat& desp2,
                        std::vector<cv::DMatch>& matches);
  bool initialize();

  // 下面的操作一般针对动态物体（采用 2D-2D 估计）
  void compute_dynamic_feature();
  void compute_dynamic_mappoints();

  // 下面采用 3D-2D 估计背景位姿
  void extractKeyPoints();
  void computeDescriptors();
  void featureMatching();
  void poseEstimationPnP();
  void optimizeMap();

  void addKeyFrame();
  void addMapPoints(std::vector<cv::DMatch>& matches,
                    std::vector<cv::Point3d>& points);
  bool checkEstimatedPose();
  bool checkKeyFrame();

  double getViewAngle(Frame::Ptr frame, MapPoint::Ptr point);
  // 让 *_ref_ = *_cur_
  void passed() {
    ref_ = curr_;
    keypoints_ref_ = keypoints_curr_;
    descriptors_ref_ = descriptors_curr_;
    dy_kpts_ref_ = dy_kpts_cur_;
    dy_desp_ref_ = dy_desp_cur_;
  }
};
}  // namespace myslam

#endif  // VISUALODOMETRY_H
