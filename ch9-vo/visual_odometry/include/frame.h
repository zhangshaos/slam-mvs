#ifndef FRAME_H
#define FRAME_H

#include "camera.h"
#include "common_include.h"
#include "mappoint.h"

namespace myslam {

// forward declare
class MapPoint;
class Frame {
 public:
  typedef std::shared_ptr<Frame> Ptr;
  unsigned long id_;    // id of this frame
  double time_stamp_;   // when it is recorded
  SE3 T_c_w_;           // transform from world to camera
  Camera::Ptr camera_;  // Pinhole RGBD Camera model
  Mat color_, depth_;   // color and depth image
  // std::vector<cv::KeyPoint>      keypoints_;  // key points in image
  // std::vector<MapPoint*>         map_points_; // associated map points
  bool is_key_frame_;  // whether a key-frame

  // 使用下列的 mask 来支持 dynamic SLAM
  Mat static_background_mask;
  std::vector<std::pair<unsigned long, cv::Mat>>
      dynamic_obj_masks;  // Obj-ID, Mask!
  std::unordered_map<unsigned long, std::vector<Eigen::Vector3f>>
      dynamic_map_points;  // Obj-ID, MapPoints

 public:  // data members
  Frame();
  Frame(long id, double time_stamp = 0, SE3 T_c_w = SE3(),
        Camera::Ptr camera = nullptr, Mat color = Mat(), Mat depth = Mat());
  ~Frame();
  Frame(const Frame& o);
  Frame& operator=(const Frame& o);
  Frame(Frame&& o) = default;
  Frame& operator=(Frame&& o) = default;

  static Frame::Ptr createFrame();

  // find the depth in depth map
  double findDepth(const cv::KeyPoint& kp);

  // Get Camera Center
  Vector3d getCamCenter() const;

  void setPose(const SE3& T_c_w);

  // check if a point is in this frame
  bool isInFrame(const Vector3d& pt_world);

  // 设置动态物体
  void setStaticBGMask(Mat mask);
  void addDynamicObjMask(unsigned long id, Mat mask);
};

}  // namespace myslam

#endif  // FRAME_H
