#include "../include/frame.h"

namespace myslam {
Frame::Frame()
    : id_(-1), time_stamp_(-1), camera_(nullptr), is_key_frame_(false) {}

Frame::Frame(long id, double time_stamp, SE3 T_c_w, Camera::Ptr camera,
             Mat color, Mat depth)
    : id_(id),
      time_stamp_(time_stamp),
      T_c_w_(T_c_w),
      camera_(camera),
      color_(color),
      depth_(depth),
      is_key_frame_(false) {}

Frame::~Frame() {}

Frame::Frame(const Frame& o) {
  camera_ = o.camera_;
  color_ = o.color_.clone();
  depth_ = o.depth_.clone();
  dynamic_map_points = o.dynamic_map_points;
  //dynamic_obj_masks = o.dynamic_obj_masks;
  dynamic_obj_masks.resize(o.dynamic_obj_masks.size());
  for (int i = 0; i < o.dynamic_obj_masks.size(); ++i) {
    dynamic_obj_masks[i].first = o.dynamic_obj_masks[i].first;
    dynamic_obj_masks[i].second = o.dynamic_obj_masks[i].second.clone();
  }
  id_ = o.id_;
  is_key_frame_ = o.is_key_frame_;
  static_background_mask = o.static_background_mask.clone();
  time_stamp_ = o.time_stamp_;
  T_c_w_ = o.T_c_w_;
}

Frame& Frame::operator=(const Frame& o) {
  Frame copy(o);
  std::swap(*this, copy);
  return *this;
}

Frame::Ptr Frame::createFrame() {
  static long factory_id = 0;
  return Frame::Ptr(new Frame(factory_id++));
}

double Frame::findDepth(const cv::KeyPoint& kp) {
  if (depth_.empty()) {
    return -1.0;
  }
  int x = cvRound(kp.pt.x);
  int y = cvRound(kp.pt.y);
  ushort d = depth_.ptr<ushort>(y)[x];
  if (d != 0) {
    return double(d) / camera_->depth_scale_;
  } else {
    // check the nearby points
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, -1, 0, 1};
    for (int i = 0; i < 4; i++) {
      d = depth_.ptr<ushort>(y + dy[i])[x + dx[i]];
      if (d != 0) {
        return double(d) / camera_->depth_scale_;
      }
    }
  }
  return -1.0;
}

void Frame::setPose(const SE3& T_c_w) { T_c_w_ = T_c_w; }

Vector3d Frame::getCamCenter() const { return T_c_w_.inverse().translation(); }

bool Frame::isInFrame(const Vector3d& pt_world) {
  Vector3d p_cam = camera_->world2camera(pt_world, T_c_w_);
  // cout<<"P_cam = "<<p_cam.transpose()<<endl;
  if (p_cam(2, 0) < 0) return false;
  Vector2d pixel = camera_->world2pixel(pt_world, T_c_w_);
  // cout<<"P_pixel = "<<pixel.transpose()<<endl<<endl;
  return pixel(0, 0) > 0 && pixel(1, 0) > 0 && pixel(0, 0) < color_.cols &&
         pixel(1, 0) < color_.rows;
}

void Frame::setStaticBGMask(Mat mask) { static_background_mask = mask; }
void Frame::addDynamicObjMask(unsigned long id, Mat mask) {
  dynamic_obj_masks.emplace_back(id, mask);
}


}  // namespace myslam
