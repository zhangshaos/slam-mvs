#ifndef MAP_H
#define MAP_H

#include "common_include.h"
#include "frame.h"
#include "mappoint.h"

namespace myslam {
class Map {
 public:
  typedef std::shared_ptr<Map> Ptr;
  std::unordered_map<unsigned long, MapPoint::Ptr> map_points_;  // all landmarks
  std::unordered_map<unsigned long, Frame::Ptr> keyframes_;      // all key-frames

  Map() {}

  void insertKeyFrame(Frame::Ptr frame);
  void insertMapPoint(MapPoint::Ptr map_point);
};
}  // namespace myslam

#endif  // MAP_H
