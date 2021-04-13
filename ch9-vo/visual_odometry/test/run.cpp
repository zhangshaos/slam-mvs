// -------------- test the visual odometry -------------
#include <boost/format.hpp>
#include <boost/timer.hpp>
#include <filesystem>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "../include/config.h"
#include "../include/visual_odometry.h"

namespace fs = std::filesystem;

namespace std {
template <>
struct hash<fs::path> {
  std::size_t operator()(fs::path const& s) const noexcept {
    return std::hash<std::string>{}(s.string());
  }
};
}  // namespace std

Eigen::Vector3i GetRandomColor() {
  return Eigen::Vector3i(rand() % 256, rand() % 256, rand() % 256);
}

std::unordered_map<std::string, unsigned long> RetriveMasks(
    const std::string& file) {
  using namespace std;
  std::unordered_map<std::string, unsigned long> ans;
  ifstream f(file);
  int id = 0;
  string line_data;
  while (getline(f, line_data)) {
    stringstream s(line_data);
    string name;
    while (s >> name) {
      ans[name] = id;
    }
    ++id;
  }
  return ans;
}

int Main() {
  srand(time(0));
  using namespace std;
  using namespace cv;
  // 1. 读取配置！
  myslam::Config::setParameterFile(
      R"(E:\git_repos\slam-mvs\ch9-vo\visual_odometry\default.yaml)");
  myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);

  // 2. 准备图像文件和 mask 文件
  auto mask_to_id = ::RetriveMasks(
      R"(E:\git_repos\slam-mvs\ch9-vo\visual_odometry\tracked.txt)");
  vector<fs::path> rgb_files;
  unordered_set<fs::path> mask_files;
  for (
      fs::directory_iterator it(
          R"(E:\datasets\3d_car_instance_sample\3d_car_instance_sample\images\test\small)"),
      end;
      it != end; ++it) {
    if (it->path().extension() == ".jpg") {
      rgb_files.emplace_back(it->path());
    } else if (it->path().extension() == ".tiff" &&
               mask_to_id.count(it->path().filename().string())) {
      mask_files.emplace(it->path());
    }
  }

  sort(rgb_files.begin(), rgb_files.end());
  vector<vector<fs::path>> masks;
  for (auto& rgb : rgb_files) {
    masks.emplace_back();
    for (int i = 0; i < 50; ++i) {
      fs::path mask_path =
          (boost::format("%s_%d_mask.tiff") % rgb.string() % i).str();
      if (mask_files.count(mask_path)) {
        masks[masks.size() - 1].emplace_back(mask_path);
      }
    }
  }

  ofstream points("./out_points.txt"), bg_points("./bg.txt");

  // 3. 开始 VO
  vo->use_dynamic = true;
  Eigen::Vector3f trans(0, 0, 0), step(0, 0, 5);
  for (int i = 0; i < rgb_files.size(); ++i) {
    auto img_name = rgb_files[i].string();
    Mat img = imread(img_name);
    Mat bg = imread((boost::format("%s_bg_mask.tiff") % img_name).str(),
                    IMREAD_GRAYSCALE);
    vector<pair<unsigned long, Mat>> rgb_masks;
    for (auto& m : masks[i]) {
      rgb_masks.emplace_back(mask_to_id[m.filename().string()],
                             imread(m.string(), IMREAD_GRAYSCALE));
    }
    vo->run(img, bg, rgb_masks);
    // 4. 输出结果
    auto id_to_maps = vo->curr_->dynamic_map_points;
    for (auto& [id, maps] : id_to_maps) {
      auto color = GetRandomColor();
      for (auto& pt : maps) {
        auto p = pt + trans;
        trans += step;
        points << boost::format("%f;%f;%f;%d;%d;%d") % p.x() % p.y() % (p.z()) %
                      color.x() % color.y() % color.z()
               << endl;
      }
    }
    //for (auto& p : vo->map_->map_points_) {
    //  auto& pos = p.second->pos_;
    //  bg_points << boost::format("%lf;%lf;%lf;255;255;255") % pos.x() %
    //                   pos.y() % pos.z()
    //            << endl;
    //}
    printf("map points size: %lld\n", vo->map_->map_points_.size());
  }

  return 0;
}

// 仅仅做静态物体的生成
int Main2() {
  srand(time(0));
  using namespace std;
  using namespace cv;
  // 1. 读取配置！
  myslam::Config::setParameterFile(
      R"(E:\git_repos\slam-mvs\ch9-vo\visual_odometry\default.yaml)");
  myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);

  // 2. 准备图像文件和 mask 文件
  auto mask_to_id = ::RetriveMasks(
      R"(E:\git_repos\slam-mvs\ch9-vo\visual_odometry\tracked.txt)");
  vector<fs::path> rgb_files;
  unordered_set<fs::path> mask_files;
  for (
      fs::directory_iterator it(
          R"(E:\datasets\3d_car_instance_sample\3d_car_instance_sample\images\test\small)"),
      end;
      it != end; ++it) {
    if (it->path().extension() == ".jpg") {
      rgb_files.emplace_back(it->path());

    } else if (it->path().extension() == ".tiff" &&
               mask_to_id.count(it->path().filename().string())) {
      mask_files.emplace(it->path());
    }
  }

  sort(rgb_files.begin(), rgb_files.end());
  vector<vector<fs::path>> masks;
  for (auto& rgb : rgb_files) {
    masks.emplace_back();
    for (int i = 0; i < 50; ++i) {
      fs::path mask_path =
          (boost::format("%s_%d_mask.tiff") % rgb.string() % i).str();
      if (mask_files.count(mask_path)) {
        masks[masks.size() - 1].emplace_back(mask_path);
      }
    }
  }

  ofstream points("./out_points.txt"), bg_points("./bg.txt");

  // 3. 开始 VO
  Eigen::Vector3f trans(0, 0, 0), step(0, 0, 5);
  for (int i = 0; i < rgb_files.size(); ++i) {
    auto img_name = rgb_files[i].string();
    Mat img = imread(img_name);
    Mat bg = imread((boost::format("%s_bg_mask.tiff") % img_name).str(),
                    IMREAD_GRAYSCALE);
    vector<pair<unsigned long, Mat>> rgb_masks;
    for (auto& m : masks[i]) {
      rgb_masks.emplace_back(mask_to_id[m.filename().string()],
                             imread(m.string(), IMREAD_GRAYSCALE));
    }
    vo->run(img, bg, rgb_masks);
    // 4. 输出结果
    auto id_to_maps = vo->curr_->dynamic_map_points;
    auto color = GetRandomColor();
    for (auto& [id, maps] : id_to_maps) {
      for (auto& pt : maps) {
        auto p = pt + trans;
        trans += step;
        points << boost::format("%f;%f;%f;%d;%d;%d") % p.x() % p.y() % p.z() %
                      color.x() % color.y() % color.z()
               << endl;
      }
    }
    printf("Map points is %lld\n", vo->map_->map_points_.size());
    //for (auto& p : vo->map_->map_points_) {
    //  auto& pos = p.second->pos_;
    //  bg_points << boost::format("%lf;%lf;%lf;255;255;255") % pos.x() %
    //                   pos.y() % pos.z()
    //            << endl;
    //}
  }

  return 0;
}

// 仅仅测试三帧测试图片！
int Main3() {
  srand(time(0));
  using namespace std;
  using namespace cv;
  // 1. 读取配置！
  myslam::Config::setParameterFile(
      R"(E:\git_repos\slam-mvs\ch9-vo\visual_odometry\default.yaml)");
  myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);

  // 2. 准备图像文件
  vector<fs::path> rgb_files{
      R"(E:\Py_Projects\airsim_usage\scenes\scene_0.png)"s,
      R"(E:\Py_Projects\airsim_usage\scenes\scene_1.png)"s,
      R"(E:\Py_Projects\airsim_usage\scenes\scene_2.png)"s};

  ofstream points("./obj.txt"), bg_points("./bg.txt");

  // 3. 开始 VO
  Eigen::Vector3f trans(0, 0, 0), step(0, 0, 5);
  for (int i = 0; i < rgb_files.size(); ++i) {
    Mat img = imread(rgb_files[i].string());
    Mat1b bg(img.rows, img.cols, 255);
    vector<pair<unsigned long, Mat>> rgb_masks;
    vo->run(img, bg, rgb_masks);
    // 4. 输出结果
    auto id_to_maps = vo->curr_->dynamic_map_points;
    auto color = GetRandomColor();
    for (auto& [id, maps] : id_to_maps) {
      for (auto& pt : maps) {
        auto p = pt + trans;
        trans += step;
        points << boost::format("%f;%f;%f;%d;%d;%d") % p.x() % p.y() % p.z() %
                      color.x() % color.y() % color.z()
               << endl;
      }
    }
    for (auto& p : vo->map_->map_points_) {
      auto& pos = p.second->pos_;
      bg_points << boost::format("%lf;%lf;%lf;255;255;255") % pos.x() %
                       pos.y() % pos.z()
                << endl;
    }
  }

  return 0;
}

int main(int argc, char** argv) {
  // 第一版测试
  return Main();
}
