// 使用此文件测试 OpenCV 的 ORB 效果

#include <filesystem>
namespace fs = std::filesystem;
using namespace std;

#include <opencv2/opencv.hpp>

int main() {
  fs::path path1 =
      "E:\\datasets\\3d_car_instance_sample\\3d_car_instance_"
      "sample\\images\\small"
      "\\180116_053947113_Camera_5.jpg";
  fs::path path2 =
      "E:\\datasets\\3d_car_instance_sample\\3d_car_instance_"
      "sample\\images\\small"
      "\\180116_053947909_Camera_5.jpg";

  auto img1 = cv::imread(path1.string()), img2 = cv::imread(path2.string());
  auto orb = cv::ORB::create(10000);

  // 开始检测特征点
  vector<cv::KeyPoint> kps1, kps2;
  orb->detect(img1, kps1);
  orb->detect(img2, kps2);

  cv::Mat desp1, desp2;
  orb->compute(img1, kps1, desp1);
  orb->compute(img2, kps2, desp2);

  cv::Mat img_show;
  cv::drawKeypoints(img1, kps1, img_show);
  cv::imshow("ORB KeyPoints", img_show);

  // 开始匹配特征点
  vector<cv::DMatch> matches;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.match(desp1, desp2, matches);

  double min_dist = 0, max_dist = 0;  //定义距离

  for (int i = 0; i < desp1.rows; ++i)  //遍历
  {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("Max dist: %f\n", max_dist);
  printf("Min dist: %f\n", min_dist);

  vector<cv::DMatch> good_matches;
  for (int j = 0; j < desp1.rows; ++j) {
    if (matches[j].distance <= max(2 * min_dist, 0.5f * (min_dist + max_dist)))
      good_matches.push_back(matches[j]);
  }

  cv::Mat img_match;

  drawMatches(img1, kps1, img2, kps2, matches, img_match);
  imshow("所有匹配点对", img_match);

  cv::Mat img_goodmatch;  //筛选后的匹配点图
  drawMatches(img1, kps1, img2, kps2, good_matches, img_goodmatch);
  imshow("筛选后的匹配点对", img_goodmatch);

  cv::waitKey(0);
  return 0;
}
