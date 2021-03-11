#define _CRT_SECURE_NO_WARNINGS
#define BOOST_USE_WINDOWS_H

#include <fstream>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  using namespace std;

  vector<cv::Mat> colorImgs, depthImgs;
  vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      poses;

  ifstream fin("./pose.txt");
  if (!fin) {
    cerr << "������pose.txt��Ŀ¼�����д˳���" << endl;
    return 1;
  }

  for (int i = 0; i < 5; i++) {
    boost::format fmt("./%s/%d.%s");  //ͼ���ļ���ʽ
    colorImgs.push_back(cv::imread((fmt % "rgb" % (i + 1) % "png").str()));
    depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(),
                                   -1));  // ʹ��-1��ȡԭʼͼ��

    double data[7] = {0};
    for (auto& d : data) fin >> d;
    Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
    Eigen::Isometry3d T(q);
    T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
    poses.push_back(T);
  }

  // ������Ʋ�ƴ��
  // ����ڲ�
  double cx = 325.5;
  double cy = 253.5;
  double fx = 518.0;
  double fy = 519.0;
  double depthScale = 1000.0;

  cout << "���ڽ�ͼ��ת��Ϊ����..." << endl;

  // �������ʹ�õĸ�ʽ�������õ���XYZRGB
  typedef pcl::PointXYZRGB PointT;
  typedef pcl::PointCloud<PointT> PointCloud;

  PointCloud::Ptr pointCloud(new PointCloud);
  for (int i = 0; i < 5; i++) {
    cout << "ת��ͼ����: " << i + 1 << endl;
    cv::Mat color = colorImgs[i];
    cv::Mat depth = depthImgs[i];
    Eigen::Isometry3d T = poses[i];
    for (int v = 0; v < color.rows; v++)
      for (int u = 0; u < color.cols; u++) {
        unsigned int d = depth.ptr<unsigned short>(v)[u];  // ���ֵ 16 bit
        if (d == 0) continue;  // Ϊ0��ʾû�в�����
        Eigen::Vector3d point;
        // ����� uv => xyz
        point[2] = double(d) / depthScale;
        point[0] = (u - cx) * point[2] / fx;
        point[1] = (v - cy) * point[2] / fy;
        Eigen::Vector3d pointWorld = T * point;

        PointT p;
        p.x = pointWorld[0];
        p.y = pointWorld[1];
        p.z = pointWorld[2];
        // OpenCV ��ɫ BGR��С�ˣ���RGB����ˣ�
        p.b = color.data[v * color.step + u * color.channels()];
        p.g = color.data[v * color.step + u * color.channels() + 1];
        p.r = color.data[v * color.step + u * color.channels() + 2];
        //unsigned char* bgr = color.ptr<unsigned char>(v) + u * color.channels();
        //unsigned char b = *bgr, g = *(bgr + 1), r = *(bgr + 2);
        pointCloud->points.push_back(p);
      }
  }
  
  pointCloud->is_dense = false;
  cout << "���ƹ���" << pointCloud->size() << "����." << endl;
  pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
  return 0;
}