
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

#define EPSILON (1e-6)

Mat_<double> LinearLSTriangulation(
    Point3d &u,   // homogenous image point (u,v,1)
    Matx34d &P,   // camera 1 matrix
    Point3d &u1,  // homogenous image point in 2nd camera
    Matx34d &P1   // camera 2 matrix
) {
  // build matrix A for homogenous equation system Ax = 0
  // assume X = (x,y,z,1), for Linear-LS method
  // which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
  Matx43d A(u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1),
            u.x * P(2, 2) - P(0, 2), u.y * P(2, 0) - P(1, 0),
            u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2),
            u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1),
            u1.x * P1(2, 2) - P1(0, 2), u1.y * P1(2, 0) - P1(1, 0),
            u1.y * P1(2, 1) - P1(1, 1), u1.y * P1(2, 2) - P1(1, 2));
  Mat_<double> B = (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)),
                    -(u.y * P(2, 3) - P(1, 3)), -(u1.x * P1(2, 3) - P1(0, 3)),
                    -(u1.y * P1(2, 3) - P1(1, 3)));

  Mat_<double> X;
  solve(A, B, X, DECOMP_SVD);

  return X;
}

Mat_<double> IterativeLinearLSTriangulation(
    Point3d &u,   // homogenous image point (u,v,1)
    Matx34d &P,   // camera 1 matrix
    Point3d &u1,  // homogenous image point in 2nd camera
    Matx34d &P1   // camera 2 matrix
) {
  double wi = 1, wi1 = 1;
  Mat_<double> X(4, 1);

  for (int i = 0; i < 10; i++) {  // Hartley suggests 10 iterations at most
    Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;
    // recalculate weights
    double p2x = Mat_<double>(Mat_<double>(P).row(2) * X)(0);
    double p2x1 = Mat_<double>(Mat_<double>(P1).row(2) * X)(0);

    // breaking point
    if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

    wi = p2x;
    wi1 = p2x1;

    // reweight equations and solve
    Matx43d A(
        (u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi,
        (u.x * P(2, 2) - P(0, 2)) / wi, (u.y * P(2, 0) - P(1, 0)) / wi,
        (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
        (u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1,
        (u1.x * P1(2, 2) - P1(0, 2)) / wi1, (u1.y * P1(2, 0) - P1(1, 0)) / wi1,
        (u1.y * P1(2, 1) - P1(1, 1)) / wi1, (u1.y * P1(2, 2) - P1(1, 2)) / wi1);
    Mat_<double> B =
        (Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi,
         -(u.y * P(2, 3) - P(1, 3)) / wi, -(u1.x * P1(2, 3) - P1(0, 3)) / wi1,
         -(u1.y * P1(2, 3) - P1(1, 3)) / wi1);

    solve(A, B, X_, DECOMP_SVD);
    X(0) = X_(0);
    X(1) = X_(1);
    X(2) = X_(2);
    X(3) = 1.0;
  }

  return X;
}

void Triangulation(const std::vector<KeyPoint>& keypoint_1,
                   const std::vector<KeyPoint>& keypoint_2,
                   const std::vector<DMatch>& matches,
                   const Mat& K, const Mat& R,
                   const Mat& t, std::vector<Point3d>& points) {
  Mat Kinv = K.inv();
  Matx34d proj1(3, 4), proj2(3, 4);
  proj1 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
  proj2 << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
           R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
           R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0);
  for (auto& m : matches) {
    const Point2f &pt1 = keypoint_1[m.queryIdx].pt,
                  &pt2 = keypoint_2[m.trainIdx].pt;
    Point3d pt1_(pt1.x, pt1.y, 1.), pt2_(pt2.x, pt2.y, 1.);
    // 将齐次像素坐标 => 归一化坐标
    pt1_ = Vec3d(Mat1d(Kinv * Mat1d(pt1_)));
    pt2_ = Vec3d(Mat1d(Kinv * Mat1d(pt2_)));
    auto p_ = IterativeLinearLSTriangulation(pt1_, proj1, pt2_, proj2);
    double w = p_.at<double>(3, 0);
    points.emplace_back(p_.at<double>(0, 0) / w,
                        p_.at<double>(1, 0) / w,
                        p_.at<double>(2, 0) / w);
  }
}