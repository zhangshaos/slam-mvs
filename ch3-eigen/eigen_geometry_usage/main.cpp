// eigen geometry usage
// 作者：章星明

#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>

#define M_PI 3.14159265354

int main() {
  using namespace std;
  // 旋转向量使用 AngleAxis, 它底层不直接是 Matrix
  // ，但运算可以当作矩阵（因为重载了运算符）
  Eigen::AngleAxisd rotation_vector(
      M_PI / 4, Eigen::Vector3d(0, 0, 1));  // 沿 Z 轴旋转 45 度

  // 欧氏变换矩阵使用 Eigen::Isometry
  Eigen ::Isometry3d T =
      Eigen::Isometry3d::Identity();  //虽然称为 3d ，实质上是 4＊4 的矩阵
  T.rotate(rotation_vector);          //按照 rotation_vector 进行旋转
  T.pretranslate(Eigen::Vector3d(1, 3, 4));  //把平移向量设成 (1,3,4)
  cout << "Transform matrix = \n" << T.matrix() << endl;

  // 对于仿射和射影变换，使用 Eigen::Affine3d 和 Eigen::Projective3d 即可

  // 四元数
  // 可以直接把 AngleAxis 赋值给四元数，反之亦然
  Eigen ::Quaterniond q = Eigen::Quaterniond(rotation_vector);
  cout << "quaternion = \n"
       << q.coeffs()
       << endl;  // 请注意 coeffs 的顺序是 (x,y,z,w), w 为实部，前三者为虚
  // 也可以把旋转矩阵赋给它
  // q = Eigen::Quaterniond(rotation_matrix);
  // cout << "quaternion = \n" << q.coeffs() << endl;

  return 0;
}