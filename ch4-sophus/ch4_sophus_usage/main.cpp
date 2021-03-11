#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

int main(int argc, char** argv) {
  cout.setf(ios_base::fixed);
  cout.precision(3);

  // ��Z��ת90�ȵ���ת����
  Eigen::Matrix3d R =
      Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

  Sophus::SO3d SO3_R(R);  // Sophus::SO(3)����ֱ�Ӵ���ת������
  // Sophus::SO3d SO3_v(0, 0, M_PI / 2);  // ��ɴ���ת�������죬�°�Ĳ�����
  Eigen::Quaterniond q(R);  // ������Ԫ��
  Sophus::SO3d SO3_q(q);
  // ������﷽ʽ���ǵȼ۵�
  // ���SO(3)ʱ����so(3)��ʽ���
  cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
  // cout << "SO(3) from vector: " << SO3_v << endl;
  cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << endl;

  // ʹ�ö���ӳ�������������
  Eigen::Vector3d so3 = SO3_R.log();
  cout << "so3 = " << so3.transpose() << endl;
  // hat Ϊ���������Գƾ���
  cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;
  // ��Եģ�veeΪ���ԳƵ�����
  cout << "so3 hat vee= "
       << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose()
       << endl;  // transpose������Ϊ���������һЩ

  // �����Ŷ�ģ�͵ĸ���
  Eigen::Vector3d update_so3(0.1, 0, 0);  //���������Ϊ��ô��
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "�Ŷ���" << update_so3.transpose() << endl;
  cout << "SO3 updated =\n" << SO3_updated.matrix() << endl;

  cout << "************���Ƿָ���*************" << endl;
  // ��SE(3)������ͬС��
  Eigen::Vector3d t(1, 0, 0);  // ��X��ƽ��1
  Sophus::SE3d SE3_Rt(R, t);   // ��R,t����SE(3)
  Sophus::SE3d SE3_qt(q, t);   // ��q,t����SE(3)
  cout << "SE3 from R,t= " << endl << SE3_Rt.matrix() << endl;
  cout << "SE3 from q,t= " << endl << SE3_qt.matrix() << endl;
  // �����se(3) ��һ����ά���������������typedefһ��
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "se3 = " << se3.transpose() << endl;
  // �۲�������ᷢ����Sophus�У�se(3)��ƽ����ǰ����ת�ں�.
  // ͬ���ģ���hat��vee�������
  cout << "se3 hat = " << endl << Sophus::SE3d::hat(se3) << endl;
  cout << "se3 hat vee = "
       << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

  // �����ʾһ�¸���
  Vector6d update_se3;  //������
  update_se3.setZero();
  update_se3(0, 0) = 0.1;
  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
  cout << "�Ŷ���" << update_se3.transpose() << endl;
  cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

  return 0;
}