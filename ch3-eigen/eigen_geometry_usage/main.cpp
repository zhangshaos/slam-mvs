// eigen geometry usage
// ���ߣ�������

#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>

#define M_PI 3.14159265354

int main() {
  using namespace std;
  // ��ת����ʹ�� AngleAxis, ���ײ㲻ֱ���� Matrix
  // ����������Ե���������Ϊ�������������
  Eigen::AngleAxisd rotation_vector(
      M_PI / 4, Eigen::Vector3d(0, 0, 1));  // �� Z ����ת 45 ��

  // ŷ�ϱ任����ʹ�� Eigen::Isometry
  Eigen ::Isometry3d T =
      Eigen::Isometry3d::Identity();  //��Ȼ��Ϊ 3d ��ʵ������ 4��4 �ľ���
  T.rotate(rotation_vector);          //���� rotation_vector ������ת
  T.pretranslate(Eigen::Vector3d(1, 3, 4));  //��ƽ��������� (1,3,4)
  cout << "Transform matrix = \n" << T.matrix() << endl;

  // ���ڷ������Ӱ�任��ʹ�� Eigen::Affine3d �� Eigen::Projective3d ����

  // ��Ԫ��
  // ����ֱ�Ӱ� AngleAxis ��ֵ����Ԫ������֮��Ȼ
  Eigen ::Quaterniond q = Eigen::Quaterniond(rotation_vector);
  cout << "quaternion = \n"
       << q.coeffs()
       << endl;  // ��ע�� coeffs ��˳���� (x,y,z,w), w Ϊʵ����ǰ����Ϊ��
  // Ҳ���԰���ת���󸳸���
  // q = Eigen::Quaterniond(rotation_matrix);
  // cout << "quaternion = \n" << q.coeffs() << endl;

  return 0;
}