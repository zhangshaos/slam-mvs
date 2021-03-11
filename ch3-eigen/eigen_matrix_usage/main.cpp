// eigen matrix usage
// ���ߣ�������

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>  // ���ܾ���Ĵ������㣨�棬����ֵ��
#include <iostream>

int main() {
  using namespace std;
  cout.setf(ios_base::fixed);
  cout.precision(3);
  printf("cout ������λС��\n");
  // Eigen �Ծ��� Matrix<Type, Rows, Cols> Ϊ�������ݵ�Ԫ������һ��ģ���ࡣ
  // ����ǰ��������Ϊ���������ͣ��У���

  // ������ֵ��ʵ�Գƾ���һ�����ԶԽǻ���
  Eigen::Matrix3f mat_33;
  mat_33 << 1, 2, 3, 0, 1, 2, 0, 0, 1;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(
      mat_33.transpose() * mat_33);
  cout << "Eigen values = " << eigen_solver.eigenvalues().transpose() << endl;
  cout << "Eigen vectors = " << eigen_solver.eigenvectors() << endl;

  // �ⷽ�̣�mat_33 * x = v_3
  // ʹ�� QR �ֽ⣨��һ������ֽ�Ϊ �������� * �����Ǿ���
  mat_33 = Eigen::Matrix3f::Random();
  Eigen::Vector3f v_3 = Eigen::VectorXf::Random(3, 1);
  cout << endl
       << mat_33 << " * x = " << v_3.transpose()
       << "\n=> x=" << mat_33.colPivHouseholderQr().solve(v_3).transpose()
       << endl;
  
  return 0;
}