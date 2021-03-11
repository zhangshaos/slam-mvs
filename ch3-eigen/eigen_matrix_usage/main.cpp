// eigen matrix usage
// 作者：章星明

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>  // 稠密矩阵的代数运算（逆，特征值）
#include <iostream>

int main() {
  using namespace std;
  cout.setf(ios_base::fixed);
  cout.precision(3);
  printf("cout 保留三位小数\n");
  // Eigen 以矩阵 Matrix<Type, Rows, Cols> 为基本数据单元。它是一个模板类。
  // 它的前三个参数为：数据类型，行，列

  // 求特征值（实对称矩阵一定可以对角化）
  Eigen::Matrix3f mat_33;
  mat_33 << 1, 2, 3, 0, 1, 2, 0, 0, 1;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(
      mat_33.transpose() * mat_33);
  cout << "Eigen values = " << eigen_solver.eigenvalues().transpose() << endl;
  cout << "Eigen vectors = " << eigen_solver.eigenvectors() << endl;

  // 解方程：mat_33 * x = v_3
  // 使用 QR 分解（将一个矩阵分解为 正交矩阵 * 上三角矩阵）
  mat_33 = Eigen::Matrix3f::Random();
  Eigen::Vector3f v_3 = Eigen::VectorXf::Random(3, 1);
  cout << endl
       << mat_33 << " * x = " << v_3.transpose()
       << "\n=> x=" << mat_33.colPivHouseholderQr().solve(v_3).transpose()
       << endl;
  
  return 0;
}