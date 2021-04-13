#define _USE_MATH_DEFINES
#include <ceres/ceres.h>

#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;

// ���ۺ����ļ���ģ��
struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}
  // �в�ļ���
  template <typename T>
  bool operator()(const T* const abc,  // ģ�Ͳ�������3ά
                  T* residual) const   // �в�
  {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) +
                                     abc[2]);  // y-exp(ax^2+bx+c)
    return true;
  }
  const double _x, _y;  // x,y����
};


int main(int argc, char** argv) {
  double a = 1.0, b = 2.0, c = 1.0;  // ��ʵ����ֵ
  int N = 100;                       // ���ݵ�
  double w_sigma = 1.0;              // ����Sigmaֵ
  cv::RNG rng;                       // OpenCV�����������
  double abc[3] = {1.0, 1.0, 1.0};         // abc�����Ĺ���ֵ

  vector<double> x_data, y_data;  // ����

  cout << "generating data: " << endl;
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
    cout << x_data[i] << " " << y_data[i] << endl;
  }

  // ������С��������
  ceres::Problem problem;
  for (int i = 0; i < N; i++) {
    problem
        .AddResidualBlock(  // ����������������
                            // ʹ���Զ��󵼣�ģ�������������ͣ����ά��(�в�ά��)������ά��(ÿ�������Ʋ�����ά��)��ά��Ҫ��ǰ��struct��һ��
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])),
            nullptr,  // �˺��������ﲻʹ�ã�Ϊ��
            abc       // �����Ʋ���
        );
  }

  // ���������
  ceres::Solver::Options options;  // �����кܶ������������
  options.linear_solver_type = ceres::DENSE_QR;  // ��������������
  options.minimizer_progress_to_stdout = true;   // �����cout

  ceres::Solver::Summary summary;  // �Ż���Ϣ
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);  // ��ʼ�Ż�
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // ������
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a : abc) cout << a << " ";
  cout << endl;

  return 0;
}
