#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
using namespace std;

// ����ģ�͵Ķ��㣬ģ��������Ż�����ά�Ⱥ���������
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual void setToOriginImpl()  // ����
  {
    _estimate << 0, 0, 0;
  }

  virtual void oplusImpl(const double* update)  // ����
  {
    _estimate += Eigen::Vector3d(update);
  }
  // ���̺Ͷ��̣�����
  virtual bool read(istream& in) { return true; }
  virtual bool write(ostream& out) const { return true; }
};

// ���ģ�� ģ��������۲�ֵά�ȣ����ͣ����Ӷ�������
class CurveFittingEdge
    : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
  // ��������ģ�����
  void computeError() {
    const CurveFittingVertex* v =
        static_cast<const CurveFittingVertex*>(_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    _error(0, 0) = _measurement -
                   std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
  }
  virtual bool read(istream& in) { return true; }
  virtual bool write(ostream& out) const { return true; }

 public:
  double _x;  // x ֵ�� y ֵΪ _measurement
};

int main(int argc, char** argv) {
  double a = 1.0, b = 2.0, c = 1.0;  // ��ʵ����ֵ
  int N = 100;                       // ���ݵ�
  double w_sigma = 1.0;              // ����Sigmaֵ
  cv::RNG rng;                       // OpenCV�����������
  double abc[3] = {0, 0, 0};         // abc�����Ĺ���ֵ

  vector<double> x_data, y_data;  // ����

  cout << "generating data: " << endl;
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
    cout << x_data[i] << " " << y_data[i] << endl;
  }

  // ����ͼ�Ż������趨g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>
      Block;  // ÿ��������Ż�����ά��Ϊ3�����ֵά��Ϊ1
  Block::LinearSolverType* linearSolver =
      new g2o::LinearSolverDense<Block::PoseMatrixType>();  // ���Է��������
  Block* solver_ptr = new Block(
      std::unique_ptr<Block::LinearSolverType>(linearSolver));  // ����������
  // �ݶ��½���������GN, LM, DogLeg ��ѡ
  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(
          std::unique_ptr<Block>(solver_ptr));
  // g2o::OptimizationAlgorithmGaussNewton* solver = new
  // g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
  // g2o::OptimizationAlgorithmDogleg* solver = new
  // g2o::OptimizationAlgorithmDogleg( solver_ptr );
  g2o::SparseOptimizer optimizer;  // ͼģ��
  optimizer.setAlgorithm(solver);  // ���������
  optimizer.setVerbose(true);      // �򿪵������

  // ��ͼ�����Ӷ���
  CurveFittingVertex* v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(0, 0, 0));
  v->setId(0);
  optimizer.addVertex(v);

  // ��ͼ�����ӱ�
  for (int i = 0; i < N; i++) {
    CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    edge->setVertex(0, v);            // �������ӵĶ���
    edge->setMeasurement(y_data[i]);  // �۲���ֵ
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 /
                         (w_sigma * w_sigma));  // ��Ϣ����Э�������֮��
    optimizer.addEdge(edge);
  }

  // ִ���Ż�
  cout << "start optimization" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(100);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // ����Ż�ֵ
  Eigen::Vector3d abc_estimate = v->estimate();
  cout << "estimated model: " << abc_estimate.transpose() << endl;

  return 0;
}