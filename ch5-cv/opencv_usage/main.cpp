#include <chrono>
#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  cv::Mat image;
  image = cv::imread("./ubuntu.png");
  if (image.data == nullptr)  //���ݲ�����,�������ļ�������
  {
    cerr << "�ļ�" << argv[1] << "������." << endl;
    return 0;
  }

  // �ļ�˳����ȡ, �������һЩ������Ϣ
  cout << "ͼ���Ϊ" << image.cols << ",��Ϊ" << image.rows << ",ͨ����Ϊ"
       << image.channels() << endl;
  cv::imshow("image", image);
  cv::waitKey(0);
  // �ж�image������
  if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
    cout << "������һ�Ų�ɫͼ��Ҷ�ͼ." << endl;
    return 0;
  }

  // ����ͼ��, ��ע�����±�����ʽ���ʹ����������ط���
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (size_t y = 0; y < image.rows; y++) {
    // ��cv::Mat::ptr���ͼ�����ָ��
    unsigned char* row_ptr =
        image.ptr<unsigned char>(y);  // row_ptr�ǵ�y�е�ͷָ��
    for (size_t x = 0; x < image.cols; x++) {
      // ����λ�� x,y ��������
      unsigned char* data_ptr =
          &row_ptr[x * image.channels()];  // data_ptr ָ������ʵ���������
      // ��������ص�ÿ��ͨ��,����ǻҶ�ͼ��ֻ��һ��ͨ��
      for (int c = 0; c != image.channels(); c++) {
        unsigned char data = data_ptr[c];  // dataΪI(x,y)��c��ͨ����ֵ
      }
    }
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "����ͼ����ʱ��" << time_used.count() << " �롣" << endl;

  // ���� cv::Mat �Ŀ���
  // ֱ�Ӹ�ֵ�����´������
  cv::Mat image_another = image;
  // �޸� image_another �ᵼ�� image �����仯
  image_another(cv::Rect(0, 0, 100, 100)).setTo(0);  // �����Ͻ�100*100�Ŀ�����
  cv::imshow("image", image);
  cv::waitKey(0);

  // ʹ��clone��������������
  cv::Mat image_clone = image.clone();
  image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
  cv::imshow("image", image);
  cv::imshow("image_clone", image_clone);
  cv::waitKey(0);

  // ����ͼ���кܶ�����Ĳ���,�����,��ת,���ŵ�,����ƪ���Ͳ�һһ������,��ο�OpenCV�ٷ��ĵ���ѯÿ�������ĵ��÷���.
  cv::destroyAllWindows();
  return 0;
}