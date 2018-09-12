#ifndef CV3IMAGEPROCESSING_H
#define CV3IMAGEPROCESSING_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

enum CV2_COLOREQUALIZE_TYPE{
  USE_RGB=0,
  USE_HSV,
  USE_YUV
};

enum CV2_IMSMOOTH_TYPE{
  BLUR = 0,
  BOX,
  GAUSSIAN,
  MEDIAN,
  BILATERAL
};

enum CV2_EDGEDETECT_TYPE{
  SOBEL = 0,
  CANNY,
  SCHARR,
  LAPLACE
};

enum CV2_SHARPENING_TYPE{
  LAPLACE_TYPE1 = 0,
  LAPLACE_TYPE2,
  SECOND_ORDER_LOG,
  UNSHARP_MASK
};

class cv3ImageProcessing
{
public:
//12  影像讀、寫、顯示
  cv3ImageProcessing(void);
  ~cv3ImageProcessing(void);
  Mat ImRead(const string& filename);
  void ImWrite(const string& filename, Mat& cvImg);
  void ImShow(const string& winname, Mat& cvImg);
  //3 去背合成
  void SplitAlpha(Mat& Foreground, Mat& Alpha, const Mat& SrcImg);
  Mat AlphaBlend(const Mat& Foreground, const Mat& Background, const Mat& Alpha);
  //4 BGR2Gray
  void ImBGR2Gray(Mat& DstImg, const Mat& SrcImg);
  Mat ImBGR2Gray(const Mat& SrcImg);
  //5 
  void CalcGrayHist(Mat& GrayHist, const Mat& SrcGray);
  void ShowGrayHist(const string& winname, const Mat& GrayHist);
  void CalcColorHist(vector<Mat>& ColorHist, const Mat& SrcColor);
  void ShowColorHist(const string& winname, const vector<Mat>& ColorHist);
  //6
  void MonoEqualize(Mat& DstGray,const Mat& SrcGray);
  void ColorEqualize(Mat& DstColor, const Mat& SrcColor,const CV2_COLOREQUALIZE_TYPE Type=USE_RGB);
  //7
  void HistMatching(Mat& DstImg, const Mat& SrcImg, const Mat& RefImg);
  //8
  void Smooth2D(Mat& DstImg, const Mat& SrcImg, int ksize = 15, const CV2_IMSMOOTH_TYPE Type = BLUR,double sigma = 1.0);
  //9
  void EdgeDetect(Mat& DstImg, const Mat& SrcImg, const CV2_EDGEDETECT_TYPE Type = SOBEL);
  //10
  void Conv2D(Mat& DstImg, const Mat& SrcImg, const Mat& Kermel);
  //11
  void ImSharpening(Mat& DstImg, const Mat& SrcImg, const CV2_SHARPENING_TYPE Type1 = LAPLACE_TYPE1, const CV2_IMSMOOTH_TYPE Type2 = BILATERAL);
  //12
private:
	void BoxFilter(Mat& DstImg, const Mat& SrcImg, int ksize = 15);
	void MedianFilter(Mat& DstImg, const Mat& SrcImg, int ksize = 3);
	void CreatGaussianKermel(Mat& Kernel,int ksize, double sigma);

public:

private:
  //String picture;
  Mat orgImage;
  Mat processImage;
  vector <Mat> channel;

};

#endif // CV2IMAGEPROCESSING_H
