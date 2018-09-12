#include "cv3imageprocessing.h"
cv3ImageProcessing::cv3ImageProcessing(void){
}
cv3ImageProcessing::~cv3ImageProcessing(void){
}
//cv2ImageProcessing::cv2ImageProcessing(const String &filename){
//  orgImage = imRead(filename);
//}
Mat cv3ImageProcessing::ImRead(const string &filename){
  orgImage = imread(filename,-1);
  return orgImage;
}
void cv3ImageProcessing::ImShow(const string &winname, Mat &cvImg){
  namedWindow(winname);
  imshow(winname,cvImg);
}
void cv3ImageProcessing::ImWrite(const string &filename, Mat &cvImg){
  imwrite(filename, cvImg);
}
void cv3ImageProcessing::SplitAlpha(Mat &Foreground, Mat &Alpha, const Mat &SrcImg){
	split(SrcImg, channel);
	//將Alpha通道範圍改為0~1
	Alpha = channel[3]/255;
	cvtColor(SrcImg, Foreground, COLOR_RGBA2RGB);
}
Mat cv3ImageProcessing::AlphaBlend(const Mat &Foreground, const Mat &Background, const Mat &Alpha){
  //split foreGround
  split(Foreground, channel);
  multiply(channel[0], Alpha, channel[0]);
  multiply(channel[1], Alpha, channel[1]);
  multiply(channel[2], Alpha, channel[2]);
  merge(channel, Foreground);
  //split backGround
  split(Background, channel);
  multiply(channel[0], Scalar::all(1.0) - Alpha, channel[0]);
  multiply(channel[1], Scalar::all(1.0) - Alpha, channel[1]);
  multiply(channel[2], Scalar::all(1.0) - Alpha, channel[2]);
  merge(channel, Background);
  //add foreGround and backGround
  add(Foreground, Background, orgImage);
  return orgImage;
}
void cv3ImageProcessing::ImBGR2Gray(Mat &DstImg, const Mat &SrcImg){
  cvtColor(SrcImg, DstImg, COLOR_BGR2GRAY);
}
Mat cv3ImageProcessing::ImBGR2Gray(const Mat &SrcImg){
  cvtColor(SrcImg, processImage, COLOR_BGR2GRAY);
  return processImage;
}
void cv3ImageProcessing::CalcGrayHist(Mat &GrayHist, const Mat &SrcGray){
  //int histSize = 256;
  //float range[] = {0, 255};
  //const float* histRange = {range};
  //calcHist(&SrcGray, 1, 0, Mat(), GrayHist, 1, &histSize, &histRange);
  /*************************************************/
  GrayHist = Mat(256, 1, CV_32F, Scalar::all(0));
  const uchar *pix;
  for (int j = 0; j < SrcGray.rows; j++) {
	  for (int k = 0; k < SrcGray.cols; k++) {
		  pix = SrcGray.ptr<uchar>(j, k);
		  GrayHist.at<float>(*pix, 0) += 1;
	  }
  }
}
void cv3ImageProcessing::ShowGrayHist(const string &winname, const Mat &GrayHist){

	int bin_w = 2;
	int space_w = 2;
	int side_w = 10;
	int hist_w = (256 * bin_w) + (255 * space_w) + side_w;
    int hist_h = 400;
    int histSize = 256;
    Mat gray_hist(hist_h, hist_w, CV_8UC3, Scalar::all(0));
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));
    /// Normalize the result to [ 0, histImage.rows-10]
	cv::normalize(GrayHist, GrayHist, 0, histImage.rows - 10, NORM_MINMAX);

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, 
			Point(((i-1)*(bin_w + space_w)) + (bin_w/2) + 5, hist_h - cvRound(GrayHist.at<float>(i-1))) ,
            Point(((i  )*(bin_w + space_w)) + (bin_w/2) + 5, hist_h - cvRound(GrayHist.at<float>(i)  )),
            Scalar( 255, 255, 255), 1);
    }
    for (int j=0; j<histSize; ++j)
    {
        int val = saturate_cast<int>(GrayHist.at<float>(j));
        rectangle(gray_hist,
			      Point((j*(bin_w + space_w)) + 5, gray_hist.rows), 
			      Point((j*(bin_w + space_w)) + 5 + bin_w, gray_hist.rows-val),
			      Scalar(255,255,255),
			      CV_FILLED,
			      8);
    }
    /// Display
	string filename = "image//" + winname + ".jpg";
	//imwrite(filename, gray_hist);
	//cout << hist_w;
    namedWindow(winname, CV_WINDOW_AUTOSIZE );
    //cv3ImageProcessing::ImShow(winname, histImage);
	imshow(winname,gray_hist);
}
void cv3ImageProcessing::CalcColorHist(vector<Mat> &ColorHist, const Mat &SrcColor){
  vector <Mat> rgb_planes;
  split(SrcColor, rgb_planes );
  Mat r_hist(256, 1, CV_32F, Scalar::all(0));
  Mat g_hist(256, 1, CV_32F, Scalar::all(0));
  Mat b_hist(256, 1, CV_32F, Scalar::all(0));
  const uchar *pix;
  for (int j = 0; j < rgb_planes[0].rows; j++) {
	  for (int k = 0; k < rgb_planes[0].cols; k++) {
		  pix = rgb_planes[0].ptr<uchar>(j, k);
		  r_hist.at<float>(*pix, 0) += 1;
	  }
  }
  for (int j = 0; j < rgb_planes[1].rows; j++) {
	  for (int k = 0; k < rgb_planes[1].cols; k++) {
		  pix = rgb_planes[1].ptr<uchar>(j, k);
		  g_hist.at<float>(*pix, 0) += 1;
	  }
  }
  for (int j = 0; j < rgb_planes[2].rows; j++) {
	  for (int k = 0; k < rgb_planes[2].cols; k++) {
		  pix = rgb_planes[2].ptr<uchar>(j, k);
		  b_hist.at<float>(*pix, 0) += 1;
	  }
  }
  /*************************************************/
  //int histSize = 256;
  //float range[] = { 0, 255 };
  //const float* histRange = { range };
  ////  imshow("R",rgb_planes[0]);
  ////  imshow("G",rgb_planes[1]);
  ////  imshow("B",rgb_planes[2]);
  //Mat r_hist, g_hist, b_hist;
  ///// Compute the histograms:
  //calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);
  //calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
  //calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
  ColorHist.push_back(r_hist);
  ColorHist.push_back(g_hist);
  ColorHist.push_back(b_hist);
}
void cv3ImageProcessing::ShowColorHist(const string &winname, const vector<Mat> &ColorHist){
	int bin_w = 2;
	int space_w = 2;
	int side_w = 10;
	int hist_w = (256 * bin_w) + (255 * space_w) + side_w;
    int hist_h = 400;
    int histSize = 256;
   Mat rgb_hist[3];
   for(int i=0; i<3; ++i)
   {
       rgb_hist[i] = Mat(hist_h, hist_w, CV_8UC3, Scalar::all(0));
   }
   Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));
   /// Normalize the result to [ 0, histImage.rows-10]
   normalize(ColorHist[0], ColorHist[0], 0, histImage.rows-10, NORM_MINMAX);
   normalize(ColorHist[1], ColorHist[1], 0, histImage.rows-10, NORM_MINMAX);
   normalize(ColorHist[2], ColorHist[2], 0, histImage.rows-10, NORM_MINMAX);

   /// Draw for each channel
   for( int i = 1; i < histSize; i++ )
   {
	   line(histImage,
		   Point(((i - 1)*(bin_w + space_w)) + (bin_w / 2) + 5, hist_h - cvRound(ColorHist[0].at<float>(i - 1))),
		   Point(((i)*(bin_w + space_w)) + (bin_w / 2) + 5, hist_h - cvRound(ColorHist[0].at<float>(i))),
		   Scalar(0, 0, 255), 1);
	   line(histImage,
		   Point(((i - 1)*(bin_w + space_w)) + (bin_w / 2) + 5, hist_h - cvRound(ColorHist[1].at<float>(i - 1))),
		   Point(((i)*(bin_w + space_w)) + (bin_w / 2) + 5, hist_h - cvRound(ColorHist[1].at<float>(i))),
		   Scalar(0, 255, 0), 1);
	   line(histImage,
		   Point(((i - 1)*(bin_w + space_w)) + (bin_w / 2) + 5, hist_h - cvRound(ColorHist[2].at<float>(i - 1))),
		   Point(((i)*(bin_w + space_w)) + (bin_w / 2) + 5, hist_h - cvRound(ColorHist[2].at<float>(i))),
		   Scalar(255, 0, 0), 1);
      
   }

   for (int j=0; j<histSize; ++j)
   {
       int val = saturate_cast<int>(ColorHist[0].at<float>(j));
	   rectangle(rgb_hist[0],
		   Point((j*(bin_w + space_w)) + 5, rgb_hist[0].rows),
		   Point((j*(bin_w + space_w)) + 5 + bin_w, rgb_hist[0].rows - val),
		   Scalar(0, 0, 255),
		   CV_FILLED,
		   8);
	   val = saturate_cast<int>(ColorHist[1].at<float>(j));
	   rectangle(rgb_hist[1],
		   Point((j*(bin_w + space_w)) + 5, rgb_hist[1].rows),
		   Point((j*(bin_w + space_w)) + 5 + bin_w, rgb_hist[1].rows - val),
		   Scalar(0, 255, 0),
		   CV_FILLED,
		   8);
	   val = saturate_cast<int>(ColorHist[2].at<float>(j));
	   rectangle(rgb_hist[2],
		   Point((j*(bin_w + space_w)) + 5, rgb_hist[2].rows),
		   Point((j*(bin_w + space_w)) + 5 + bin_w, rgb_hist[2].rows - val),
		   Scalar(255, 0, 0),
		   CV_FILLED,
		   8);
      
   }
   string filename = "image//" + winname + ".jpg";
   /// Display
   namedWindow(winname, CV_WINDOW_AUTOSIZE );
   imshow(winname, histImage);
   imwrite(filename, histImage);
   //imshow("R", rgb_hist[0]);
   //imshow("G", rgb_hist[1]);
   //imshow("B", rgb_hist[2]);
}
void cv3ImageProcessing::MonoEqualize(Mat &DstGray, const Mat &SrcGray){
	Mat GrayHist(1, 256, CV_32F, Scalar::all(0));
	const uchar *pix;
	for (int j = 0; j < SrcGray.rows; j++){
		for (int k = 0; k < SrcGray.cols; k++) {
			pix = SrcGray.ptr<uchar>(j, k);
			GrayHist.at<float>(0, *pix) += 1;
		}
	}
	//cv3ImageProcessing::CalcGrayHist(GrayHist, SrcGray);
	float addf = 0;
	Mat LookUpTable(1, 256, CV_8U);
	for (int i = 0; i < 256; i++) {
		addf = addf + GrayHist.at<float>(0, i);
		GrayHist.at<float>(0, i) = addf * 255 / (SrcGray.rows * SrcGray.cols);
		LookUpTable.at<uchar>(0, i) = saturate_cast<uchar>(GrayHist.at<float>(0, i));
	}
	LUT(SrcGray, LookUpTable, DstGray);
	//cout << LookUpTable;
	//equalizeHist(SrcGray, DstGray);
}
void cv3ImageProcessing::ColorEqualize(Mat &DstColor, const Mat &SrcColor, const CV2_COLOREQUALIZE_TYPE Type){
  switch (Type) {
  case USE_RGB:
    split(SrcColor, channel);
	cv3ImageProcessing::MonoEqualize(channel[0], channel[0]);
	cv3ImageProcessing::MonoEqualize(channel[1], channel[1]);
	cv3ImageProcessing::MonoEqualize(channel[2], channel[2]);
    //equalizeHist(channel[0], channel[0]);
    //equalizeHist(channel[1], channel[1]);
    //equalizeHist(channel[2], channel[2]);
    merge(channel, DstColor);
    break;
  case USE_HSV:
	cvtColor(SrcColor, DstColor, CV_BGR2HSV);
	split(DstColor, channel);
	cv3ImageProcessing::MonoEqualize(channel[2], channel[2]);
	merge(channel, DstColor);
	cvtColor(DstColor, DstColor, CV_HSV2BGR);
    break;
  case USE_YUV:
	cvtColor(SrcColor, DstColor, CV_BGR2YUV);
	split(DstColor, channel);
	cv3ImageProcessing::MonoEqualize(channel[0], channel[0]);
	merge(channel, DstColor);
	cvtColor(DstColor, DstColor, CV_YUV2BGR);
    break;
  }
}
void cv3ImageProcessing::HistMatching(Mat &DstImg, const Mat &SrcImg, const Mat &RefImg){
	
  cvtColor(SrcImg, SrcImg, CV_BGR2HSV);
  cvtColor(RefImg, RefImg, CV_BGR2HSV);
  vector<Mat>channelRef;
  vector<Mat>channelSrc;
  Mat src_hist, ref_hist;
  split(SrcImg, channelSrc);
  cv3ImageProcessing::CalcGrayHist(src_hist, channelSrc[2]);
  //cv2ImageProcessing::ShowGrayHist("srcHist", src_hist);
  split(RefImg, channelRef);
  cv3ImageProcessing::CalcGrayHist(ref_hist, channelRef[2]);
  //cv2ImageProcessing::ShowGrayHist("dstHist", dst_hist);

    float src_cdf[256] = { 0 };
    float ref_cdf[256] = { 0 };

    // 源圖像和目標圖像的大小不一樣，要將得到的直方圖進行歸一化處理
    src_hist /= (SrcImg.rows * SrcImg.cols);
    ref_hist /= (RefImg.rows * RefImg.cols);

    // 計算原始直方圖和規定直方圖的累積概率
    for (int i = 0; i < 256; i++)
    {
        if (i == 0)
        {
            src_cdf[i] = src_hist.at<float>(i);
            ref_cdf[i] = ref_hist.at<float>(i);
        }
        else
        {
            src_cdf[i] = src_cdf[i - 1] + src_hist.at<float>(i);
            ref_cdf[i] = ref_cdf[i - 1] + ref_hist.at<float>(i);
        }
    }

	
    // 累積概率的差值
    float diff_cdf[256][256];
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			diff_cdf[i][j] = fabs(src_cdf[i] - ref_cdf[j]);
		}
	}
    // 構建灰度級映射表
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++)
    {
        // 查找源灰度級為ｉ的映射灰度
        //　和ｉ的累積概率差值最小的規定化灰度
        float min = diff_cdf[i][0];
        int index = 0;
        for (int j = 1; j < 256; j++)
        {
            if (min > diff_cdf[i][j])
            {
                min = diff_cdf[i][j];
                index = j;
            }
        }
        lut.at<uchar>(i) = static_cast<uchar>(index);
    }

    // 應用查找表，做直方圖規定化
    LUT(channelSrc[2], lut, channelSrc[2]);
    merge(channelSrc, DstImg);
    cvtColor(DstImg, DstImg, CV_HSV2BGR);
    cvtColor(SrcImg, SrcImg, CV_HSV2BGR);
    cvtColor(RefImg, RefImg, CV_HSV2BGR);
}
void cv3ImageProcessing::Smooth2D(Mat &DstImg, const Mat &SrcImg, int ksize, const CV2_IMSMOOTH_TYPE Type, double sigma){
  Size kksize(ksize, ksize);
  Mat SmoothKernel;
  switch (Type) {
  case BLUR:
	  if(SrcImg.channels() == 1)
		  cv3ImageProcessing::BoxFilter(DstImg, SrcImg, ksize);
	  SmoothKernel = Mat(ksize, ksize, CV_32FC1, Scalar::all(1));
	  SmoothKernel /= ksize * ksize;
	  filter2D(SrcImg, DstImg, CV_8U, SmoothKernel);
    break;
  case BOX:
	if (SrcImg.channels() == 1)
		cv3ImageProcessing::BoxFilter(DstImg, SrcImg, ksize);
	SmoothKernel = Mat(ksize, ksize, CV_32FC1, Scalar::all(1));
	SmoothKernel /= ksize * ksize;
	filter2D(SrcImg, DstImg, CV_8U, SmoothKernel);
    break;
  case GAUSSIAN:
	cv3ImageProcessing::CreatGaussianKermel(SmoothKernel, ksize, sigma);
	sepFilter2D(SrcImg, DstImg, CV_8U, SmoothKernel, SmoothKernel);
    break;
  case MEDIAN:
	cv3ImageProcessing::MedianFilter(DstImg, SrcImg, ksize);
    //medianBlur(SrcImg, DstImg, ksize);
	//for (int i = 0; i < SrcImg.rows; i++) {
	//	for (int j = 0; j < SrcImg.cols; j++) {
	//		int addR = 0, addG = 0, addB = 0;
	//		for (int k = 0; k < kksize.width; k++) {
	//			for (int l = 0; l < kksize.height; l++) {
	//				addR = addR + saturate_cast<int>(tempimg.at<Vec3b>(i + k, j + l)[0]);
	//				addG = addG + saturate_cast<int>(tempimg.at<Vec3b>(i + k, j + l)[1]);
	//				addB = addB + saturate_cast<int>(tempimg.at<Vec3b>(i + k, j + l)[2]);
	//			}
	//		}
	//	DstImg.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(addR / (ksize * ksize));
	//	DstImg.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(addG / (ksize * ksize));
	//	DstImg.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(addB / (ksize * ksize));
	//	}
	//}
    break;
  case BILATERAL:
    bilateralFilter(SrcImg, DstImg, 1, 150, 150);
    break;
  }
  //free(sum);
  //free(buff);
}
void cv3ImageProcessing::EdgeDetect(Mat &DstImg, const Mat &SrcImg, const CV2_EDGEDETECT_TYPE Type){
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  switch (Type) {
  case SOBEL:
    Sobel(SrcImg, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Sobel(SrcImg, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, DstImg);
    threshold(DstImg, DstImg, 80, 255, THRESH_BINARY | THRESH_OTSU);
    break;
  case CANNY:
    Canny(SrcImg, DstImg, 50, 150, 3);
    //(in,out,depth,x方向微分階數,y方向微分階數,縮放值)
    threshold(DstImg, DstImg, 128, 255, THRESH_BINARY_INV);  //反轉影像，讓邊緣呈現黑線
    break;
  case SCHARR:
    Scharr(SrcImg, DstImg, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(DstImg, DstImg);
    threshold(DstImg, DstImg, 80, 255, THRESH_BINARY | THRESH_OTSU);
    break;
  case LAPLACE:
    Laplacian(SrcImg, DstImg, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    //(in,out,depth,核心預設1輸入值必為正整數)
    convertScaleAbs(DstImg, DstImg);  //轉成CV_8U
    threshold(DstImg, DstImg, 80, 255, THRESH_BINARY | THRESH_OTSU);
    break;
  }
}
void cv3ImageProcessing::Conv2D(Mat &DstImg, const Mat &SrcImg, const Mat &Kermel){
  filter2D(SrcImg, DstImg, SrcImg.depth(), Kermel);
}
void cv3ImageProcessing::ImSharpening(Mat &DstImg, const Mat &SrcImg, const CV2_SHARPENING_TYPE Type1, const CV2_IMSMOOTH_TYPE Type2){
  Mat kernal(3, 3, CV_32F, Scalar(0));
  Mat kernal2(3, 3, CV_32F, Scalar(-1));
  Mat kernal3(5, 5, CV_32F, Scalar(0));
  Mat gain(3, 3, CV_32F, Scalar(1));
  cv3ImageProcessing::Smooth2D(DstImg, SrcImg, 15, Type2);
  switch (Type1) {
  case LAPLACE_TYPE1:
    kernal.at<float>(1,1) = 5.0;
    kernal.at<float>(0,1) = -1.0;
    kernal.at<float>(2,1) = -1.0;
    kernal.at<float>(1,0) = -1.0;
    kernal.at<float>(1,2) = -1.0;
    kernal = kernal.mul(gain);
    cv3ImageProcessing::Conv2D(DstImg, DstImg, kernal);
    break;
  case LAPLACE_TYPE2:
    kernal2.at<float>(1,1) = 9.0;
    cv3ImageProcessing::Conv2D(DstImg, DstImg, kernal2);
    break;
  case SECOND_ORDER_LOG:
    kernal3.at<float>(2,2) = -15.0;
    kernal3.at<float>(2,1) = 2.0;
    kernal3.at<float>(2,3) = 2.0;
    kernal3.at<float>(1,2) = 2.0;
    kernal3.at<float>(3,2) = 2.0;
    kernal3.at<float>(0,2) = 1.0;
    kernal3.at<float>(4,2) = 1.0;
    kernal3.at<float>(1,1) = 1.0;
    kernal3.at<float>(1,3) = 1.0;
    kernal3.at<float>(2,0) = 1.0;
    kernal3.at<float>(2,4) = 1.0;
    kernal3.at<float>(3,1) = 1.0;
    kernal3.at<float>(3,3) = 1.0;
    cv3ImageProcessing::Conv2D(DstImg, DstImg, kernal3);
    addWeighted(SrcImg, 0.5, DstImg, 0.5, 0, DstImg);
    break;
  case UNSHARP_MASK:
    double sigma = 3;
    int threshold = 1;
    int nAmount = 200;
    float amount = nAmount/100.0f;
    Mat imgBlurred;
    GaussianBlur(SrcImg, imgBlurred, Size(), sigma, sigma);

    Mat lowContrastMask = abs(SrcImg-imgBlurred)<threshold;
    DstImg = SrcImg*(1+amount)+imgBlurred*(-amount);
    SrcImg.copyTo(DstImg, lowContrastMask);
    break;
  }
}

void cv3ImageProcessing::BoxFilter(Mat & DstImg, const Mat & SrcImg, int ksize)
{
	
	Mat tempimg;
	DstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	int side = (ksize - 1) / 2;
	copyMakeBorder(SrcImg, tempimg, side, side, side, side, BORDER_REPLICATE);
	int width = tempimg.cols, height = tempimg.rows;
	int m_w = ksize, m_h = ksize; // window_size 
	int boxwidth = width - m_w + 1, boxheight = height - m_h + 1;
	int *sum = (int *)malloc(boxwidth *boxheight * sizeof(int));
	int *buff = (int *)malloc(width * sizeof(int));
	memset(sum, 0, boxwidth *boxheight * sizeof(int));
	memset(buff, 0, width * sizeof(int));

	for (int y = 0; y<m_h; y++) {
		for (int x = 0; x<width; x++) {
			uchar pixel = tempimg.at<uchar>(y, x);
			buff[x] += pixel;
			// printf("%d:%d\n",x ,buff[x]); 
		}
	}

	for (int y = 0; y<boxheight - 1; y++) {
		int Xsum = 0;

		for (int j = 0; j<m_w; j++) {
			Xsum += buff[j];
			// sum of pixel from (0,0 ) to (m_h,m_w) (also x = 0) 
		}

		for (int x = 0; x<boxwidth; x++) {
			if (x != 0) {
				Xsum = Xsum - buff[x - 1] + buff[m_w - 1 + x];
				// Xsum:sum of cols range from x to x+m_w ,rows range from 0 to 4 
			}
			sum[y*boxwidth + x] = Xsum;
		}

		for (int x = 0; x<width; x++) {
			uchar pixel = tempimg.at<uchar>(y, x);
			// img[y *width + x];     
			uchar pixel2 = tempimg.at<uchar>(y + m_h, x);
			// img[(y+mheight) *width + x];     
			buff[x] = buff[x] - pixel + pixel2;
		}
	}
	// 遍歷，得到每個點的和，傳給矩陣result 
	for (int y = 0; y<boxheight; y++) {
		for (int x = 0; x<boxwidth; x++) {
			DstImg.at<uchar>(y, x) = saturate_cast<uchar>((double)(sum[y * boxwidth + x] / (m_h * m_w)));
		} // end the first for 
	} // end the second for 
	free(sum);
	free(buff);
}

void cv3ImageProcessing::MedianFilter(Mat & DstImg, const Mat & SrcImg, int ksize)
{
	Mat tempimg;
	int side = (ksize - 1) / 2;
	copyMakeBorder(SrcImg, tempimg, side, side, side, side, BORDER_REPLICATE);
	int width = tempimg.cols, height = tempimg.rows;
	int m_w = ksize, m_h = ksize; // window_size
	int boxwidth = width - m_w + 1, boxheight = height - m_h + 1;
	uchar *buff = (uchar *)malloc(m_w * m_h * sizeof(uchar));
	memset(buff, 0, m_w * m_h * sizeof(uchar));
	uchar *buff2 = (uchar *)malloc(m_w * m_h * sizeof(uchar));
	memset(buff2, 0, m_w * m_h * sizeof(uchar));
	uchar *buff3 = (uchar *)malloc(m_w * m_h * sizeof(uchar));
	memset(buff3, 0, m_w * m_h * sizeof(uchar));
	if (SrcImg.channels() == 1) {
		DstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	}
	else
	{
		DstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC3);
	}
	for (int i = 0; i < boxheight; i++) {
		for (int j = 0; j < boxwidth; j++) {
			for (int k = 0; k < m_h; k++) {
				for (int l = 0; l < m_w; l++) {
					if (SrcImg.channels() == 1)
						buff[k*m_w + l] = tempimg.at<uchar>(i + k, j + l);
					else {
						buff[k*m_w + l]  = tempimg.at<Vec3b>(i + k, j + l)[0];
						buff2[k*m_w + l] = tempimg.at<Vec3b>(i + k, j + l)[1];
						buff3[k*m_w + l] = tempimg.at<Vec3b>(i + k, j + l)[2];
					}
				}
			}
			for (int k = 0; k < m_h * m_w - 1; k++) {
				uchar temp;
				for (int l = 0; l < m_h * m_w -1 - k; l++) {
					if (buff[l] > buff[l + 1]) {
						temp = buff[l];
						buff[l] = buff[l + 1];
						buff[l + 1] = temp;
					}
					if (SrcImg.channels() != 1) {
						if (buff2[l] > buff2[l + 1]) {
							temp = buff2[l];
							buff2[l] = buff2[l + 1];
							buff2[l + 1] = temp;
						}
						if (buff3[l] > buff3[l + 1]) {
							temp = buff3[l];
							buff3[l] = buff3[l + 1];
							buff3[l + 1] = temp;
						}
					}
				}
			}
			if (SrcImg.channels() == 1)
				DstImg.at<uchar>(i, j) = saturate_cast<uchar>(buff[((m_h*m_w) + 1) / 2]);
			else
			{
				DstImg.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(buff[((m_h*m_w) + 1) / 2]);
				DstImg.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(buff2[((m_h*m_w) + 1) / 2]);
				DstImg.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(buff3[((m_h*m_w) + 1) / 2]);
			}
		}
	}
	free(buff);
	free(buff2);
	free(buff3);
}

void cv3ImageProcessing::CreatGaussianKermel(Mat& Kernel, int ksize, double sigma)
{
	double *gau = (double *)malloc(ksize * sizeof(double));
	memset(gau, 0, ksize * sizeof(double));
	double sum = 0;
	for (int i = 0; i<ksize; i++) {
		gau[i] = exp(-(pow((i - (ksize - 1) / 2), 2) / pow(2 * sigma, 2)));
		sum += gau[i];
	}
	for (int i = 0; i<ksize; i++) {
		gau[i] /= sum;
	}
	//cout << gau[0] << endl
	//	<< gau[1] << endl
	//	<< gau[2] << endl;

	Kernel = Mat(ksize, 1, CV_32FC1);
	for (int i = 0; i<ksize; i++) {
		Kernel.at<float>(i, 0) = saturate_cast<float>(gau[i]);
	}
	free(gau);
}
