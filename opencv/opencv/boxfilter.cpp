//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <opencv2\opencv.hpp>  
//#include <time.h>
//#include <istream>
//void displayImageNewWindow(char *title, CvArr* img) {
//	cvNamedWindow(title, 1);
//	cvShowImage(title, img);
//}
//void box_filter(IplImage* img, IplImage* result) {
//	//cv::Mat tempimg(img);
//	// init part 
//	CvScalar s;
//	int width = img->width, height = img->height;
//	int m_w = 50, m_h = 50; // window_size 
//	int boxwidth = width - m_w, boxheight = height - m_h;
//	int *sum = (int *)malloc(boxwidth *boxheight * sizeof(double));
//	int *buff = (int *)malloc(width * sizeof(double));
//	memset(sum, 0, boxwidth *boxheight * sizeof(int));
//	memset(buff, 0, width * sizeof(int));
//
//	// set buff:from 0 to 4 rows,per col 
//	int x, y, j;
//	for (y = 0; y<m_h; y++) {
//		for (x = 0; x<width; x++) {
//			uchar pixel = CV_IMAGE_ELEM(img, uchar, y, x);
//			buff[x] += pixel;
//			// printf("%d:%d\n",x ,buff[x]); 
//		}
//	}
//
//	for (y = 0; y<height - m_h; y++) {
//		int Xsum = 0;
//
//		for (j = 0; j<m_w; j++) {
//			Xsum += buff[j]; 
//			// sum of pixel from (0,0 ) to (m_h,m_w) (also x = 0) 
//		}
//
//		for (x = 0; x<boxwidth; x++) {
//			if (x != 0) {
//				Xsum = Xsum - buff[x - 1] + buff[m_w - 1 + x]; 
//				// Xsum:sum of cols range from x to x+m_w ,rows range from 0 to 4 
//			}
//			sum[y*boxwidth + x] = (float)Xsum;
//		}
//
//		for (x = 0; x<width; x++) {
//			uchar pixel = CV_IMAGE_ELEM(img, uchar, y, x); 
//			// img[y *width + x];     
//			uchar pixel2 = CV_IMAGE_ELEM(img, uchar, y + m_h, x); 
//			// img[(y+mheight) *width + x];     
//			buff[x] = buff[x] - pixel + pixel2;
//		}
//	}
//	// 遍歷，得到每個點的和，傳給矩陣result 
//	for (y = 0; y<height; y++) {
//		for (x = 0; x<width; x++) {
//			if (y>m_h / 2 && y<height - m_h / 2 && x>m_w / 2 && x< width - m_w / 2) {
//				s.val[0] = sum[(y - m_h / 2) *boxwidth + (x - m_h / 2)] / (m_h* m_w);
//				cvSet2D(result, y, x, s);
//			}
//			else {
//				s.val[0] = 255;
//				cvSet2D(result, y, x, s);
//				
//			} // end else 
//		} // end the first for 
//	} // end the second for 
//}
//int main(int argc, char ** argv) {
//	int a = 5;
//	std::cout << (a-1)/2 << std::endl;
//	std::cout << a / 2 << std::endl;
//	IplImage* left = cvLoadImage("equ_750_517_2.jpg");
//	//cv::Mat matimg(left);
//	IplImage* dst = cvCreateImage(cvGetSize(left), 8, 1);
//	IplImage* mat = cvCreateImage(cvGetSize(left), 8, 1);
//	cvZero(mat);
//	cvCvtColor(left, dst, CV_RGB2GRAY);
//	displayImageNewWindow(" src ", dst);
//	
//		//計時
//		double time0;
//		time0 = static_cast<double>(cv::getTickCount());
//	box_filter(dst, mat);
//		time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
//		std::cout << time0 << std::endl;
//	
//	displayImageNewWindow(" mat ", mat);
//	//cv::imshow("matimg", matimg);
//	cvWaitKey();
//	return  0;
//}