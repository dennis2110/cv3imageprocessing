#include <iostream>
#include"cv3imageprocessing.h"
using namespace std;
cv3ImageProcessing cv3;
void box_filter(const Mat &src, Mat &dst, int ksize) {
	//src���W�[��t�Ԫ��Ϲ�
	//dst�����n�᪺�Ϲ��A�j�p���P���[��t�ɪ��Ϲ��ۦP
	// init part
	uchar s;
	int width = src.cols, height = src.rows;
	int m_w = ksize, m_h = ksize; // window_size 
	int boxwidth = width - m_w +1, boxheight = height - m_h+1;

	cout << "width		" <<width << endl;
	cout << "height		" <<height << endl;
	cout << "m_w		" <<m_w << endl;
	cout << "m_h		" <<m_h << endl;
	cout << "boxwidth	" <<boxwidth << endl;
	cout << "boxheight	" <<boxheight << endl;

	int *sum = (int *)malloc(boxwidth *boxheight * sizeof(int));
	int *buff = (int *)malloc(width * sizeof(int));
	memset(sum, 0, boxwidth *boxheight * sizeof(int));
	memset(buff, 0, width * sizeof(int));
	dst = Mat(boxheight, boxwidth, CV_8U);
	// set buff:from 0 to 4 rows,per col 
	int x, y, j;
	for (y = 0; y<m_h; y++) {
		for (x = 0; x<width; x++) {
			uchar pixel = src.at<uchar>(y, x);
			buff[x] += pixel;
			// printf("%d:%d\n",x ,buff[x]); 
		}
	}

	for (y = 0; y<boxheight-1; y++) {
		int Xsum = 0;

		for (j = 0; j<m_w; j++) {
			Xsum += buff[j];
			// sum of pixel from (0,0 ) to (m_h,m_w) (also x = 0) 
		}

		for (x = 0; x<boxwidth; x++) {
			if (x != 0) {
				Xsum = Xsum - buff[x - 1] + buff[m_w - 1 + x];
				// Xsum:sum of cols range from x to x+m_w ,rows range from 0 to 4 
			}
			sum[y*boxwidth + x] = Xsum;
		}

		for (x = 0; x<width; x++) {
			uchar pixel = src.at<uchar>(y, x);
			// img[y *width + x];     
			uchar pixel2 = src.at<uchar>(y + m_h, x);
			// img[(y+mheight) *width + x];     
			buff[x] = buff[x] - pixel + pixel2;
		}
	}
	// �M���A�o��C���I���M�A�ǵ��x�}result 
	for (y = 0; y<boxheight; y++) {
		for (x = 0; x<boxwidth; x++) {
			s = saturate_cast<uchar>((double)(sum[(y) * boxwidth + x] / (m_h* m_w)));
			dst.at<uchar>(y, x) = s;
		} // end the first for 
	} // end the second for 
	free(sum);
	free(buff);
}
Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//�H�����Ȧ�C
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//�Ϲ��q�D�P�w
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//�Q���T
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//�H�����Ȧ�C
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//�Ϲ��q�D�P�w
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//�����T
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}
void HistMatchingGray(Mat &DstImg, const Mat &SrcImg, const Mat &RefImg) {

	Mat src_hist, ref_hist;
	cv3.CalcGrayHist(src_hist, SrcImg);
	cv3.CalcGrayHist(ref_hist, RefImg);

	float src_cdf[256] = { 0 };
	float ref_cdf[256] = { 0 };

	// ���Ϲ��M�ؼйϹ����j�p���@�ˡA�n�N�o�쪺����϶i���k�@�ƳB�z
	src_hist /= (SrcImg.rows * SrcImg.cols);
	ref_hist /= (RefImg.rows * RefImg.cols);

	// �p���l����ϩM�W�w����Ϫ��ֿn���v
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


	// �ֿn���v���t��
	float diff_cdf[256][256];
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			diff_cdf[i][j] = fabs(src_cdf[i] - ref_cdf[j]);
		}
	}
	// �c�ئǫׯŬM�g��
	Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++)
	{
		// �d�䷽�ǫׯŬ��񪺬M�g�ǫ�
		//�@�M�񪺲ֿn���v�t�ȳ̤p���W�w�Ʀǫ�
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

	// ���άd���A������ϳW�w��
	LUT(SrcImg, lut, DstImg);
}
int main(int argc, char const *argv[]) {
	Mat srcimg = cv3.ImRead("11.jpg");
	Mat refimg = cv3.ImRead("22.jpg");
	//Mat gray = cv3.ImBGR2Gray(srcimg);
	//Mat noise = addSaltNoise(srcimg, 6000);
	//(2,2,CV_8UC3,Scalar::all(1));

	float box[9]      = { 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9 };
	float edgevertical[9] = { 1, 0, -1,2, 0, -2, 1, 0, -1 };
	float edgehorizontal[9] = { 1, 2, 1,0, 0, 0, -1, -2, -1 };
	Mat kmV = Mat(3, 3, CV_32FC1, edgevertical);
	Mat kmH = Mat(3, 3, CV_32FC1, edgehorizontal);
	Mat kernel;
	Mat imgV, imgH, dstimg, dstimg2, dstimg3;
	

	///processing
	
	cv3.EdgeDetect(dstimg, srcimg, SOBEL);
	cv3.EdgeDetect(dstimg2, refimg, SOBEL);
	//cv3.Smooth2D(dstimg,  srcimg,  3, GAUSSIAN, 1);
	//cv3.Smooth2D(dstimg2, srcimg, 3, GAUSSIAN, 0.1);
	//cv3.Smooth2D(dstimg3, srcimg, 3, GAUSSIAN, 100);
	
	//filter2D(gray, imgV, CV_16S, kmV);
	//convertScaleAbs(imgV, imgV);
	//filter2D(gray, imgH, CV_16S, kmH);
	//convertScaleAbs(imgH, imgH);
	//addWeighted(imgV, 0.5, imgH, 0.5, 0, dstimg);
	//cv3.EdgeDetect(dstimg2, gray);

	///show
	cv3.ImShow("dst2", dstimg2);
	cv3.ImShow("src", srcimg);
	cv3.ImShow("dst", dstimg);
	cv3.ImShow("ref", refimg);
	//cv3.ImWrite("image//dstimg3.jpg", dstimg3);
	//cv3.ImWrite("image//dstimg2.jpg", dstimg2);
	//cv3.ImWrite("image//dstimg.jpg", dstimg);
	//cv3.ImWrite("image//srcimg.jpg", srcimg);

	///�p��
	//double time0;
	//time0 = static_cast<double>(cv::getTickCount());
	//time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
	//std::cout << time0 << std::endl;

	///write
	//cv3.ImWrite("image//gray.jpg", gray);
	//cv3.ImWrite("image//img5.jpg", gray3);

	waitKey(0);
	return 0;
}