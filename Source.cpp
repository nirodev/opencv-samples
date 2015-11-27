#include < time.h>
#include < opencv2\opencv.hpp>
#include < opencv2\gpu\gpu.hpp>
#include < opencv2\legacy\legacy.hpp>
#include < string>
#include < stdio.h>




#define RWIDTH 320 //640 //800
#define RHEIGHT 240 //480 //600

#define RWIDTH2 800
#define RHEIGHT2 600

#define MIN_BLOB 10
#define MAX_BLOB 500

using namespace std;
using namespace cv;

int main()
{
	/////////////////////////////////////////////////////////////////////////
	//MOG routine
	gpu::MOG2_GPU pMOG2_g(30);
	pMOG2_g.history = 3000; //300;
	pMOG2_g.varThreshold = 128; //64; //128; //64; //32;//; 
	pMOG2_g.bShadowDetection = false; // true;//
	Mat Mog_Mask;
	gpu::GpuMat Mog_Mask_g, Mog_MaskMorpho_g;
	/////////////////////////////////////////////////////////////////////////


	//Load avi file
	VideoCapture cap("C:\\Users\\niro273\\Desktop\\Videos\\PRG6.avi");
	/////////////////////////////////////////////////////////////////////////

	//Mat and GpuMat
	Mat o_frame;
	gpu::GpuMat o_frame_gpu;
	gpu::GpuMat r_frame_gpu;
	gpu::GpuMat r_frame2_gpu;
	gpu::GpuMat rg_frame2_gpu;
	gpu::GpuMat r_binary_gpu;
	gpu::GpuMat r_blur_gpu;
	gpu::GpuMat r_binary2_gpu;

	//Mopology
	Mat element;
	element = getStructuringElement(MORPH_RECT, Size(9, 9), Point(4, 4));
	/////////////////////////////////////////////////////////////////////////

	//capture
	cap >> o_frame;
	if (o_frame.empty())
		return 0;

	//for 3channel blur
	vector< gpu::GpuMat> gpurgb(3);
	vector< gpu::GpuMat> gpurgb2(3);
	/////////////////////////////////////////////////////////////////////////

	//scale and check processing time
	unsigned long AAtime = 0, BBtime = 0;
	float scaleX = float(o_frame.size().width) / RWIDTH;
	float scaleY = float(o_frame.size().height) / RHEIGHT;
	float scaleX2 = RWIDTH2 / float(RWIDTH);
	float scaleY2 = RHEIGHT2 / float(RHEIGHT);
	//////////////////////////////////////////////////////////////////////////

	//for fREAK
	//gpu::GpuMat keypoints1GPU;
	vector< KeyPoint > keypoints;
	Mat descriptorsA;
	gpu::FAST_GPU fast(25);
	FREAK extractor;
	//////////////////////////////////////////////////////////////////////////


	while (1)
	{
		/////////////////////////////////////////////////////////////////////////
		cap >> o_frame;
		if (o_frame.empty())
			return 0;

		//get frame and upload to gpu
		o_frame_gpu.upload(o_frame);
		//resize
		gpu::resize(o_frame_gpu, r_frame_gpu, Size(RWIDTH, RHEIGHT)); //for blob labeling
		//gpu::resize(o_frame_gpu, r_frame2_gpu, Size(RWIDTH2, RHEIGHT2)); //for FREAK featrue 
		AAtime = getTickCount();
		//blur
		gpu::split(r_frame_gpu, gpurgb);
		gpu::blur(gpurgb[0], gpurgb2[0], Size(3, 3));
		gpu::blur(gpurgb[1], gpurgb2[1], Size(3, 3));
		gpu::blur(gpurgb[2], gpurgb2[2], Size(3, 3));
		gpu::merge(gpurgb2, r_blur_gpu);
		//mog
		pMOG2_g.operator()(r_blur_gpu, Mog_Mask_g, -1);
		//mopnology
		//gpu::morphologyEx(Mog_Mask_g, Mog_MaskMorpho_g, CV_MOP_CLOSE, element);
		gpu::morphologyEx(Mog_Mask_g, Mog_MaskMorpho_g, CV_MOP_DILATE, element);
		//binary
		gpu::threshold(Mog_MaskMorpho_g, r_binary_gpu, 128, 255, CV_THRESH_BINARY);
		//Blob Labeling
		//Find contour   
		Mat ContourImg;
		r_binary_gpu.download(ContourImg);
		//less blob delete   
		vector< vector< Point> > contours;
		findContours(ContourImg,
			contours, // a vector of contours   
			CV_RETR_EXTERNAL, // retrieve the external contours   
			CV_CHAIN_APPROX_NONE); // all pixels of each contours  

		vector< Rect > output;
		vector< vector< Point> >::iterator itc = contours.begin();

		///Display  
		Mat showMat2_r;
		r_frame2_gpu.download(showMat2_r);
		//fast feature
		//gpu::cvtColor(r_frame2_gpu, rg_frame2_gpu, CV_RGB2GRAY);
		//fast(rg_frame2_gpu, gpu::GpuMat(), keypoints);
		//for (unsigned int i = 0; i < keypoints.size(); i++)
		//{
		//	const KeyPoint& kp = keypoints[i];
			//circle draw
		//	circle(showMat2_r, Point(kp.pt.x, kp.pt.y), 10, Scalar(255, 0, 0, 255));
		//}


		//FREAK descriptor 
		//Mat rg_frame2;
		//rg_frame2_gpu.download(rg_frame2);
		//extractor.compute(rg_frame2, keypoints, descriptorsA);


		//Blob labeling
		while (itc != contours.end()) {

			//Create bounding rect of object   
			//rect draw on origin image   
			Rect mr = boundingRect(Mat(*itc));

			mr.x = mr.x * scaleX2;
			mr.y = mr.y * scaleY2;
			mr.width = mr.width * scaleX2;
			mr.height = mr.height * scaleY2;

			rectangle(showMat2_r, mr, CV_RGB(255, 0, 0));
			++itc;
		}

		//processing time print
		BBtime = getTickCount();
		float pt = (BBtime - AAtime) / getTickFrequency();
		float fpt = 1 / pt;
		printf("gpu %.4lf / %.4lf \n", pt, fpt);

		//Display 
		Mat showMat_r_blur;
		Mat showBinary;
		r_binary_gpu.download(showBinary);
		imshow("binary2", showBinary);
		Mog_Mask_g.download(Mog_Mask);
		r_blur_gpu.download(showMat_r_blur);
		imshow("blur", showMat_r_blur);
		imshow("origin", showMat2_r);
		imshow("mog_mask", Mog_Mask);
		/////////////////////////////////////////////////////////////////////////

		if (waitKey(10) > 0)
			break;
	}

	return 0;
}



