
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2\opencv.hpp>


using namespace std;
using namespace cv;


int main()
{
	Mat image = imread("C:\\Users\\Adam\\Desktop\\samples\\hulk2.jpg", CV_LOAD_IMAGE_COLOR);
	imshow("Display window", image);
	waitKey();
}
