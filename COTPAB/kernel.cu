
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2\opencv.hpp>
#include "slic.h"

using namespace std;
using namespace cv;


int main()
{
	/* Load the image and convert to Lab colour space. */
	Mat image = imread("C:\\Users\\Adam\\Desktop\\samples\\completed.jpg", 1);
	Mat lab_image = image.clone();
	cvtColor(image, lab_image, CV_BGR2Lab);

	/* Yield the number of superpixels and weight-factors from the user. */
	int w = image.cols;
	int h = image.rows;
	int nr_superpixels = 5000;
	int nc = 80;

	double step = (sqrt((w * h) / (double)nr_superpixels));
	//1400*900-as képnél, 1000 superpixellel --> 35,496 --> vízszintesen 39,444, függõlegesen 25,354

	/* Perform the SLIC superpixel algorithm. */
	Slic slic;
	slic.generate_superpixels(lab_image, step, nc);
	slic.create_connectivity(lab_image);

	/* Display the contours and show the result. */
	Mat tt = image.clone();
	slic.display_contours(tt, Vec3b(0, 0, 255));
	imwrite("C:\\Users\\Adam\\Desktop\\0MATsamplewitchLines.jpg", tt);

	//----------------------
	slic.colour_with_cluster_means(image);
	imwrite("C:\\Users\\Adam\\Desktop\\1MATsamplefilled.jpg", image);
	//----------------------

	cvWaitKey(0)
}
