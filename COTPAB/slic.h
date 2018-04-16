#ifndef SLIC_H
#define SLIC_H

/* slic.h.
*
* Written by: Pascal Mettes.
*
* This file contains the class elements of the class Slic. This class is an
* implementation of the SLIC Superpixel algorithm by Achanta et al. [PAMI'12,
* vol. 34, num. 11, pp. 2274-2282].
*
* This implementation is created for the specific purpose of creating
* over-segmentations in an OpenCV-based environment.
*/

#include <opencv2\opencv.hpp>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;
using namespace cv;

/* The number of iterations run by the clustering algorithm. */
#define NR_ITERATIONS 10
#define numberOfNeighbors 8
#define maxColorDistance 39//15


/*
* class Slic.
*
* In this class, an over-segmentation is created of an image, provided by the
* step-size (distance between initial cluster locations) and the colour
* distance parameter.
*/
class Slic {
private:
	/* The cluster assignments and distance values for each pixel. */
	vector<vector<int> > clusters;
	vector<vector<double> > distances;

	/* The LAB and xy values of the centers. */
	vector<vector<double> > centers;
	int *neighbors; 
	/* The number of occurences of each center. */
	vector<int> center_counts;

	/* The step size per cluster, and the colour (nc) and distance (ns)
	* parameters. */
	int step, nc, ns;

	int rows;
	int cols;

	int centersRowPieces;
	int centersColPieces;

	/* Compute the distance between a center and an individual pixel. */
	double compute_dist(int ci, Point pixel, Vec3b colour);
	/* Find the pixel with the lowest gradient in a 3x3 surrounding. */
	Point find_local_minimum(Mat image, Point center);

	/* Remove and initialize the 2d vectors. */
	void clear_data();
	void init_data(Mat image);

public:
	/* Class constructors and deconstructors. */
	Slic();
	~Slic();

	/* Generate an over-segmentation for an image. */
	void generate_superpixels(Mat image, int step, int nc);
	/* Enforce connectivity for an image. */
	void create_connectivity(Mat image);

	/* Draw functions. Resp. displayal of the centers and the contours. */
	void display_contours(Mat image, Vec3b colour);
	void colour_with_cluster_means(Mat image);
	float colorDistance(Vec3b actuallPixel, Vec3b neighborPixel);
	void neighborMerge();
};

#endif
