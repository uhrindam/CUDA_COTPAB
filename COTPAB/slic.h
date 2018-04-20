#ifndef SLIC_H
#define SLIC_H

#include <opencv2\opencv.hpp>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;
using namespace cv;

#define NR_ITERATIONS 10
#define numberOfNeighbors 8
#define maxColorDistance 39

class Slic {
private:
	vector<vector<int> > clusters;
	vector<vector<double> > distances;
	vector<vector<double> > centers;
	vector<int> center_counts;
	int *neighbors;

	int step;
	int nc;
	int ns;
	int rows;
	int cols;
	int centersRowPieces;
	int centersColPieces;

	double compute_dist(int ci, Point pixel, Vec3b colour);
	void clear_data();
	void init_data(Mat image);

public:
	Slic();
	~Slic();
	void generate_superpixels(Mat image, int step, int nc);
	void colour_with_cluster_means(Mat image);
	float colorDistance(Vec3b actuallPixel, Vec3b neighborPixel);
	void neighborMerge();
};
#endif