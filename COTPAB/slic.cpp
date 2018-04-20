#include "slic.h"

Slic::Slic() {}

Slic::~Slic() {
	clear_data();
}

void Slic::clear_data() {
	clusters.clear();
	distances.clear();
	centers.clear();
	center_counts.clear();
}

void Slic::init_data(Mat image) {
	rows = image.rows;
	cols = image.cols;

	/* Initialize the cluster and distance matrices. */
	for (int i = 0; i < cols; i++) {
		vector<int> cr;
		vector<double> dr;
		for (int j = 0; j < rows; j++) {
			cr.push_back(-1);
			dr.push_back(FLT_MAX);
		}
		clusters.push_back(cr);
		distances.push_back(dr);
	}

	centersColPieces = 0;
	centersRowPieces = 0;
	/* Initialize the centers and counters. */
	for (int i = step; i < cols - step / 2; i += step) {
		for (int j = step; j < rows - step / 2; j += step) {
			vector<double> center;
			/* Find the local minimum (gradient-wise). */
			Vec3b colour = image.at<Vec3b>(j, i);

			center.push_back(colour.val[0]);
			center.push_back(colour.val[1]);
			center.push_back(colour.val[2]);
			center.push_back(i);
			center.push_back(j);

			centers.push_back(center);
			center_counts.push_back(0);
		}
		centersColPieces++;
	}
	centersRowPieces = centers.size() / centersColPieces;

	neighbors = new int[centers.size() * numberOfNeighbors];
	for (int i = 0; i < centers.size(); i++)
	{
		for (int j = 0; j < numberOfNeighbors; j++)
		{
			neighbors[i*numberOfNeighbors + j] = -1;
		}
	}
}

double Slic::compute_dist(int ci, Point pixel, Vec3b colour) {
	//színtávolság
	double dc = sqrt(pow(centers[ci][0] - colour.val[0], 2) + pow(centers[ci][1]
		- colour.val[1], 2) + pow(centers[ci][2] - colour.val[2], 2));
	//euklideszi távolság
	double ds = sqrt(pow(centers[ci][3] - pixel.x, 2) + pow(centers[ci][4] - pixel.y, 2));

	return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));
}

void Slic::generate_superpixels(Mat image, int step, int nc) {
	this->step = step;
	this->nc = nc;
	this->ns = step;

	/* Clear previous data (if any), and re-initialize it. */
	clear_data();

	init_data(image);

	/* Run EM for 10 iterations (as prescribed by the algorithm). */
	for (int i = 0; i < NR_ITERATIONS; i++) {
		for (int j = 0; j < (int)centers.size(); j++) {
			/* Only compare to pixels in a 2 x step by 2 x step region.*/
			for (int k = centers[j][3] - step; k < centers[j][3] + step; k++) {
				for (int l = centers[j][4] - step; l < centers[j][4] + step; l++) {

					if (k >= 0 && k < image.cols && l >= 0 && l < image.rows) {
						Vec3b colour = image.at<Vec3b>(l, k);
						double d = compute_dist(j, Point(k, l), colour);

						/* Update cluster allocation if the cluster minimizes the
						distance. */
						if (d < distances[k][l]) {
							distances[k][l] = d;
							clusters[k][l] = j;
						}
					}
				}
			}
		}

		/* Clear the center values. */
		for (int j = 0; j < (int)centers.size(); j++) {
			centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
			center_counts[j] = 0;
		}

		/* Compute the new cluster centers. */
		for (int j = 0; j < image.cols; j++) {
			for (int k = 0; k < image.rows; k++) {
				int c_id = clusters[j][k];
				distances[j][k] = FLT_MAX;

				if (c_id != -1) {
					Vec3b colour = image.at<Vec3b>(k, j);

					centers[c_id][0] += colour.val[0];
					centers[c_id][1] += colour.val[1];
					centers[c_id][2] += colour.val[2];
					centers[c_id][3] += j;
					centers[c_id][4] += k;

					center_counts[c_id] += 1;
				}
			}
		}

		/* Normalize the clusters. */
		for (int j = 0; j < (int)centers.size(); j++) {
			centers[j][0] /= center_counts[j];
			centers[j][1] /= center_counts[j];
			centers[j][2] /= center_counts[j];
			centers[j][3] /= center_counts[j];
			centers[j][4] /= center_counts[j];
		}
	}
}

void Slic::colour_with_cluster_means(Mat image) {
	for (int i = 0; i < image.cols; i++) {
		for (int j = 0; j < image.rows; j++) {
			int idx = clusters[i][j];
			Vec3b ncolour = image.at<Vec3b>(j, i);

			ncolour.val[0] = centers[idx][0];
			ncolour.val[1] = centers[idx][1];
			ncolour.val[2] = centers[idx][2];

			image.at<Vec3b>(j, i) = ncolour;
		}
	}
}

float Slic::colorDistance(Vec3b actuallPixel, Vec3b neighborPixel)
{
	float dc = sqrt(pow(actuallPixel.val[0] - neighborPixel.val[0], 2) + pow(actuallPixel.val[1] - neighborPixel.val[1], 2)
		+ pow(actuallPixel.val[2] - neighborPixel.val[2], 2));
	return dc;
}

void Slic::neighborMerge()
{
	const int dx8[numberOfNeighbors] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[numberOfNeighbors] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	int *centersIn1D = new int[centers.size() * 5];
	for (int i = 0; i < centers.size(); i++)
	{
		centersIn1D[i * 5 + 0] = centers[i][0];
		centersIn1D[i * 5 + 1] = centers[i][1];
		centersIn1D[i * 5 + 2] = centers[i][2];
		centersIn1D[i * 5 + 4] = centers[i][3];
		centersIn1D[i * 5 + 5] = centers[i][4];
	}

	for (int i = 0; i < (int)centers.size(); i++)
	{
		Vec3b actuallCluster;
		actuallCluster.val[0] = centersIn1D[i * 5];
		actuallCluster.val[1] = centersIn1D[i * 5 + 1];
		actuallCluster.val[2] = centersIn1D[i * 5 + 2];

		int clusterRow = i / centersRowPieces;
		int clusterCol = i % centersRowPieces;

		for (int j = 0; j < numberOfNeighbors; j++)
		{
			if (clusterCol + dy8[j] >= 0 && clusterCol + dy8[j] < centersRowPieces
				&& clusterRow + dx8[j] >= 0 && clusterRow + dx8[j] < centersColPieces)
			{
				Vec3b neighborPixel;
				neighborPixel.val[0] = centersIn1D[(centersRowPieces* (clusterRow + dx8[j]) + (clusterCol + dy8[j])) * 5 + 0];
				neighborPixel.val[1] = centersIn1D[(centersRowPieces* (clusterRow + dx8[j]) + (clusterCol + dy8[j])) * 5 + 1];
				neighborPixel.val[2] = centersIn1D[(centersRowPieces* (clusterRow + dx8[j]) + (clusterCol + dy8[j])) * 5 + 2];

				if (centersRowPieces * clusterRow + clusterCol < centersRowPieces * (clusterRow + dx8[j]) + (clusterCol + dy8[j]) &&
					colorDistance(actuallCluster, neighborPixel) < maxColorDistance)
				{
					neighbors[(centersRowPieces * clusterRow + clusterCol) * numberOfNeighbors + j] = centersRowPieces * (clusterRow + dx8[j]) + (clusterCol + dy8[j]);
				}
			}
		}
	}

	vector<vector<int> > changes;
	for (int i = 0; i < (int)centers.size(); i++)
	{
		vector<int> change;
		change.push_back(i);
		change.push_back(-1);
		changes.push_back(change);
	}

	for (int i = 0; i < centers.size(); i++)
	{
		for (int j = 0; j < numberOfNeighbors; j++)
		{
			int cluster = neighbors[i * numberOfNeighbors + j];
			if (cluster != -1)
			{
				int neighborIDX = changes[cluster][1];
				int clusterIDX = i;
				while (neighborIDX != -1)
				{
					neighborIDX = changes[neighborIDX][1];
					if (neighborIDX != -1)
						clusterIDX = changes[neighborIDX][0];
				}
				if (changes[clusterIDX][1] != -1)
					changes[cluster][1] = changes[clusterIDX][1];
				else
					changes[cluster][1] = clusterIDX;
			}
		}
	}

	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			if (changes[clusters[i][j]][1] != -1)
			{
				clusters[i][j] = changes[clusters[i][j]][1];
			}
		}
	}
}