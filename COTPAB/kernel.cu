#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2\opencv.hpp>
#include "slic.h"

using namespace std;
using namespace cv;

#define nc 80 //maximum vizsgált távolság a centroidok keresésekor
#define numberofSuperpixels 4500
#define iteration 10
#define maxColorDistance 39//15
#define numberOfNeighbors 8

int cols;
int rows;
int step;
int centersLength;
int centersRowPieces;
int centersColPieces;
int *clusters;
float *distances;
float *centers;
int *center_counts;
uchar3 *colors;
float *pixelColorsWithClusterMeanColor;
int *neighbors;

__device__ int *d_clusters;								//cols * rows
__device__ float *d_distances;							//cols * rows
__device__ float *d_centers;							//centersLength * 5
__device__ int *d_center_counts;						//centersLength
__device__ uchar3 *d_colors;							//cols * rows
__device__ float * d_pixelColorsWithClusterMeanColor;	//cols * rows * 3
__device__ int *d_neighbors;								//centerlength * 8

__device__ float compute_dist(int ci, int y, int x, uchar3 colour, float *d_centers, int pitch, int d_step)
{
	//színtávolság
	float dc = sqrt(pow(d_centers[ci *pitch + 0] - colour.x, 2) + pow(d_centers[ci *pitch + 1] - colour.y, 2)
		+ pow(d_centers[ci *pitch + 2] - colour.z, 2));
	//euklideszi távolság
	float ds = sqrt(pow(d_centers[ci *pitch + 3] - x, 2) + pow(d_centers[ci *pitch + 4] - y, 2));

	return sqrt(pow(dc / nc, 2) + pow(ds / d_step, 2));
}

__global__ void compute(int d_cols, int d_rows, int d_step, int d_centersLength, int *d_clusters, float *d_distances,
	float *d_centers, int *d_center_counts, uchar3 *d_colors, int pitch)
{
	int clusterIDX = blockIdx.x * blockDim.x + threadIdx.x;

	if (clusterIDX < d_centersLength)
	{
		for (int pixelY = d_centers[clusterIDX *pitch + 3] - (d_step*1.5); pixelY < d_centers[clusterIDX *pitch + 3] + (d_step*1.5); pixelY++)
		{
			for (int pixelX = d_centers[clusterIDX *pitch + 4] - (d_step*1.5); pixelX < d_centers[clusterIDX *pitch + 4] + (d_step*1.5); pixelX++)
			{
		//for (int pixelY = d_centers[clusterIDX *pitch + 3] - (d_step); pixelY < d_centers[clusterIDX *pitch + 3] + (d_step); pixelY++)
		//{
		//	for (int pixelX = d_centers[clusterIDX *pitch + 4] - (d_step); pixelX < d_centers[clusterIDX *pitch + 4] + (d_step); pixelX++)
		//	{
				if (pixelX >= 0 && pixelX < d_rows && pixelY >= 0 && pixelY < d_cols)
				{
					uchar3 colour = d_colors[d_rows*pixelY + pixelX];
					float distance = compute_dist(clusterIDX, pixelX, pixelY, colour, d_centers, pitch, d_step);
					if (distance < d_distances[d_rows*pixelY + pixelX])
					{
						d_distances[d_rows*pixelY + pixelX] = distance;
						d_clusters[d_rows*pixelY + pixelX] = clusterIDX;
					}
				}
			}
		}
		//a centroidok alaphelyzetbe állítása
		d_centers[clusterIDX *pitch + 0] = 0;
		d_centers[clusterIDX *pitch + 1] = 0;
		d_centers[clusterIDX *pitch + 2] = 0;
		d_centers[clusterIDX *pitch + 3] = 0;
		d_centers[clusterIDX *pitch + 4] = 0;
		d_center_counts[clusterIDX] = 0;
	}

}

__global__ void compute1(int d_cols, int d_rows, int *d_clusters, float *d_distances,
	float *d_centers, int *d_center_counts, uchar3 *d_colors, int pitch)
{
	int idIn1D = blockIdx.x * blockDim.x + threadIdx.x;
	if (idIn1D < d_cols*d_rows)
	{
		d_distances[idIn1D] = FLT_MAX;

		int whichCluster = d_clusters[idIn1D];
		atomicAdd(&d_centers[whichCluster*pitch + 0], d_colors[idIn1D].x);
		atomicAdd(&d_centers[whichCluster*pitch + 1], d_colors[idIn1D].y);
		atomicAdd(&d_centers[whichCluster*pitch + 2], d_colors[idIn1D].z);
		atomicAdd(&d_centers[whichCluster*pitch + 3], idIn1D / d_rows);
		atomicAdd(&d_centers[whichCluster*pitch + 4], idIn1D % d_rows);

		atomicAdd(&d_center_counts[whichCluster], 1);
	}
}

__global__ void compute2(int d_centersLength, float *d_centers, int *d_center_counts, int pitch)
{
	int idIn1D = blockIdx.x * blockDim.x + threadIdx.x;
	if (idIn1D < d_centersLength)
	{
		d_centers[idIn1D*pitch + 0] = (int)(d_centers[idIn1D*pitch + 0] / d_center_counts[idIn1D]);
		d_centers[idIn1D*pitch + 1] = (int)(d_centers[idIn1D*pitch + 1] / d_center_counts[idIn1D]);
		d_centers[idIn1D*pitch + 2] = (int)(d_centers[idIn1D*pitch + 2] / d_center_counts[idIn1D]);
		d_centers[idIn1D*pitch + 3] = (int)(d_centers[idIn1D*pitch + 3] / d_center_counts[idIn1D]);
		d_centers[idIn1D*pitch + 4] = (int)(d_centers[idIn1D*pitch + 4] / d_center_counts[idIn1D]);
	}
}

__global__ void compute3(int d_cols, int d_rows, int *d_clusters, float *d_pixelColorsWithClusterMeanColor, float*d_centers)
{
	int idIn1D = blockIdx.x * blockDim.x + threadIdx.x;
	if (idIn1D < d_cols*d_rows)
	{
		d_pixelColorsWithClusterMeanColor[idIn1D * 3 + 0] = d_centers[d_clusters[idIn1D] * 5 + 0];
		d_pixelColorsWithClusterMeanColor[idIn1D * 3 + 1] = d_centers[d_clusters[idIn1D] * 5 + 1];
		d_pixelColorsWithClusterMeanColor[idIn1D * 3 + 2] = d_centers[d_clusters[idIn1D] * 5 + 2];
	}
}

float colorDistance(uchar3 actuallPixel, uchar3 neighborPixel)
{
	float dc = sqrt(pow(actuallPixel.x - neighborPixel.x, 2) + pow(actuallPixel.y - neighborPixel.y, 2)
		+ pow(actuallPixel.z - neighborPixel.z, 2));
	return dc;
}

void neighborMerge()
{
	const int dx8[numberOfNeighbors] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[numberOfNeighbors] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	for (int i = 0; i < centersLength; i++)
	{
		uchar3 actuallCluster;
		actuallCluster.x = centers[i * 5];
		actuallCluster.y = centers[i * 5 + 1];
		actuallCluster.z = centers[i * 5 + 2];

		int clusterRow = i / centersRowPieces;
		int clusterCol = i % centersRowPieces;

		for (int j = 0; j < numberOfNeighbors; j++)
		{
			if (clusterCol + dy8[j] >= 0 && clusterCol + dy8[j] < centersColPieces
				&& clusterRow + dx8[j] >= 0 && clusterRow + dx8[j] < centersRowPieces)
			{
				uchar3 neighborPixel;
				neighborPixel.x = centers[(centersRowPieces* (clusterRow + dx8[j]) + (clusterCol + dy8[j])) * 5 + 0];
				neighborPixel.y = centers[(centersRowPieces* (clusterRow + dx8[j]) + (clusterCol + dy8[j])) * 5 + 1];
				neighborPixel.z = centers[(centersRowPieces* (clusterRow + dx8[j]) + (clusterCol + dy8[j])) * 5 + 2];

				if (centersRowPieces * clusterRow + clusterCol < centersRowPieces * (clusterRow + dx8[j]) + (clusterCol + dy8[j]) &&
					colorDistance(actuallCluster, neighborPixel) < maxColorDistance)
				{
					neighbors[(centersRowPieces * clusterRow + clusterCol) * numberOfNeighbors + j] = centersRowPieces * (clusterRow + dx8[j]) + (clusterCol + dy8[j]);
				}
			}
		}
	}

	int2 *changes = new int2[centersLength];
	for (int i = 0; i < centersLength; i++)
	{
		changes[i].x = i;
		changes[i].y = -1;
	}

	for (int i = 0; i < centersLength; i++)
	{
		for (int j = 0; j < numberOfNeighbors; j++)
		{
			int cluster = neighbors[i * numberOfNeighbors + j];
			if (cluster != -1)
			{
				int neighborIDX = changes[cluster].y;
				int clusterIDX = i;
				while (neighborIDX != -1)
				{
					neighborIDX = changes[neighborIDX].y;
					if (neighborIDX != -1)
						clusterIDX = changes[neighborIDX].x;
				}
				if (changes[clusterIDX].y != -1)
					changes[cluster].y = changes[clusterIDX].y;
				else
					changes[cluster].y = clusterIDX;
			}
		}
	}

	for (int i = 0; i < cols*rows; i++)
	{
		if (changes[clusters[i]].y != -1)
		{
			clusters[i] = changes[clusters[i]].y;
		}
	}

	//for (int i = 0; i < centersLength; i++)
	//{
	//	cout << changes[i].x << "\t" << changes[i].y << "\t" << neighbors[i * 8 + 0] << "\t" << neighbors[i * 8 + 1] << "\t" << neighbors[i * 8 + 2] << "\t"
	//		<< neighbors[i * 8 + 3] << "\t" << neighbors[i * 8 + 4] << "\t" << neighbors[i * 8 + 5] << "\t" << neighbors[i * 8 + 6]	<< "\t" << neighbors[i * 8 + 7] << endl;
	//}
	//nem lesz centerslenght méretû, de ebben elfér minden
	//int2 *changes= new int2[centersLength];
	//int changesIDX = 0;
	//for (int i = 0; i < centersLength; i++)
	//{
	//	for (int j = 0; j < 8; j++)
	//	{
	//		int neighborIDX = neighbors[i * 8 + j];
	//		if (neighborIDX != -1)
	//		{
	//			int2 actualChange;
	//			actualChange.x = i;
	//			actualChange.y = neighborIDX;
	//			changes[changesIDX++] = actualChange;
	//			if (j < 4)
	//				neighbors[neighborIDX * 8 + j + 4] = -1;
	//			else
	//				neighbors[neighborIDX * 8 + j - 4] = -1;
	//		}
	//	}
	//}
	//for (int i = 0; i < changesIDX; i++)
	//{
	//}
}

void initData(Mat image)
{
	clusters = new int[cols*rows];
	distances = new float[cols*rows];
	pixelColorsWithClusterMeanColor = new float[cols*rows * 3];
	for (int i = 0; i < cols*rows; i++)
	{
		clusters[i] = -1;
		distances[i] = FLT_MAX;
		pixelColorsWithClusterMeanColor[i * 3 + 0] = 0;
		pixelColorsWithClusterMeanColor[i * 3 + 1] = 0;
		pixelColorsWithClusterMeanColor[i * 3 + 2] = 0;
	}

	//Ez azért kell mert elõre nem tudom, hogy hány eleme lesz a centers-nek, ezért elõször egy vectorhoz adogatom hozzá az elemeket
	// majd késõbb létrehozom a tömböt annyi elemmel, ahány eleme van a segédvectornak, majd átmásolom az adatokat.
	centersColPieces = 0;
	centersRowPieces = 0;
	vector<vector<float> > h_centers;
	for (int i = step; i < cols - step / 2; i += step) {
		for (int j = step; j < rows - step / 2; j += step) {
			vector<float> center;
			/* Find the local minimum (gradient-wise). */
			//Point nc = find_local_minimum(image, Point(i, j));
			Vec3b colour = image.at<Vec3b>(j, i);//nc.y, nc.x);

			center.push_back(colour.val[0]);
			center.push_back(colour.val[1]);
			center.push_back(colour.val[2]);
			center.push_back(i);//nc.x);
			center.push_back(j);//nc.y);

			h_centers.push_back(center);
		}
		centersColPieces++;
	}

	centersLength = h_centers.size();
	centersRowPieces = centersLength / centersColPieces;

	centers = new float[centersLength * 5];
	center_counts = new int[centersLength];
	neighbors = new int[centersLength * numberOfNeighbors];

	int idx = 0;
	for (int i = 0; i < centersLength; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			centers[idx] = h_centers[i][j];
			idx++;
		}
		for (int j = 0; j < numberOfNeighbors; j++)
		{
			neighbors[i * numberOfNeighbors + j] = -1;
		}
		center_counts[i] = 0;
	}

	//Bejárom a képet, majd minden pixel színét (3 érték) elmentem egy uchar3 változóba
	colors = new uchar3[rows*cols];
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			Vec3b colour = image.at<Vec3b>(j, i);
			colors[i * rows + j] = make_uchar3(colour.val[0], colour.val[1], colour.val[2]);
		}
	}
}

void dataCopy()
{
	cudaMalloc((void**)&d_clusters, sizeof(int)*rows*cols);
	cudaMemcpy(d_clusters, clusters, sizeof(int)*rows*cols, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_distances, sizeof(float)*rows*cols);
	cudaMemcpy(d_distances, distances, sizeof(float)*rows*cols, cudaMemcpyHostToDevice);

	//size_t pitch = 5;
	//cudaMallocPitch((void**)&d_centers, &pitch, sizeof(float) * centersLength, 5);
	//cudaMemcpy2D(d_centers, pitch, centers, sizeof(float) * centersLength, sizeof(float) * centersLength, 5, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_centers, sizeof(float)*centersLength * 5);
	cudaMemcpy(d_centers, centers, sizeof(float)*centersLength * 5, cudaMemcpyHostToDevice);


	cudaMalloc((void**)&d_center_counts, sizeof(int)*centersLength);
	cudaMemcpy(d_center_counts, center_counts, sizeof(int)*centersLength, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_colors, sizeof(uchar3)*rows*cols);
	cudaMemcpy(d_colors, colors, sizeof(uchar3)*rows*cols, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_pixelColorsWithClusterMeanColor, sizeof(float)*cols * rows * 3);
	cudaMemcpy(d_pixelColorsWithClusterMeanColor, pixelColorsWithClusterMeanColor, sizeof(float)*cols * rows * 3, cudaMemcpyHostToDevice);
}

void dataFree()
{
	cudaFree(d_clusters);
	cudaFree(d_distances);
	cudaFree(d_centers);
	cudaFree(d_center_counts);
	cudaFree(d_colors);
	cudaFree(d_pixelColorsWithClusterMeanColor);
}

void colour_with_cluster_means(Mat image) {
	cout << "FILL" << endl;

	for (int i = 0; i < image.cols; i++)
	{
		for (int j = 0; j < image.rows; j++)
		{
			Vec3b ncolour = Vec3b();

			ncolour.val[0] = pixelColorsWithClusterMeanColor[(i*image.rows + j) * 3 + 0];
			ncolour.val[1] = pixelColorsWithClusterMeanColor[(i*image.rows + j) * 3 + 1];
			ncolour.val[2] = pixelColorsWithClusterMeanColor[(i*image.rows + j) * 3 + 2];

			image.at<Vec3b>(j, i) = ncolour;
		}
	}
}

void display_contours(Mat image, Vec3b colour) {
	cout << "Display contours" << endl;

	const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel
	* is already taken to be a contour. */
	vector<Point> contours;
	vector<vector<bool> > istaken;
	for (int i = 0; i < image.cols; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.rows; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}

	/* Go through all the pixels. */
	for (int i = 0; i < image.cols; i++) {
		for (int j = 0; j < image.rows; j++) {
			int nr_p = 0;

			/* Compare the pixel to its 8 neighbours. */
			for (int k = 0; k < 8; k++) {
				int x = i + dx8[k], y = j + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[x][y] == false && clusters[i*image.rows + j] != clusters[x*image.rows + y]) {
						nr_p += 1;
					}
				}
			}

			/* Add the pixel to the contour list if desired. */
			if (nr_p >= 2) {
				contours.push_back(Point(i, j));
				istaken[i][j] = true;
			}
		}
	}

	/* Draw the contour pixels. */
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<Vec3b>(contours[i].y, contours[i].x) = colour;
	}
}

int main(int ArgsC, char* Args[])
{
	string readPath;
	string writePath;
	string isGPUProcess = "gpu";
	if (ArgsC < 2)
	{
		readPath = "C:\\Users\\Adam\\Desktop\\samples\\hulk1.jpg";
		writePath = "C:\\Users\\Adam\\Desktop\\xmen.jpg";
		if (Args[3] == "cpu")
			isGPUProcess = "cpu";
	}
	else
	{
		readPath = Args[1];
		writePath = Args[2];
	}
	Mat image = imread(readPath, 1);
	cols = image.cols;
	rows = image.rows;

	step = (sqrt((cols * rows) / (double)numberofSuperpixels));
	if (isGPUProcess == "gpu")
	{
		initData(image);

		int howManyBlocks = centersLength / 700;
		int threadsPerBlock = (centersLength / howManyBlocks) + 1;

		int threadsToBeStarted2 = rows*cols;
		int howManyBlocks2 = threadsToBeStarted2 / 700;
		int threadsPerBlock2 = (threadsToBeStarted2 / howManyBlocks2) + 1;
		for (int i = 0; i < iteration; i++)
		{
			dataCopy();
			compute << <howManyBlocks, threadsPerBlock >> > (cols, rows, step, centersLength, d_clusters, d_distances, d_centers, d_center_counts, d_colors, 5);
			compute1 << <howManyBlocks2, threadsPerBlock2 >> > (cols, rows, d_clusters, d_distances, d_centers, d_center_counts, d_colors, 5);
			compute2 << <howManyBlocks, threadsPerBlock >> > (centersLength, d_centers, d_center_counts, 5);

			cudaMemcpy(distances, d_distances, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
			cudaMemcpy(clusters, d_clusters, sizeof(int)*rows*cols, cudaMemcpyDeviceToHost);
			cudaMemcpy(centers, d_centers, sizeof(int)*centersLength * 5, cudaMemcpyDeviceToHost);
			cudaMemcpy(center_counts, d_center_counts, sizeof(int)*centersLength, cudaMemcpyDeviceToHost);

			dataFree();
		}

		neighborMerge();

		dataCopy();

		compute3 << <howManyBlocks2, threadsPerBlock2 >> > (cols, rows, d_clusters, d_pixelColorsWithClusterMeanColor, d_centers);

		cudaMemcpy(pixelColorsWithClusterMeanColor, d_pixelColorsWithClusterMeanColor, sizeof(float)*rows*cols * 3, cudaMemcpyDeviceToHost);

		dataFree();

		//ofstream myfile2;
		//myfile2.open("2.txt");
		//for (int i = 0; i < centersLength; i++)
		//{
		//	myfile2 << centers[i * 5 + 0] << " " << centers[i * 5 + 1] << " " << centers[i * 5 + 2] << " " << endl;
		//}
		//myfile2.close();

		int a = 0;
		for (int i = 0; i < rows*cols; i++) { if (clusters[i] == -1) { a++; } }
		int b = rows*cols - a;

		printf("%i elinditott szal\n", threadsPerBlock2*howManyBlocks2);
		printf("%i steps\n", step);
		printf("%i rows\n", rows);
		printf("%i cols\n", cols);
		printf("%i darab cluster\n", centersLength);
		printf("%i darab pixel\n", rows*cols);
		printf("%i darab elinditott szal\n", threadsPerBlock*howManyBlocks);
		printf("%i darab clusterhez van renderve\n", b);
		printf("%i darab nincs clusterhez renderve\n", a);

		int dis = 0;
		for (int i = 0; i < rows*cols; i++) { if (distances[i] == FLT_MAX) { dis++; } }
		printf("%i dis\n", dis);


		int mennyi = 0;
		for (int i = 0; i < centersLength; i++) { mennyi += center_counts[i]; }
		printf("%i darab pixel\n", rows*cols);
		printf("%i mennyi\n", mennyi);

		Mat cwtm = image.clone();
		colour_with_cluster_means(cwtm);
		imwrite(writePath, cwtm);
	}
}
