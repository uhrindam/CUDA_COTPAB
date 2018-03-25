
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2\opencv.hpp>
//#include <opencv2\cudaimgproc.hpp>

#include "slic.h"

using namespace std;
using namespace cv;

#define nc 80 //maximum vizsgált távolság a centroidok keresésekor
#define numberofSuperpixels 5000
#define iteration 10

float *ff;
vector<vector<float> > h_centers;

int cols;
int rows;
int step;
int centersLength;
int *clusters;
float *distances;
float *centers;
int *center_counts;
uchar3 *colors;

__device__ int *d_clusters;			//1D --> cols * rows
__device__ float *d_distances;		//1D --> cols * rows
__device__ float *d_centers;		//2D --> centersLength * 5
__device__ int *d_center_counts;	//1D --> centersLength
__device__ uchar3 *d_colors;		//1D --> cols * rows

__device__ float compute_dist(int ci, int x, int y, uchar3 colour, float *d_centers, int pitch, int d_step)
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
	int howManyPixels = d_cols*d_rows - 1;
	int clusterIDX = 0;

	for (int i = 0; i < iteration; i++)
	{
		/*if (threadIdx.x < howManyPixels)
			d_distances[threadIdx.x] = FLT_MAX;

		__syncthreads();*/

		//if (threadIdx.x >= howManyPixels)
		//if (threadIdx.x >= howManyPixels)
		//{
			//ha a szál id-je nagyobb mint a pixelek száma, akkor az egy cluster
			//szál, amely szálnak az indexe itt kerül inicializálásra
		//clusterIDX = threadIdx.x - howManyPixels;---------------------------------------------------------
		clusterIDX = blockIdx.x * blockDim.x + threadIdx.x;
		/* Only compare to pixels in a 2 x step by 2 x step region. ----------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!-------------------------- */

		/*int YE = d_centers[clusterIDX *pitch + 3] - d_step;
		int YV = d_centers[clusterIDX *pitch + 3] + d_step;
		int Y = d_centers[clusterIDX *pitch + 3];
		int XE = d_centers[clusterIDX *pitch + 4] - d_step;
		int X = d_centers[clusterIDX *pitch + 4];
		int XV = d_centers[clusterIDX *pitch + 4] + d_step;
		if (clusterIDX == 4500)
		{
			printf("%i\n", YE);
			printf("%i kozepe \n", Y);
			printf("%i\n", YV);
			printf("%i\n", XE);
			printf("%i kozepe \n", X);
			printf("%i\n", XV);
		}


		for (int pixelY = YE; pixelY < YV; pixelY++)
		{
			for (int pixelX = XE; pixelX < XV; pixelX++)
			{*/
		for (int pixelY = d_centers[clusterIDX *pitch + 3] - d_step; pixelY < d_centers[clusterIDX *pitch + 3] + d_step; pixelY++)
		{
			for (int pixelX = d_centers[clusterIDX *pitch + 4] - d_step; pixelX < d_centers[clusterIDX *pitch + 4] + d_step; pixelX++)
			{
				if (pixelY > 1000)
				{

				}
				if (pixelX >= 0 && pixelX < d_rows && pixelY >= 0 && pixelY < d_cols) {
					//float dc = sqrt(pow(d_centers[clusterIDX *pitch + 0] - d_colors[d_cols*pixelY + pixelX].x, 2) + 
					//	pow(d_centers[clusterIDX *pitch + 1] - d_colors[d_cols*pixelY + pixelX].y, 2)
					//	+ pow(d_centers[clusterIDX *pitch + 2] - d_colors[d_cols*pixelY + pixelX].z, 2));
					////euklideszi távolság
					//float ds = sqrt(pow(d_centers[clusterIDX *pitch + 3] - pixelX, 2) + pow(d_centers[clusterIDX *pitch + 4] - pixelY, 2));
					//float distance = sqrt(pow(dc / nc, 2) + pow(ds / d_step, 2));

					uchar3 colour = d_colors[d_cols*pixelY + pixelX];

					float distance = compute_dist(clusterIDX, pixelX, pixelY, colour, d_centers, pitch, d_step);
					if (distance < d_distances[d_cols*pixelY + pixelX])
					{
						d_distances[d_cols*pixelY + pixelX] = distance;
						d_clusters[d_cols*pixelY + pixelX] = clusterIDX;
					}
					//else
					//{
					//	if (clusterIDX == 4500)
					//	{
					//		float a = d_distances[d_cols*pixelY + pixelX];
					//		int c = d_clusters[d_cols*pixelY + pixelX];

					//		int cE = d_centers[clusterIDX *pitch + 0];
					//		int cM = d_centers[clusterIDX *pitch + 1];
					//		int cH = d_centers[clusterIDX *pitch + 2];
					//		int cN = d_centers[clusterIDX *pitch + 3];
					//		int cO = d_centers[clusterIDX *pitch + 4];

					//		int g = d_centers[clusterIDX *pitch + 5];
					//		int gg = d_centers[clusterIDX *pitch + 6];
					//		int ggg = d_centers[clusterIDX *pitch + 7];
					//		int gggg = d_centers[clusterIDX *pitch + 8];


					//		printf("rows: %i, cols: %i\n", d_rows, d_cols);
					//		printf("%i %i %i\n", blockIdx.x, blockDim.x, threadIdx.x);
					//		printf("%u %u %u %u %u pixel\n", (unsigned int)colour.x, (unsigned int)colour.y, (unsigned int)colour.z, (unsigned int)pixelX, (unsigned int)pixelY);
					//		printf("%u %u %u %u %u cluster\n", (unsigned int)cE, (unsigned int)cM, (unsigned int)cH, (unsigned int)cN, (unsigned int)cO);

					//		//printf("%u %u %u %u %u cl\n", (unsigned int)cO, (unsigned int)g, (unsigned int)gg, (unsigned int)ggg, (unsigned int)gggg);
					//		printf("%f jelenglegi\n", a);
					//		printf("%f uj\n", distance);
					//		printf("%i jelenglegi\n", c);
					//		printf("%i uj\n\n", clusterIDX);


					//		int kk = 0;
					//		//printf("hoszabb");
					//	}
					//}
				}
			}
		}

		//}
	}

	//d_distances[threadIdx.x] = compute_dist(885, threadIdx.x % d_rows, threadIdx.x / d_rows, d_colors[threadIdx.x], d_centers, pitch, d_step);
}

void initData(Mat image)
{
	clusters = new int[cols*rows];
	distances = new float[cols*rows];
	for (int i = 0; i < cols*rows; i++)
	{
		clusters[i] = -1;
		distances[i] = FLT_MAX;
	}

	//Ez azért kell mert elõre nem tudom, hogy hány eleme lesz a centers-nek, ezért elõször egy vectorhoz adomgatom hozzá az elemeket
	// majd késõbb létrehozom a tömböt annyi elemmel, ahány eleme van a segédvectornak, majd átmásolom az adatokat.
	//vector<vector<float> > h_centers;
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
	}

	centersLength = h_centers.size();

	centers = new float[centersLength * 5];
	center_counts = new int[centersLength];
	int idx = 0;
	for (int i = 0; i < centersLength; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			centers[idx] = h_centers[i][j];
			idx++;
		}
		if (i == 4500)
		{
			ff = new float[5];
			ff[0] = h_centers[i][0];
			ff[1] = h_centers[i][1];
			ff[2] = h_centers[i][2];
			ff[3] = h_centers[i][3];
			ff[4] = h_centers[i][4];
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
}

void dataFree()
{
	cudaFree(d_clusters);
	cudaFree(d_distances);
	cudaFree(d_centers);
	cudaFree(d_center_counts);
	cudaFree(d_colors);
}

int main()
{
	Mat image = imread("C:\\Users\\Adam\\Desktop\\samples\\completed.jpg", 1);
	cols = image.cols;
	rows = image.rows;

	step = (sqrt((cols * rows) / (double)numberofSuperpixels));

	initData(image);
	dataCopy();

	int pitchInt = 5;
	int startedThreads = rows*cols + centersLength - 1;
	int h = (centersLength / 5) + 1;
	compute << <5, h >> > (cols, rows, step, centersLength, d_clusters, d_distances, d_centers, d_center_counts, d_colors, pitchInt);

	cudaMemcpy(distances, d_distances, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, d_clusters, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);

	dataFree();

	int a = 0;

	for (int i = 0; i < rows*cols; i++)
	{
		if (clusters[i] == -1)
		{
			//cout << i << "x: " << i/cols << ", y: " << i % cols << endl;

			a++;
		}
	}

	//ofstream myfile;
	//myfile.open("example.txt");
	//for (int i = 0; i < centersLength; i++)
	//{
	//	myfile << i << " " << h_centers[i][0] << " " << h_centers[i][1] << " " << h_centers[i][2] << " " << h_centers[i][3] << " " << h_centers[i][4] <<  "\n";

	//}
	//myfile.close();

	printf("%u %u %u %u %u cluster\n\n", (unsigned int)ff[0], (unsigned int)ff[1],
		(unsigned int)ff[2], (unsigned int)ff[3], (unsigned int)ff[4]);

	int b = rows*cols - a;
	printf("\n%i\n", step);
	printf("%i rows\n", rows);
	printf("%i cols\n", cols);
	printf("%i darab pixel\n", rows*cols);
	printf("%i darab cluster\n", centersLength);
	printf("%i darab elinditott szal\n", h * 5);
	printf("%i darab clusterhez van renderve\n", b);
	printf("%i darab nincs clusterhez renderve\n\n", a);



	/*for (int i = 0; i < rows*cols; i++)
	{
		if (distances[i] != FLT_MAX)
			cout << distances[i] << endl;
	}*/

	printf("vege");
	///* Load the image and convert to Lab colour space. */
	//Mat image = imread("C:\\Users\\Adam\\Desktop\\samples\\completed.jpg", 1);
	//Mat lab_image = image.clone();
	//cvtColor(image, lab_image, CV_BGR2Lab);

	///* Yield the number of superpixels and weight-factors from the user. */
	//int w = image.cols;
	//int h = image.rows;
	//int nr_superpixels = 5000;
	//int nc = 80;

	//double step = (sqrt((w * h) / (double)nr_superpixels));
	////1400*900-as képnél, 1000 superpixellel --> 35,496 --> vízszintesen 39,444, függõlegesen 25,354

	///* Perform the SLIC superpixel algorithm. */
	//Slic slic;
	//slic.generate_superpixels(lab_image, step, nc);
	//slic.create_connectivity(lab_image);

	///* Display the contours and show the result. */
	//Mat tt = image.clone();
	//slic.display_contours(tt, Vec3b(0, 0, 255));
	//imwrite("C:\\Users\\Adam\\Desktop\\0MATsamplewitchLines.jpg", tt);

	////----------------------
	//slic.colour_with_cluster_means(image);
	//imwrite("C:\\Users\\Adam\\Desktop\\1MATsamplefilled.jpg", image);
	////----------------------

	getchar();
}
