
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
__device__ float *d_centers;		//1D --> centersLength * 5
__device__ int *d_center_counts;	//1D --> centersLength
__device__ uchar3 *d_colors;		//1D --> cols * rows

__device__ float compute_dist(int ci, int y, int x, uchar3 colour, float *d_centers, int pitch, int d_step)
{
	//színtávolság
	float dc = sqrt(pow(d_centers[ci *pitch + 0] - colour.x, 2) + pow(d_centers[ci *pitch + 1] - colour.y, 2)
		+ pow(d_centers[ci *pitch + 2] - colour.z, 2));
	//euklideszi távolság
	float ds = sqrt(pow(d_centers[ci *pitch + 3] - x, 2) + pow(d_centers[ci *pitch + 4] - y, 2));

	return sqrt(pow(dc / nc, 2) + pow(ds / d_step, 2));
}

__device__ void compute0(int clusterIDX, int d_cols, int d_rows, int d_step, int d_centersLength, int *d_clusters, float *d_distances,
	float *d_centers, int *d_center_counts, uchar3 *d_colors, int pitch)
{
	for (int pixelY = d_centers[clusterIDX *pitch + 3] - (d_step*1.5); pixelY < d_centers[clusterIDX *pitch + 3] + (d_step*1.5); pixelY++)
	{
		for (int pixelX = d_centers[clusterIDX *pitch + 4] - (d_step*1.5); pixelX < d_centers[clusterIDX *pitch + 4] + (d_step*1.5); pixelX++)
		{

			if (pixelX >= 0 && pixelX < d_rows && pixelY >= 0 && pixelY < d_cols) {

				uchar3 colour = d_colors[d_cols*pixelX + pixelY];

				float distance = compute_dist(clusterIDX, pixelX, pixelY, colour, d_centers, pitch, d_step);
				if (distance < d_distances[d_cols*pixelX + pixelY])
				{
					d_distances[d_cols*pixelX + pixelY] = distance;
					d_clusters[d_cols*pixelX + pixelY] = clusterIDX;
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

__device__ void compute1(int idIn1D, int d_cols, float *d_centers, int *d_center_counts, uchar3 *d_colors, int pitch)
{
		d_distances[idIn1D] = FLT_MAX;
		
		/*int whichCluster = d_clusters[idIn1D];
		d_centers[whichCluster*pitch + 0] += d_colors[idIn1D].x;
		d_centers[whichCluster*pitch + 1] += d_colors[idIn1D].y;
		d_centers[whichCluster*pitch + 2] += d_colors[idIn1D].z;
		d_centers[whichCluster*pitch + 3] += idIn1D / d_cols;
		d_centers[whichCluster*pitch + 4] += idIn1D % d_cols;

		atomicAdd(&d_center_counts[whichCluster], 1);*/
}


__global__ void compute(int d_cols, int d_rows, int d_step, int d_centersLength, int *d_clusters, float *d_distances,
	float *d_centers, int *d_center_counts, uchar3 *d_colors, int pitch)
{
	int howManyPixels = d_cols*d_rows - 1;
	int idIn1D = blockIdx.x * blockDim.x + threadIdx.x;
	//ha a szál id-je nagyobb mint a pixelek száma, akkor az egy cluster
	//szál, amely szálnak az indexe itt kerül inicializálásra
	int clusterIDX = idIn1D - howManyPixels;

	//for (int i = 0; i < iteration; i++)
	//{
		if (idIn1D > howManyPixels)
		{
			for (int pixelY = d_centers[clusterIDX *pitch + 3] - (d_step*1.5); pixelY < d_centers[clusterIDX *pitch + 3] + (d_step*1.5); pixelY++)
			{
				for (int pixelX = d_centers[clusterIDX *pitch + 4] - (d_step*1.5); pixelX < d_centers[clusterIDX *pitch + 4] + (d_step*1.5); pixelX++)
				{

					if (pixelX >= 0 && pixelX < d_rows && pixelY >= 0 && pixelY < d_cols) {

						uchar3 colour = d_colors[d_cols*pixelX + pixelY];

						float distance = compute_dist(clusterIDX, pixelX, pixelY, colour, d_centers, pitch, d_step);
						if (distance < d_distances[d_cols*pixelX + pixelY])
						{
							d_distances[d_cols*pixelX + pixelY] = distance;
							d_clusters[d_cols*pixelX + pixelY] = clusterIDX;
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
		__syncthreads();

		//if (idIn1D <= howManyPixels)
		//{
		//	compute1(idIn1D, d_cols, d_centers, d_center_counts, d_colors,  pitch);
		//}
		//	d_distances[idIn1D] = FLT_MAX;
		//	//printf("%f", d_distances[idIn1D]);
		//	int whichCluster = d_clusters[idIn1D];
		//	/*d_centers[whichCluster*pitch + 0] += d_colors[idIn1D].x;
		//	d_centers[whichCluster*pitch + 1] += d_colors[idIn1D].y;
		//	d_centers[whichCluster*pitch + 2] += d_colors[idIn1D].z;
		//	d_centers[whichCluster*pitch + 3] += idIn1D / d_cols;
		//	d_centers[whichCluster*pitch + 4] += idIn1D % d_cols;*/

		//	//atomicAdd(&d_center_counts[whichCluster], 1);

		//	/*int c_id = clusters[j][k];

		//	if (c_id != -1) {
		//		Vec3b colour = image.at<Vec3b>(k, j);

		//		centers[c_id][0] += colour.val[0];
		//		centers[c_id][1] += colour.val[1];
		//		centers[c_id][2] += colour.val[2];
		//		centers[c_id][3] += j;
		//		centers[c_id][4] += k;

		//		center_counts[c_id] += 1;
		//	}*/
		//}
		//__syncthreads();

	//}

	//d_distances[threadIdx.x] = compute_dist(885, threadIdx.x % d_rows, threadIdx.x / d_rows, d_colors[threadIdx.x], d_centers, pitch, d_step);
}

__global__ void compute1(int d_cols, int d_rows, int d_step, int d_centersLength, int *d_clusters, float *d_distances,
	float *d_centers, int *d_center_counts, uchar3 *d_colors, int pitch)
{
	int howManyPixels = d_cols*d_rows - 1;
	int idIn1D = blockIdx.x * blockDim.x + threadIdx.x;
	//ha a szál id-je nagyobb mint a pixelek száma, akkor az egy cluster
	//szál, amely szálnak az indexe itt kerül inicializálásra
	int clusterIDX = idIn1D - howManyPixels;

	if (idIn1D <= howManyPixels)
	{
		d_distances[idIn1D] = FLT_MAX;
		//printf("%f", d_distances[idIn1D]);
		int whichCluster = d_clusters[idIn1D];
		d_centers[whichCluster*pitch + 0] += d_colors[idIn1D].x;
		d_centers[whichCluster*pitch + 1] += d_colors[idIn1D].y;
		d_centers[whichCluster*pitch + 2] += d_colors[idIn1D].z;
		d_centers[whichCluster*pitch + 3] += idIn1D / d_cols;
		d_centers[whichCluster*pitch + 4] += idIn1D % d_cols;

		atomicAdd(&d_center_counts[whichCluster], 1);
	}
	__syncthreads();
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

	int threadsToBeStarted = rows*cols + centersLength - 1;
	int howManyBlocks = threadsToBeStarted / 700;
	int threadsPerBlock = (threadsToBeStarted / howManyBlocks) + 1;

	//for (int i = 0; i < iteration; i++)
	//{
		compute << <howManyBlocks, threadsPerBlock >> > (cols, rows, step, centersLength, d_clusters, d_distances, d_centers, d_center_counts, d_colors, 5);
		compute1 << <howManyBlocks, threadsPerBlock >> > (cols, rows, step, centersLength, d_clusters, d_distances, d_centers, d_center_counts, d_colors, 5);
	//}

	cudaMemcpy(distances, d_distances, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, d_clusters, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(centers, d_centers, sizeof(float)*centersLength * 5, cudaMemcpyDeviceToHost);
	cudaMemcpy(center_counts, d_center_counts, sizeof(float)*centersLength, cudaMemcpyDeviceToHost);

	dataFree();

	int a = 0;
	for (int i = 0; i < rows*cols; i++)
	{
		if (clusters[i] == -1)
		{
			a++;
		}
	}
	int b = rows*cols - a;

	printf("%i steps\n", step);
	printf("%i rows\n", rows);
	printf("%i cols\n", cols);
	printf("%i darab cluster\n", centersLength);
	printf("%i darab pixel\n", rows*cols);
	printf("%i darab elinditott szal\n", threadsPerBlock*howManyBlocks);
	printf("%i darab clusterhez van renderve\n", b);
	printf("%i darab nincs clusterhez renderve\n", a);
	
	int dis = 0;
	for (int i = 0; i < rows*cols; i++)
	{
		if (distances[i] == FLT_MAX)
		{
			dis++;
		}
	}
	printf("%i dis\n", dis);


	int mennyi = 0;
	for (int i = 0; i < centersLength; i++)
	{
		//cout << center_counts[i] << endl;
		mennyi += center_counts[i];
	}
	printf("%i mennyi\n", mennyi);

	//getchar();

	//getchar();
	//for (int i = 0; i < rows*cols; i++)
	//{
	//	cout << distances[i] << endl;
	//}

	//for (int i = 0; i < centersLength; i += 5)
	//{
	//	cout << centers[i] << " " << centers[i + 1] << " " << centers[i + 2] << " " << centers[i + 3] << " " << centers[i + 4] << endl;
	//}

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
