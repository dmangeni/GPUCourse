
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "HighPerformanceTimer.h"

using namespace cv;
using namespace std;

void threshold(int thresh, int width, int height, unsigned char* data) {
	int len = width * height;
	for (unsigned char* index = data; index < &data[len]; index++) {
		if (*index > thresh) {
			*index = 255;
		}
		else {
			*index = 0;
		}
	}
}
void change_to_black_and_white(Mat &my_image, unsigned int thresh) {
	cvtColor(my_image, my_image, COLOR_RGB2GRAY);

	unsigned char THRESHOLD = 128;
	threshold(THRESHOLD, my_image.rows, my_image.cols, my_image.data);
}
void setCuda_device() {
	// Choose which GPU to run on, change this on a multi-GPU system.
	if (cudaSetDevice(0) != cudaSuccess) {
		throw("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}
void clean_up_deviceMemory(unsigned char**dev_a, unsigned char**dev_b) {

	if (!(*dev_a == nullptr))
		cudaFree(&dev_a);
	if (!(*dev_b == nullptr))
		cudaFree(&dev_b);
}
cudaError_t cuda_gray(Mat *image, unsigned int thresh) {
	
	unsigned char* dev_image_a = nullptr;
	unsigned char* dev_modified_image_a = nullptr;
	unsigned int size = (*image).cols * (*image).rows;

	cudaError_t cudaStatus;

	//CudaMalloc Original Image space
	if ((cudaStatus = cudaMalloc(&dev_image_a, sizeof(char)*size)) != cudaSuccess) {
		//clean up device memory
		clean_up_deviceMemory(&dev_image_a, &dev_modified_image_a);
		throw("CudaMalloc Original Image Space Failed.");
	}
	//CudaMalloc the modified image space
	if ((cudaStatus = cudaMalloc(&dev_modified_image_a, sizeof(unsigned char)*size)) != cudaSuccess) {
		//clean up device memory
		clean_up_deviceMemory(&dev_image_a, &dev_modified_image_a);
		throw("CudaMalloc Modified Image Space Failed.");
	}

	//Cuda Memcopy original image data
	if ((cudaStatus = cudaMemcpy(dev_image_a,image->data, (sizeof(unsigned char)*size), cudaMemcpyHostToDevice)) != cudaSuccess) {
		//clean up device memory
		throw("CudaMemCopy Failed.");
	}

	threshold(thresh, (*image).rows, (*image).cols, (*image).data);

	return cudaStatus;
}
void displayImage(Mat &image) {
	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image);
}

int main(int argc, char* argv[]) {
	
	HighPrecisionTime h;

	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!image.data) {
		cout << "Could not open or find image" << endl;
		return -1;
	}

	//Gray the image
	Mat blacked_image = image;
	unsigned char THRESHOLD = 128;
	double total_time = 0;
	unsigned int num_runs = 100;

	try {

		for (int i = 0; i < num_runs; i++) {
			h.TimeSinceLastCall();
			threshold(THRESHOLD, blacked_image.rows, blacked_image.cols, blacked_image.data);
			total_time += h.TimeSinceLastCall();
		}

		change_to_black_and_white(blacked_image, THRESHOLD);
		displayImage(blacked_image);
		waitKey(0);

		//Graying on the GPU
		setCuda_device();
		cudaError_t cudaStatus;
		if ((cudaStatus = cuda_gray(&image,THRESHOLD)) != cudaSuccess) {
			throw("Cuda Modifying Image Failed!");
		}

		
	
	}
	catch (char * err_message) {
		cout << "ERROR: " << err_message << endl;
	}
	

#if	defined(WIN32) || defined(_WIN64)
	system("pause");
#endif
	return 0;
}
