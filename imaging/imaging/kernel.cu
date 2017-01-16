
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "HighPerformanceTimer.h"
#include <iomanip>

using namespace cv;
using namespace std;


//Globals
unsigned char* dev_image = nullptr;
unsigned char* dev_modified_image = nullptr;

//Mat* gpu_modified_image;
String window2;

__global__ void addKernel(int thresh, unsigned char* dev_original_image, unsigned char*dev_modified_image, unsigned int size)
{
	//int i = threadIdx.x;
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (i < size) {
		if (dev_original_image[i]> thresh) {
			dev_modified_image[i] = 255;
		}
		else {
			dev_modified_image[i] = 0;
		}
	}
}
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
void gray_image(Mat &my_image, unsigned int thresh) {
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
void cuda_allocate_memory(Mat *src_image) {
	
	unsigned int size = (*src_image).cols * (*src_image).rows;
	cudaError_t cudaStatus;

	//CudaMalloc original Image space
	if ((cudaStatus = cudaMalloc(&dev_image, sizeof(char)*size)) != cudaSuccess) {
		//clean up device memory
		clean_up_deviceMemory(&dev_image, &dev_modified_image);
		throw("CudaMalloc Original Image Space Failed.");
	}
	//CudaMalloc modified image space
	if ((cudaStatus = cudaMalloc(&dev_modified_image, sizeof(unsigned char)*size)) != cudaSuccess) {
		//clean up device memory
		clean_up_deviceMemory(&dev_image, &dev_modified_image);
		throw("CudaMalloc Modified Image Space Failed.");
	}

	//Cuda Memcopy original image data
	if ((cudaStatus = cudaMemcpy(dev_image, src_image->data, (sizeof(unsigned char)*size), cudaMemcpyHostToDevice)) != cudaSuccess) {
		//clean up device memory
		throw("CudaMemCopy Failed.");
	}

	//double totaltime;
	//HighPrecisionTime h;
	//h.TimeSinceLastCall();

	/*int max_threads_per_block = 1024;
	int num_blocks = (size + 1023)/1024;


	for (int i = 0; i < num_runs; i++) {

		addKernel <<< num_blocks, max_threads_per_block >>>(thresh, dev_image_a, dev_modified_image_a, size);
		totaltime += h.TimeSinceLastCall();
	}*/

	//std::cout << "Runtime of graying-GPU:: " << std::fixed << std::showpoint << std::setprecision(10) << totaltime / num_runs << std::endl;

	/*/ Check for any errors launching the kernel
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
		clean_up_deviceMemory(&dev_image_a, &dev_modified_image_a);
		throw("Add kernel launch failed.");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
		clean_up_deviceMemory(&dev_image_a, &dev_modified_image_a);
		throw("cudaDeviceSynchronize returned error after launching addKernel!");
	}

	// Copy output from GPU buffer to host memory.
	if ((cudaStatus = cudaMemcpy(modified_image_h, dev_modified_image_a, size, cudaMemcpyDeviceToHost)) != cudaSuccess) {
		clean_up_deviceMemory(&dev_image_a, &dev_modified_image_a);
		throw("cudaMemcpy from device to host failed!");
	}

	//Clean up device memory
	clean_up_deviceMemory(&dev_image_a, &dev_modified_image_a);*/
}
void displayImage(Mat &image, string my_window) {
	namedWindow(my_window, WINDOW_NORMAL);
	imshow(my_window, image);
}
void on_trackbar(int thresh, void*gpu_modified_image) {

	Mat* temp_new_image = (Mat*)gpu_modified_image;
	unsigned int size = (*temp_new_image).cols * (*temp_new_image).rows;

	int max_threads_per_block = 1024;
	int num_blocks = (size + 1023) / 1024;


	cudaError_t cudaStatus;
	addKernel <<< num_blocks, max_threads_per_block >> >(thresh, dev_image, dev_modified_image, size);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
		clean_up_deviceMemory(&dev_image, &dev_modified_image);
		throw("cudaDeviceSynchronize returned error after launching addKernel!");
	}
	
	// Copy output from GPU buffer to host memory.
	if ((cudaStatus = cudaMemcpy((*(Mat*)gpu_modified_image).data, dev_modified_image, size, cudaMemcpyDeviceToHost)) != cudaSuccess) {
		clean_up_deviceMemory(&dev_image, &dev_modified_image);
		throw("cudaMemcpy from device to host failed!");
	}
	
	//Display the modified image
	imshow(window2, (*(Mat*)gpu_modified_image));

	//clean up device memory
	clean_up_deviceMemory(&dev_image, &dev_modified_image);	
}
int main(int argc, char* argv[]) {
	
	HighPrecisionTime h;

	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	//Open your image
	Mat src_image;
	src_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!src_image.data) {
		cout << "Could not open or find image" << endl;
		return -1;
	}

	//Initialize variables
	Mat cpu_modified_image = src_image;
	unsigned char THRESHOLD = 128;
	unsigned int num_runs = 100;
	double total_time = 0;
	
	try {

		//Gray on the CPU
		for (int i = 0; i < num_runs; i++) {
			h.TimeSinceLastCall();
			threshold(THRESHOLD, cpu_modified_image.rows, cpu_modified_image.cols, cpu_modified_image.data);
			total_time += h.TimeSinceLastCall();
		}

		std::cout << "Runtime of graying-CPU:: " << std::fixed << std::showpoint << std::setprecision(10) << total_time / num_runs << std::endl;

		String window1 = "window_1";
		gray_image(cpu_modified_image, THRESHOLD);
		displayImage(cpu_modified_image, window1);
		
		//Graying on the GPU
		//Variable to hold gpu modified image space
		Mat gpu_modified_image = src_image;
		cvtColor(gpu_modified_image, gpu_modified_image, COLOR_RGB2GRAY);

		//Initialize variables for the slider
		const int slider_max = 255;
		int slider = 128;

		setCuda_device();
		cuda_allocate_memory(&src_image);

		//Display the image
		window2 = "window_2";
		//displayImage(gpu_modified_image, window2);

		//namedWindow(window2, WINDOW_NORMAL);
	
		//create the trackbar and display
		createTrackbar("trackbar1", window2, &slider, slider_max, on_trackbar, (void*)&gpu_modified_image);
		on_trackbar(THRESHOLD, (void*)&gpu_modified_image);

		waitKey(0);
	}
	catch (char * err_message) {
		cout << "ERROR: " << err_message << endl;
	}	

#if	defined(WIN32) || defined(_WIN64)
	system("pause");
#endif
	return 0;
}
