
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include<Windows.h>
#include "../HighPerformanceTimer/HighPerformanceTimer.h"
#include <iomanip>
#include <vector>
#include <climits>


typedef int array_type_t;

__global__ void addKernel(array_type_t *c, const array_type_t *a, const array_type_t *b, unsigned int arraySize)
{
	//int i = threadIdx.x;
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < arraySize) {
		c[i] = a[i] + b[i];
	}
}
void setCuda_device() {
	// Choose which GPU to run on, change this on a multi-GPU system.
	if (cudaSetDevice(0) != cudaSuccess) {
		throw("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}
void clean_up_deviceMemory(array_type_t **dev_a, array_type_t**dev_b, array_type_t**dev_c) {

	if (!(*dev_a == nullptr))
		cudaFree(&dev_a);
	if (!(*dev_b == nullptr))
		cudaFree(&dev_b);
	if (!(*dev_c == nullptr))
		cudaFree(&dev_c);
}
void clean_up_cpuMemory(array_type_t **a, array_type_t**b, array_type_t**c) {

	if (!(*a == nullptr))
		free(*a);
	if (!(*b == nullptr))
		free(*b);
	if (!(*c == nullptr))
		free(*c);
}
bool allocMemory(array_type_t**a, array_type_t**b, int**c, unsigned int size) {
	bool retval = false;

	if (!((*a = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)) {
		retval = true;
	}
	if (!((*b = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)) {
		retval = true;
	}
	if (!((*c = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)) {
		retval = true;
	}

	return retval;
}
bool fill_array(array_type_t*a, array_type_t*b, array_type_t*c, unsigned int size) {
	for (int i = 0; i < size; i++) {
		array_type_t random_number = (rand() % 100) + 1;
		a[i] = random_number;
		b[i] = random_number;
		c[i] = 0;
	}
	return (!(a == nullptr) || (b == nullptr));
}
bool addVecSerialCPU(array_type_t *a, array_type_t*b, array_type_t*c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
	return (!(c == nullptr));
}
cudaError_t cuda_add(const array_type_t*a, const array_type_t*b, array_type_t*c, unsigned int arraySize, size_t num_runs) {
	
	//Set the device pointers
	array_type_t* dev_a = 0;
	array_type_t* dev_b = 0;
	array_type_t* dev_c = 0;
	cudaError_t cudaStatus;

	//Allocate device memory
	if((cudaStatus = cudaMalloc(&dev_a,(sizeof(array_type_t)*arraySize))) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("cudaMalloc failed!");
	}

	if ((cudaStatus = cudaMalloc(&dev_b, (sizeof(array_type_t)*arraySize))) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("cudaMalloc failed!");
	}

	if ((cudaStatus = cudaMalloc(&dev_c, (sizeof(array_type_t)*arraySize))) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("cudaMalloc failed!");
	}

	//start timing
	HighPrecisionTime h;
	double totaltime = 0;
	h.TimeSinceLastCall();

	//Copy memory from the host memory to the device
	if ((cudaStatus = cudaMemcpy(dev_a,a, (sizeof(array_type_t)*arraySize), cudaMemcpyHostToDevice)) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("cudaMalloc of dev_a failed!");
	}

	if ((cudaStatus = cudaMemcpy(dev_b,b,(sizeof(array_type_t)*arraySize), cudaMemcpyHostToDevice)) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("cudaMalloc of dev_b failed!");
	}

	totaltime += h.TimeSinceLastCall();
	//std::cout << "Allocated and moved memory onto GPU. Moving time: " << totaltime << std::endl;

	int max_threads_per_block = 1024;
	int num_blocks = arraySize/1024 + 1;

	for (int i = 0; i < num_runs; i++) {
		//Launch a kernel on the GPU with one thread for each element.
		addKernel << < num_blocks, max_threads_per_block >> >(dev_c, dev_a, dev_b, arraySize);
		//Add the timing code.
		totaltime += h.TimeSinceLastCall();
	}

	std::cout << "Runtime of addingVectors-GPU:: " << std::fixed << std::showpoint << std::setprecision(10) << totaltime / num_runs << std::endl;

	// Check for any errors launching the kernel
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("Add kernel launch failed.");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("cudaDeviceSynchronize returned error after launching addKernel!");
	}

	// Copy output vector from GPU buffer to host memory.
	if ((cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(arraySize), cudaMemcpyDeviceToHost)) != cudaSuccess) {
		clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);
		throw("cudaMemcpy failed!");
	}
	
	//Clean up device memory
	clean_up_deviceMemory(&dev_a, &dev_b, &dev_c);

	return cudaStatus;
}

int main(int argc, char*argv[])
{
	unsigned int arraySize = 5;
	size_t num_runs = 100;

	if (argc > 2) {
		arraySize = std::stoi(argv[1]);

		if (arraySize > INT_MAX) {
			arraySize = INT_MAX;
		}
		num_runs = std::stoi(argv[2]);
		std::cout << "Size of array: " << num_runs << std::endl;

	}
	else {
		std::cout << "ERROR: Usage: nameofprogram sizeofarray number of runs" << std::endl;
	}

	//Malloc 3 arrays
	array_type_t *a = nullptr;
	array_type_t *b = nullptr;
	array_type_t *c = nullptr;

	std::vector<int>sizes = { 100, 1000, 10000, 100000, 5000000 };
	std::vector<int> reps = { 10, 100, 1000 };

	try {
		bool malloc_retval = allocMemory(&a, &b, &c, arraySize);
		if (!malloc_retval)
			throw "ERROR: allocating memory for the arrays.";

		//Initialize randomness
		srand(GetTickCount());

		//Initialize randomness
		srand(GetTickCount());
		//A program that uses a commandline argument to fill an array
		std::cout << "Filling Arrays with random numbers:" << std::endl;
		if (!fill_array(a, b, c, arraySize)) {
			throw "ERROR: filling arrays with random numbers.";
		}

		//startTimer
		double function_performance = 0;

		//Declare time object
		HighPrecisionTime h;
		h.TimeSinceLastCall();
		for (int i = 0; i < num_runs; i++) {
			//h.TimeSinceLastCall();
			addVecSerialCPU(a, b, c, arraySize);
			function_performance += h.TimeSinceLastCall();
		}
		function_performance = function_performance / num_runs;
		std::cout << "Runtime of addingVectors-CPU:: " << std::fixed << std::showpoint << std::setprecision(10) << function_performance / num_runs << std::endl;

		setCuda_device();
		if ((cuda_add(a, b, c, arraySize, num_runs)) != cudaSuccess){
			throw("cuda_add failed!.");
		}

		/*printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
			c[0], c[1], c[2], c[3], c[4]);*/
		
		//cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		if (cudaDeviceReset() != cudaSuccess) {
			throw("cudaDeviceReset failed!");
			//return 1;
		}

	}
	catch (char* err_message){
		std::cout << err_message << std::endl;
	}

#if	defined(WIN32) || defined(_WIN64)
	system("pause");
#endif
	clean_up_cpuMemory(&a, &b, &c);
    return 0;
}
