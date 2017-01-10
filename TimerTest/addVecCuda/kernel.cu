
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include<Windows.h>
#include "../HighPerformanceTimer/HighPerformanceTimer.h"
#include <omp.h>

typedef int array_type_t;
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
void setCuda_device() {
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ((cudaStatus = cudaSetDevice(0)) != cudaSuccess) {
		throw("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}
void cuda_add(const array_type_t*a, const array_type_t*b, array_type_t*c, unsigned int arraySize) {
	
	//Set the device pointers
	array_type_t* dev_a = 0;
	array_type_t* dev_b = 0;
	array_type_t* dev_c = 0;
	cudaError_t cudaStatus;

	//Allocate device memory
	if((cudaStatus = cudaMalloc(&dev_a,(sizeof(array_type_t)*arraySize))) != cudaSuccess) {
		throw("cudaMalloc failed!");
	}

	if ((cudaStatus = cudaMalloc(&dev_b, (sizeof(array_type_t)*arraySize))) != cudaSuccess) {
		throw("cudaMalloc failed!");
	}

	//Copy memory from the host memory to the device
	if ((cudaStatus = cudaMemcpy(dev_a,a, (sizeof(array_type_t)*arraySize), cudaMemcpyHostToDevice)) != cudaSuccess) {
		throw("cudaMalloc of dev_a failed!");
	}

	if ((cudaStatus = cudaMemcpy(dev_b,b,(sizeof(array_type_t)*arraySize), cudaMemcpyHostToDevice)) != cudaSuccess) {
		throw("cudaMalloc of dev_b failed!");
	}

	//Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, arraySize>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
		throw("Add kernel launch failed.");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	if ((cudaStatus= cudaDeviceSynchronize()) != cudaSuccess) {
		throw("cudaDeviceSynchronize returned error after launching addKernel!",);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(arraySize), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

}
int main()
{
	const unsigned int arraySize = 5;
    const array_type_t a[arraySize] = { 1, 2, 3, 4, 5 };
    const array_type_t b[arraySize] = { 10, 20, 30, 40, 50 };
    array_type_t c[arraySize] = { 0 };

	

	try {
		setCuda_device();
		cuda_add(a, b, c, arraySize);
	}
	catch (char* err_message){
		std::cout << err_message << std::endl;
	}

    /*/ Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

#if	defined(WIN32) || defined(_WIN64)
	system("pause");
#endif
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

   

    /*/ Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }*/

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
