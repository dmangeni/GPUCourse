
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include<Windows.h>
#include "../HighPerformanceTimer/HighPerformanceTimer.h"


typedef int array_type_t;

bool allocMemory(array_type_t** a, array_type_t**b, int**c, int size) {
	bool retval = false;

	if (!((*a = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)) {
		retval = true;
	}
	if (!((*b = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)){
		retval = true;
	}
	if (!((*c = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)) {
		retval = true;
	}
	
	return retval;
}
void clean_up(array_type_t **a, array_type_t**b, array_type_t**c){
	
	if (!(*a == nullptr))
		free(*a);
	if (!(*b == nullptr))
		free(*b);
	if (!(*c == nullptr))
		free(*c);
}
bool fill_array(array_type_t *a, array_type_t*b, array_type_t*c, int size) {

	for (int i = 0; i < size; i++) {
		array_type_t random_number = (rand() % 100) + 1;
		a[i] = random_number;
		b[i] = random_number;
	}
	return (!(a == nullptr)||(b == nullptr));
}

void print_arrays(array_type_t *my_array, int size) {
	for (int i = 0; i < size; i++) {
		std::cout << my_array[i] << " ";
		if (i % 5 == 0 && i != 0) {
			std::cout << "\n";
		}
	}
	std::cout<<std::endl;
}
bool addVecSerialCPU(array_type_t *a, array_type_t*b, array_type_t*c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}

	return (!(c == nullptr));
}



int main(int argc, char*argv[]) {

	//Start timing.
	HighPrecisionTime h;

	int size = 100;
	if (argc > 1) {
		size = std::stoi(argv[1]);
		std::cout << "Size of array: " << size << std::endl;
	
	}
	else {
		std::cout << "ERROR: Usage: nameofprogram sizeofarray" << std::endl;
	}
		
	//Malloc 3 arrays
	array_type_t *a = nullptr;
	array_type_t *b = nullptr;
	array_type_t *c = nullptr;

	try {
		bool malloc_retval = allocMemory(&a, &b, &c, size);
		if (!malloc_retval)
			throw "ERROR: allocating memory for the arrays.";

		//Initialize randomness
		srand(GetTickCount());
		//A program that uses a commandline argument to fill an array
		std::cout << "Filling Arrays with random numbers:" << std::endl;
		//fill_array(a, b, c, size);
		if (!fill_array(a, b, c, size)) {
			throw "ERROR: filling arrays with random numbers.";
		}
		print_arrays(a, size);
		print_arrays(b, size);

		//Add the two vectors

		//startTimer
		const int AVERAGE_TIMES = 100;
		h.TimeSinceLastCall();
		double function_performance = 0;
		for (int i = 0; i < AVERAGE_TIMES; i++) {
			addVecSerialCPU(a, b, c, size);
			function_performance += h.TimeSinceLastCall();
			if (!addVecSerialCPU(a, b, c, size)) {
				throw "ERROR: filling array C.";
			}
		}
		function_performance = function_performance / AVERAGE_TIMES;

		std::cout <<"Runtime of AddVector:: " << function_performance << std::endl;
		
	}
	catch (char* err_message) {
		std::cout << err_message << std::endl;
	}

	clean_up(&a, &b, &c);
	system("pause");
	return 0;
}


