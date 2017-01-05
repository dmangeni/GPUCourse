
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include<Windows.h>

typedef int array_type_t;

bool allocMemory(array_type_t** a, array_type_t**b, int**c, int size) {
	bool retval = false;

	if (!((*a = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)) {
		retval = true;
	}
	if (!((*b = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)){
		retval = true;
	}
	if (!((*b = (array_type_t*)(malloc(sizeof(array_type_t)*size))) == nullptr)) {
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
void fill_array(array_type_t *a, array_type_t*b, array_type_t*c, int size) {

	for (int i = 0; i < size; i++) {
		array_type_t random_number = (rand() % 100) + 1;
		a[i] = random_number;
		b[i] = random_number;
	}

	for (int i = 0; i < size; i++) {
		std::cout << a[i] << " ";
		//std::cout << a[i] << " ";
	}
}


int main(int argc, char*argv[]) {

	int size = 100;
	if (argc > 1) {
		size = std::stoi(argv[1]);
		std::cout << "size of array: " << size << std::endl;
		std::cout << "argc: " << argc << std::endl;
		std::cout << "argv[0] " << argv[0] << std::endl;

		//std::cout << std::stoi(argv[1]) << std::endl;
	}
	else {
		std::cout << "ERROR: Usage: nameofprogram argument1" << std::endl;
	}
		
	//Malloc 3 arrays
	array_type_t *a = nullptr;
	array_type_t *b = nullptr;
	array_type_t *c = nullptr;

	try {
		bool malloc_retval = allocMemory(&a, &b, &c, size);
		if (!malloc_retval)
			throw "ERROR: allocating memory for the arrays.";
	}
	catch (char* err_message) {
		std::cout << err_message << std::endl;
	}

	clean_up(&a, &b, &c);
	
	//Initialize randomness
	srand(GetTickCount());

	fill_array(a, b, c, size);

	system("pause");
	return 0;
}


