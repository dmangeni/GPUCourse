
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

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
int main(int argc, char* argv[]) {
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

	//cout << "NUmber of Channels: " << image.channels << endl;

	Mat image2 = image;
	cvtColor(image2, image2, COLOR_RGB2GRAY);
	
	unsigned char THRESHOLD = 128;
	threshold(THRESHOLD, image2.rows, image2.cols, image2.data);

	namedWindow("Display window", WINDOW_NORMAL);
	imshow("Display window", image2);

	waitKey(0);

#if	defined(WIN32) || defined(_WIN64)
	system("pause");
#endif
	return 0;
}
