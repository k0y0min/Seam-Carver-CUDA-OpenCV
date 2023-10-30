// https://github.com/k0y0min/Seam-Carver-CUDA-OpenCV
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
//#include <filesystem> or #include <dirent.h> for linux
#include <windows.h>
using namespace std;
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

//Add CUDA support
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/dnn.hpp>
using namespace cv::cuda;

string directoryPath = "D:\\CUDA-SeamCarver\\SeamCarver2\\project\\SeamCarver\\images";

__global__ void computePixelEnergy(const cv::cuda::PtrStepSz<cv::Vec3i> image, cv::cuda::PtrStepSz<double> energy) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < image.cols - 1 && x > 0 && y < image.rows - 1 && y > 0) {
		double sum = 0.0;
		for (int c = 0; c < 3; ++c) {
			double dx = image(y, x + 1).val[c] - image(y, x - 1).val[c];
			double dy = image(y + 1, x).val[c] - image(y - 1, x).val[c];
			sum += (dx * dx) + (dy * dy);
		}
		energy(y, x) = sqrt(sum);
	}
	else if (x == image.cols - 1 || x == 0 || y == image.rows - 1 || y == 0) {
		energy(y, x) = 1000.0;
	}
}

cv::cuda::GpuMat computeForwardEnergy_(const cv::cuda::GpuMat image_device) {
	cv::cuda::GpuMat energy_device(image_device.rows, image_device.cols, CV_64F);

	const int maxThreadsPerBlock = 1024;  // Maximum threads per block supported by the GPU
	dim3 blockDim, gridDim;
	blockDim.x = std::min(image_device.cols, maxThreadsPerBlock);
	blockDim.y = std::min(image_device.rows, maxThreadsPerBlock / (int)blockDim.x);

	// Calculating the grid size based on the image dimensions and block size
	gridDim.x = (image_device.cols + blockDim.x - 1) / blockDim.x;
	gridDim.y = (image_device.rows + blockDim.y - 1) / blockDim.y;

	// Launch the CUDA kernel
	computePixelEnergy << <gridDim, blockDim >> > (image_device, energy_device);
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();
	return energy_device;
}

__global__ void dp_subroutine(cv::cuda::PtrStepSz<int> sol, cv::cuda::PtrStepSz<double> cM, int row) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	double min_energy;
	if (x > 0 && x < cM.cols - 1) {
		int t = x - 1;
		min_energy = cM(row - 1, t);
		for (int k = x; k <= x + 1; k++) {
			if (cM(row - 1, k) < min_energy) {
				min_energy = cM(row - 1, k);
				t = k;
			}
		}
		sol(row, x) = t;
	}
	else if (x == 0) {
		int t = x;
		min_energy = cM(row - 1, t);
		if (cM(row - 1, x + 1) < min_energy) {
			min_energy = cM(row - 1, x + 1);
			t = x + 1;
		}
		sol(row, x) = t;
	}
	else if (x == cM.cols - 1) {
		int t = x - 1;
		min_energy = cM(row - 1, t);
		if (cM(row - 1, x) < min_energy) {
			min_energy = cM(row - 1, x);
			t = x;
		}
		sol(row, x) = t;
	}
	else return;
	cM(row, x) += min_energy;
}

cv::cuda::GpuMat carve_vertical_seam(cv::cuda::GpuMat image_device) {
	cv::cuda::GpuMat energy_device = computeForwardEnergy_(image_device);
	cudaError_t cudaStatus;
	//add noise to the image
	cv::RNG rng(0);
	cv::Mat noise(energy_device.size(), CV_64F);
	int total = image_device.rows * image_device.cols;
	double std_dev = 1 / (1000 * std::sqrt(total));
	rng.fill(noise, cv::RNG::NORMAL, 0, std_dev);
	//just use noise?
	cv::cuda::GpuMat noise_device;
	noise_device.upload(noise);

	cv::cuda::add(energy_device, noise_device, energy_device);
	noise_device.~GpuMat();
	cv::cuda::GpuMat cM_device(energy_device.size(), CV_64F);
	energy_device.copyTo(cM_device);
	cv::cuda::GpuMat sol_device(energy_device.size(), CV_32S, cv::Scalar(0.0));

	for (int i = 1; i < energy_device.rows; i++) {
		dp_subroutine <<<(energy_device.cols / 1024) + 1, std::min(1024, energy_device.cols) >>> (sol_device, cM_device, i);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaDeviceSynchronize();
	}

	cv::Mat sol, cM;
	sol_device.download(sol);
	cM_device.download(cM);

	vector<int> minSeam(energy_device.rows, 0);
	int j = -1;
	double min_val = -1;
	for (int i = 0; i < cM.cols; i++) {
		double val = cM.at<double>(cM.rows - 1, i);
		if (min_val < val) {
			j = i;
			min_val = val;
		}
	}

	for (int i = cM.rows - 1; i >= 0; i--) {
		minSeam[i] = j;
		if (i == 0) break;
		j = sol.at<int>(i, j);
	}

	cv::Mat image;
	image_device.download(image);
	cv::Mat carved_image(image_device.rows, image_device.cols - 1, image_device.type());
	for (int i = 0; i < carved_image.rows; i++) {
		for (int j = 0; j < carved_image.cols - 1; j++) {
			if (j >= minSeam[i]) carved_image.at<cv::Vec3i>(i, j) = image.at<cv::Vec3i>(i, j + 1);
			else carved_image.at<cv::Vec3i>(i, j) = image.at<cv::Vec3i>(i, j);
		}
	}
	cv::cuda::GpuMat carved_image_device;
	carved_image_device.upload(carved_image);
	return carved_image_device;
}



cv::cuda::GpuMat carve_horizontal_seam(cv::cuda::GpuMat image_device) {
	// recompute energy if using energy computing methods like the ones involving gradient
	// transpose the long way beacuase cv::cuda::transpose got issues
	// kind of a bottleneck; TODO : write a kernel for transpose 
	cv::Mat image;
	image_device.download(image);
	cv::transpose(image, image);
	cv::cuda::GpuMat image_device_transposed;
	image_device_transposed.upload(image);
	image_device_transposed = carve_vertical_seam(image_device_transposed);
	image_device_transposed.download(image);
	cv::transpose(image, image);
	image_device_transposed.upload(image);
	return image_device_transposed;
}

std::string convertBackSlashesToForwardSlashes(const std::string& inputPath) {
	std::string outputPath = inputPath;
	for (size_t i = 0; i < outputPath.length(); i++) {
		if (outputPath[i] == '\\') {
			outputPath[i] = '/';
		}
	}
	return outputPath;
}

std::vector<std::string> listFilesInDirectory(const std::string& directoryPath) {
	std::vector<std::string> fileNames;
	WIN32_FIND_DATA findFileData;
	HANDLE hFind = FindFirstFile((directoryPath + "\\*").c_str(), &findFileData);

	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				fileNames.push_back(findFileData.cFileName);
			}
		} while (FindNextFile(hFind, &findFileData) != 0);
		FindClose(hFind);
	}

	return fileNames;
}

void run_batch(int num_seams_v, int num_seams_h, int num_iterations, string extension) {
	if (extension.size() == 0) extension = ".jpg";
	vector<cv::Mat> images;
	std::vector<std::string> fileNames = listFilesInDirectory(directoryPath);
	if (num_iterations == -1) num_iterations = fileNames.size();
	for (int i = 0; i < num_iterations; i++) {
		std::string filePath = directoryPath + "\\" + fileNames[i];
		cv::Mat image = cv::imread(convertBackSlashesToForwardSlashes(filePath));
		image.convertTo(image, CV_32SC3);
		images.push_back(image);
	}

	auto start = std::chrono::high_resolution_clock::now();
	for (int idx = 0; idx < images.size(); idx++) {
		cv::cuda::GpuMat image_device;
		image_device.upload(images[idx]);
		for (int i = 1; i <= max(num_seams_v, num_seams_h); i++) {
			if ((i == num_seams_v) && (num_seams_v == num_seams_h)) {
				image_device = carve_vertical_seam(image_device);
				image_device = carve_horizontal_seam(image_device);
			}
			else if ((i > min(num_seams_h, num_seams_v)) && num_seams_v > num_seams_h) image_device = carve_vertical_seam(image_device);
			else if ((i > min(num_seams_h, num_seams_v)) && num_seams_v < num_seams_h) image_device = carve_horizontal_seam(image_device);
			else {
				image_device = carve_vertical_seam(image_device);
				image_device = carve_horizontal_seam(image_device);
			}
		}
		cv::Mat carved_image;
		image_device.download(carved_image);
		cv::imwrite(directoryPath + "/" + fileNames[idx] + "_carved" + extension, carved_image);
		printf("\r%d / %d images done", idx + 1, images.size());
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	printf("\r%d images | %lld seconds\n", num_iterations, duration.count());
}

int main(int argc, char* argv[]) {
	cudaSetDevice(0);
	string extension;
	int dimV, dimH, num;
	if(argc != 4 && argc != 5 && argc != 6){
		std::cerr << "Invalid number of arguments. \ncarve dimV dimH dir num extension \n" 
			"dimV, dimH - number of vertical/horizontal seams to carve; dir - location of the directory containing the images(only)\n"
			"num - number of files from the directory to carve; extension - .png/.jpg\n";
		return -1;
	}
	try {
		dimV = atoi(argv[1]);
		dimH = atoi(argv[2]);
		directoryPath = argv[3];
		if (argc < 5) num = -1;
		else num = atoi(argv[4]);
		if (argc < 6) extension = "";
		else extension = argv[5];
	}
	catch (exception& e) {
		cerr << e.what() << endl;
	}
	run_batch(dimV, dimH, num, extension);
	return 0;
}
