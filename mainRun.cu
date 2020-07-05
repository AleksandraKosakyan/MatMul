
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BLOCK_SIZE 32
#define N 2048

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}                 \



__global__ void matMult(float* A, float* B, float* C){

	int bx = blockIdx.x;
	int by = blockIdx.y;


	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0;

	int ia = N * BLOCK_SIZE * by + N * ty;

	int ib = BLOCK_SIZE * bx + tx;


	for (int k = 0; k < N; k++) {
		sum += A[ia + k] * B[ib + k * N];
	}
	// Индекс C[i][j]
	int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	
	C[ic + N * ty + tx] = sum;
}




void cpu_ikj(float* A, float* B, float* C) {
	for (int i = 0; i < N; i++) {
		for (int k = 0; k < N; k++) {
			for (int j = 0; j < N; j++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

void printMatrix(float* C) {
	for ( int i = 0; i < N; i++){
		for ( int j = 0; j < N; j++){
			printf("%.3f ", C[i * N + j]);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}




int main() {
	setlocale(LC_ALL, "RUS");


	float *A = (float*) malloc(N * N *sizeof(float));
	float *B = (float*) malloc(N * N* sizeof(float));
	float *C_GPU = (float*) malloc(N * N *sizeof(float));
	float *C_CPU = (float*) malloc(N * N*  sizeof(float));



	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			A[i + j * N] = i + j;
			B[i + j * N] = i + j;
		}


	dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	cudaEvent_t start;
	cudaEvent_t stop;

	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));


	float* adev, *bdev, *cdev;


	CUDA_CHECK_ERROR(cudaMalloc((void**)&adev, N * N * sizeof(float *)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&bdev, N * N * sizeof(float *)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&cdev, N * N * sizeof(float *)));

	
	CUDA_CHECK_ERROR(cudaMemcpy(adev, A, N * N * sizeof(float *), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(bdev, B, N * N * sizeof(float *), cudaMemcpyHostToDevice));

	
	cudaEventRecord(start, 0);    
								

	matMult << < dimGrid, dimBlock >> > (adev, bdev , cdev);

	cudaEventRecord(stop, 0);    

	float timeGPU = 0;

	
	cudaEventSynchronize(stop);   

	
	cudaEventElapsedTime(&timeGPU, start, stop);    

	std::cout << "Время умножения матриц размером " << N << "x" << N << " на GPU = " << timeGPU << " мсек" << std::endl;

	
	CUDA_CHECK_ERROR(cudaMemcpy(C_GPU, cdev, N * N * sizeof(float *), cudaMemcpyDeviceToHost));

	

	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));

	CUDA_CHECK_ERROR(cudaFree(adev));
	CUDA_CHECK_ERROR(cudaFree(bdev));
	CUDA_CHECK_ERROR(cudaFree(cdev));

	double start_time = clock();

	cpu_ikj(A, B, C_CPU);

	double end_time = clock();

	std::cout << "Время умножения матриц размером " << N << "x" << N << " на CPU = " << ((end_time - start_time)) *1000 / CLOCKS_PER_SEC << " мсек" << std::endl;
	
	
	delete A;
	delete B;
	delete C_GPU;
	delete C_CPU;
	system("pause");
	return 0;
}
