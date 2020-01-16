#include <stdio.h>
#define SIZE 1024

// Vector addition using GPU way
// __global__ keyword used for recognizing function to be run on GPU rather than CPU

__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
	// Launching individual thread index
	// Note: For loops are not used here
	int i = threadIdx.x;

	// Just to sense check whether number of threads initiated are less than specified size
	if (i < n)
		c[i] = a[i] + b[i];
}




int main() {

	// Variable initiated for running on host/CPU
	int *a;
	int *b;
	int *c;

	// Variable initiated for running on device/GPU
	int *gpu_a;
	int *gpu_b;
	int *gpu_c;

	// Memory management for CPU variables using malloc function
	a = (int *)malloc(SIZE * sizeof(int));
	b = (int *)malloc(SIZE * sizeof(int));
	c = (int *)malloc(SIZE * sizeof(int));

	// Memory management for GPU variables using cudaMalloc function
	cudaMalloc(&gpu_a, SIZE * sizeof(int));
	cudaMalloc(&gpu_b, SIZE * sizeof(int));
	cudaMalloc(&gpu_c, SIZE * sizeof(int));

	// Creating values for CPU input variables a & b using loop for illustration purpose only
	for (int i = 0; i <SIZE; i++)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	// Copying CPU variables into GPU from host(CPU) to device (GPU)
	cudaMemcpy(gpu_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Number of blocks used are 1, and 
	// Number of threads used are specified as: 1024
	VectorAdd << <1, SIZE >> > (gpu_a, gpu_b, gpu_c, SIZE);

	// wait for GPU to finish before accessing the host
	cudaDeviceSynchronize();

	// Copying GPU variable to CPU for printing purpose etc.
	cudaMemcpy(c, gpu_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);


	// Iterate over CPU variable and print top 10
	printf("GPU Code\n");
	for (int j = 0; j < 10; j++)
	{
		printf("c[%d] = %d\n", j, c[j]);
	}


	// Freeing up GPU variables
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);

	// Freeing up CPU variables
	free(a);
	free(b);
	free(c);

	return 0;


}
