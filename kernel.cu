/*添加了共享内存，改进了加法*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define DATA_SIZE 1048576 //1024*1024=1K*1K=1M
#define THREAD_NUM 1024
#define BLOCK_NUM 128

int data[DATA_SIZE];
int clkrate;
int cputime;

void GenerateNumbers(int *number, int size)
{
	for (int i = 0; i < size; i++)
	{
		number[i] = rand() % 10;
	}
}

void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);

	if (count == 0)
	{
		fprintf(stderr, "No device.\n");
		return false;
	}

	int i;

	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printDeviceProp(prop);
		clkrate = prop.clockRate;

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
				break;
		}
	}

	if (i == count)
	{
		fprintf(stderr, "No device supporting.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

__global__ static void sumOfSquares(int *num, int *result, clock_t *time)
{

	extern __shared__ int shared[];

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	shared[tid] = 0;

	int i;

	if (tid == 0)
		time[bid] = clock();

	for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM)
	{

		shared[tid] += num[i] * num[i] * num[i];
	}

	__syncthreads();

	int offset = 1, mask = 1;

	while (offset < THREAD_NUM)
	{
		if ((tid & mask) == 0)
		{
			shared[tid] += shared[tid + offset];
		}

		offset += offset;
		mask = offset + mask;
		__syncthreads();
	}

	if (tid == 0)
	{
		result[bid] = shared[0];
		time[bid + BLOCK_NUM] = clock();
	}
}

int main()
{
	

	if (!InitCUDA())
		return 0;

	GenerateNumbers(data, DATA_SIZE);

	int *gpudata, *result;

	clock_t *time;

	cudaMalloc((void **)&gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void **)&result, sizeof(int) * BLOCK_NUM);
	cudaMalloc((void **)&time, sizeof(clock_t) * BLOCK_NUM * 2);

	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> >(gpudata, result, time);

	int cpustart = clock();

	int sum[BLOCK_NUM];
	clock_t time_use[BLOCK_NUM * 2];

	cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_use, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);

	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);

	int final_sum = 0;

	for (int i = 0; i < BLOCK_NUM; i++)
	{

		final_sum += sum[i];
	}

	cputime = clock() - cpustart;

	clock_t min_start, max_end;

	min_start = time_use[0];

	max_end = time_use[BLOCK_NUM];

	for (int i = 1; i < BLOCK_NUM; i++)
	{
		if (min_start > time_use[i])
			min_start = time_use[i];
		if (max_end < time_use[i + BLOCK_NUM])
			max_end = time_use[i + BLOCK_NUM];
	}

	printf("---------------------------------------------------------\n");
	printf("\nGPUsum: %d \n", final_sum);
	printf("timestamp: %d \n", max_end - min_start);
	printf("time: %f ms \n", float(max_end - min_start) / (clkrate));

	final_sum = 0;

	for (int i = 0; i < DATA_SIZE; i++)
	{
		final_sum += data[i] * data[i] * data[i];
	}
	printf("CPUtime: %d\n", cputime);
	printf("CPUsum: %d \n\n", final_sum);

	return 0;
}