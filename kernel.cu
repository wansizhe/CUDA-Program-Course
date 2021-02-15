#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define THREAD_NUM 256

#define MATRIX_SIZE 1000

const int blocks_num = MATRIX_SIZE*(MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;

//��ӡ�豸��Ϣ
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

//CUDA ��ʼ��
bool InitCUDA()
{
	int count;

	//ȡ��֧��Cuda��װ�õ���Ŀ
	cudaGetDeviceCount(&count);

	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");

		return false;
	}

	int i;

	for (i = 0; i < count; i++)
	{

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//��ӡ�豸��Ϣ
		printDeviceProp(prop);

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}

	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;

}

//�����������
void matgen(float* a, int n)
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{

			a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);

		}
	}
}

// __global__ ���� ���м������˷�
__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n, clock_t* time)
{

	//��ʾĿǰ�� thread �ǵڼ��� thread���� 0 ��ʼ���㣩
	const int tid = threadIdx.x;

	//��ʾĿǰ�� thread ���ڵڼ��� block���� 0 ��ʼ���㣩
	const int bid = blockIdx.x;

	//�� bid �� tid �������� thread Ӧ�ü���� row �� column
	const int idx = bid * THREAD_NUM + tid;
	const int row = idx / n;
	const int column = idx % n;

	int i;

	//��¼���㿪ʼ��ʱ��
	clock_t start;

	//ֻ�� thread 0���� threadIdx.x = 0 ��ʱ�򣩽��м�¼��ÿ�� block �����¼��ʼʱ�估����ʱ��
	if (tid == 0) time[bid] = clock();

	//�������˷�
	if (row < n && column < n)
	{
		float t = 0;

		for (i = 0; i < n; i++)
		{
			t += a[row * n + i] * b[i * n + column];
		}
		c[row * n + column] = t;
	}

	//����ʱ��,��¼�����ֻ�� thread 0���� threadIdx.x = 0 ��ʱ�򣩽��У�ÿ�� block �����¼��ʼʱ�估����ʱ��
	if (tid == 0)
	{
		time[bid + blocks_num] = clock();
	}
}





int main()
{

	//CUDA ��ʼ��
	if (!InitCUDA()) return 0;

	//�������
	float *a, *b, *c, *d;

	int n = MATRIX_SIZE;

	//�����ڴ�
	a = (float*)malloc(sizeof(float)* n * n);
	b = (float*)malloc(sizeof(float)* n * n);
	c = (float*)malloc(sizeof(float)* n * n);
	d = (float*)malloc(sizeof(float)* n * n);

	//�������������
	srand(0);

	//������ɾ���
	matgen(a, n);
	matgen(b, n);

	/*�����ݸ��Ƶ��Կ��ڴ���*/
	float *cuda_a, *cuda_b, *cuda_c;

	clock_t* time;

	//cudaMalloc ȡ��һ���Կ��ڴ� 
	cudaMalloc((void**)&cuda_a, sizeof(float)* n * n);
	cudaMalloc((void**)&cuda_b, sizeof(float)* n * n);
	cudaMalloc((void**)&cuda_c, sizeof(float)* n * n);
	cudaMalloc((void**)&time, sizeof(clock_t)* blocks_num * 2);


	//cudaMemcpy �������ľ����Ƶ��Կ��ڴ���
	//cudaMemcpyHostToDevice - ���ڴ渴�Ƶ��Կ��ڴ�
	//cudaMemcpyDeviceToHost - ���Կ��ڴ渴�Ƶ��ڴ�
	cudaMemcpy(cuda_a, a, sizeof(float)* n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, sizeof(float)* n * n, cudaMemcpyHostToDevice);

	// ��CUDA ��ִ�к��� �﷨����������<<<block ��Ŀ, thread ��Ŀ, shared memory ��С>>>(����...);
	matMultCUDA << < blocks_num, THREAD_NUM, 0 >> >(cuda_a, cuda_b, cuda_c, n, time);

	/*�ѽ������ʾоƬ���ƻ����ڴ�*/

	clock_t time_use[blocks_num * 2];

	//cudaMemcpy ��������Դ��и��ƻ��ڴ�
	cudaMemcpy(c, cuda_c, sizeof(float)* n * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_use, time, sizeof(clock_t)* blocks_num * 2, cudaMemcpyDeviceToHost);

	//Free
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	cudaFree(time);

	//��ÿ�� block ����Ŀ�ʼʱ�䣬������Ľ���ʱ�������ȡ��������ʱ��
	clock_t min_start, max_end;

	min_start = time_use[0];

	max_end = time_use[blocks_num];

	for (int i = 1; i < blocks_num; i++)
	{
		if (min_start > time_use[i]) min_start = time_use[i];

		if (max_end < time_use[i + blocks_num]) max_end = time_use[i + blocks_num];
	}

	//�˺�������ʱ��
	clock_t final_time = max_end - min_start;



	//CPU����˷����������d
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double t = 0;

			for (int k = 0; k < n; k++)
			{

				t += a[i * n + k] * b[k * n + j];

			}

			d[i * n + j] = t;

		}
	}

	//��֤��ȷ���뾫ȷ��

	float max_err = 0;

	float average_err = 0;


	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (d[i * n + j] != 0)
			{
				//fabs�󸡵���x�ľ���ֵ
				float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);

				if (max_err < err) max_err = err;

				average_err += err;
			}
		}
	}

	printf("Max error: %g Average error: %g\n", max_err, average_err / (n * n));


	printf("gputime: %d\n", final_time);



	return 0;

}