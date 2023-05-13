//pthread普通高斯消去 静态线程+ barrier同步+三重循环全部纳入线程函数+AVX
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include<pthread.h>
#include<semaphore.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#define NUM_THREADS 4
LARGE_INTEGER frequency;


float **m;//主函数动态分配
int N=0;

typedef struct {
    int t_id;
}threadParam_t;

//barrier定义
pthread_barrier_t barrier_division;
pthread_barrier_t barrier_elimination;

void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id; //线程编号

    for(int k=0;k<N;k++)
    {
        //一个工作线程进行除法
        if(t_id==0)
        {
             __m256 va, vt;
            vt = _mm256_set_ps(m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]);
            int j = 0;
            for (j = k + 1;j <= N - 8;j += 8)
            {
                va = _mm256_loadu_ps(m[k] + j);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(m[k] + j, va);
            }
            //处理剩余下标
            while (j < N)
            {
                m[k][j] = m[k][j] / m[k][k];
                j++;
            }
            m[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_division);

        //循环划分任务
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS)
        {
            __m256 vaik, vakj, vaij, vx;
            vaik = _mm256_set_ps(m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]);
            int j=0;
            for (j = k + 1;j <= N - 8;j += 8)
            {
                vakj = _mm256_loadu_ps(m[k] + j);
                vaij = _mm256_loadu_ps(m[i] + j);
                vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(m[i] + j, vaij);
            }
            while (j < N)
            {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
                j++;
            }
            m[i][k] = 0;
        }

        pthread_barrier_wait(&barrier_elimination);


    }
    pthread_exit(NULL);
}

int main(){

    int a[12]={128,256,384,512,640,768,896,1024,1280,1536,1792,2048};
    for(int tms=0;tms<12;tms++)
    {
        N=a[tms];//指定矩阵规模
        //N=10;
        //生成测试数据
        srand(1);
        m=malloc(sizeof(float*)*N);
        for(int i=0;i<N;i++)
        {
            m[i]=malloc(sizeof(float)*N);
            for(int j=0;j<N;j++)
                m[i][j]=0;
        }
        for(int i=0;i<N;i++)
        {
            m[i][i]=1.0;
            for(int j=i+1;j<N;j++)
                m[i][j]=rand();
        }
        for(int k=0;k<N;k++)
            for(int i=k+1;i<N;i++)
                for(int j=0;j<N;j++)
                    m[i][j]+=m[k][j];

        double dff=0, begin_=0, _end=0, time=0;
        QueryPerformanceFrequency(&frequency);//获得时钟频率
		dff = (double)frequency.QuadPart;//取得频率
		QueryPerformanceCounter(&frequency);
		begin_ = frequency.QuadPart;//获得初始值

        //动态线程高斯消去
        //初始化信号量
        //初始化barrier
        pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
        pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);


        //创建线程
        pthread_t handles[NUM_THREADS];// 创建对应的Handle
        threadParam_t param[NUM_THREADS];// 创建对应的线程数据结构
        for(int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
        }


        for(int t_id=0;t_id<NUM_THREADS;++t_id)
            pthread_join(handles[t_id],NULL);

        pthread_barrierattr_destroy(&barrier_division);
        pthread_barrierattr_destroy(&barrier_elimination);

        QueryPerformanceCounter(&frequency);
		_end = frequency.QuadPart;//获得终止值
		time = (_end - begin_) / dff;//差值除以频率得到时间
		printf("矩阵规模:%d 时间:%fms\n",a[tms],time*1000);

//        for(int k=0;k<N;k++)
//        {
//            for(int j=0;j<N;j++)
//            {
//                printf("%10.2f ",m[k][j]);
//            }
//            printf("\n");
//        }

        for(int i=0;i<N;i++)
            free(m[i]);

        free(m);
    }
	return 0;
}
