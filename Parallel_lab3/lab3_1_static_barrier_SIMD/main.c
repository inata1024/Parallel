//pthread��ͨ��˹��ȥ ��̬�߳�+ barrierͬ��+����ѭ��ȫ�������̺߳���+AVX
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
#include <immintrin.h> //AVX��AVX2
#define NUM_THREADS 4
LARGE_INTEGER frequency;


float **m;//��������̬����
int N=0;

typedef struct {
    int t_id;
}threadParam_t;

//barrier����
pthread_barrier_t barrier_division;
pthread_barrier_t barrier_elimination;

void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id; //�̱߳��

    for(int k=0;k<N;k++)
    {
        //һ�������߳̽��г���
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
            //����ʣ���±�
            while (j < N)
            {
                m[k][j] = m[k][j] / m[k][k];
                j++;
            }
            m[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_division);

        //ѭ����������
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
        N=a[tms];//ָ�������ģ
        //N=10;
        //���ɲ�������
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
        QueryPerformanceFrequency(&frequency);//���ʱ��Ƶ��
		dff = (double)frequency.QuadPart;//ȡ��Ƶ��
		QueryPerformanceCounter(&frequency);
		begin_ = frequency.QuadPart;//��ó�ʼֵ

        //��̬�̸߳�˹��ȥ
        //��ʼ���ź���
        //��ʼ��barrier
        pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
        pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);


        //�����߳�
        pthread_t handles[NUM_THREADS];// ������Ӧ��Handle
        threadParam_t param[NUM_THREADS];// ������Ӧ���߳����ݽṹ
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
		_end = frequency.QuadPart;//�����ֵֹ
		time = (_end - begin_) / dff;//��ֵ����Ƶ�ʵõ�ʱ��
		printf("�����ģ:%d ʱ��:%fms\n",a[tms],time*1000);

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