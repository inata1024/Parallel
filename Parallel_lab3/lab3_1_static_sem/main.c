//pthread普通高斯消去 静态线程+ 信号量同步版本
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include<pthread.h>
#include<semaphore.h>
#define NUM_THREADS 4
LARGE_INTEGER frequency;


float **m;//主函数动态分配
int N=0;

typedef struct {
    int t_id;
}threadParam_t;

//定义信号量
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS]; //每个线程有自己专属的信号量
sem_t sem_workerend[NUM_THREADS];

void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id; //线程编号

    for(int k=0;k<N;k++)
    {
        sem_wait(&sem_workerstart[t_id]);
        //循环划分任务
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS)
        {
            //消去
            for(int j=k+1;j<N;j++)
                m[i][j]=m[i][j]-m[i][k]*m[k][j];
            m[i][k]=0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
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
        sem_init(&sem_main, 0, 0);
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }


        //创建线程
        pthread_t handles[NUM_THREADS];// 创建对应的Handle
        threadParam_t param[NUM_THREADS];// 创建对应的线程数据结构
        for(int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
        }


        for(int k=0;k<N;k++)
        {
            for(int j=k+1;j<N;j++)
                m[k][j]=m[k][j]/m[k][k];
            m[k][k]=1.0;

            //唤醒工作线程
            for(int t_id=0;t_id<NUM_THREADS;++t_id)
                sem_post(&sem_workerstart[t_id]);

            //主线程睡眠
            //信号量减去NUM_THREADS 每个工作线程最后都会 sem_post(&sem_main)
            for(int t_id=0;t_id<NUM_THREADS;++t_id)
                sem_wait(&sem_main);

            //主程序唤醒工作线程
            for(int t_id=0;t_id<NUM_THREADS;++t_id)
                sem_post(&sem_workerend[t_id]);
        }

        for(int t_id=0;t_id<NUM_THREADS;++t_id)
            pthread_join(handles[t_id],NULL);

        sem_destroy(&sem_main);
        sem_destroy(sem_workerend);
        sem_destroy(sem_workerstart);


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
