//pthread普通高斯消去 静态线程+ 信号量同步+三重循环全部纳入线程函数
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
sem_t sem_leader;
sem_t sem_division[NUM_THREADS-1];
sem_t sem_elimination[NUM_THREADS-1];

void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id; //线程编号

    for(int k=0;k<N;k++)
    {
        //一个工作线程进行除法
        if(t_id==0)
        {
            for(int j=k+1;j<N;j++)
                m[k][j]=m[k][j]/m[k][k];
            m[k][k]=1;
        }
        else
            sem_wait(&sem_division[t_id-1]);//阻塞，等待完成除法操作

        if(t_id==0)
        {
            for(int i=0;i<NUM_THREADS-1;i++)
                sem_post(&sem_division[i]);
        }

        //循环划分任务
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS)
        {
            for(int j=k+1;j<N;j++)
                m[i][j]=m[i][j]-m[i][k]*m[k][j];
            m[i][k]=0;
        }

        if(t_id==0)
        {
            for(int i=0;i<NUM_THREADS-1;i++)
                sem_wait(&sem_leader);

            for(int i=0;i<NUM_THREADS-1;i++)
                sem_post(&sem_elimination[i]);
        }
        else{
            sem_post(&sem_leader);
            sem_wait(&sem_elimination[t_id-1]);
        }
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

        //静态线程高斯消去
        //初始化信号量
        sem_init(&sem_leader, 0, 0);
        for (int i = 0; i < NUM_THREADS-1; ++i)
        {
            sem_init(&sem_division[i], 0, 0);
            sem_init(&sem_elimination[i], 0, 0);
        }


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

        sem_destroy(&sem_leader);
        sem_destroy(sem_division);
        sem_destroy(sem_elimination);


        QueryPerformanceCounter(&frequency);
		_end = frequency.QuadPart;//获得终止值
		time = (_end - begin_) / dff;//差值除以频率得到时间
		printf("矩阵规模:%d 时间:%fms\n",a[tms],time*1000);
//
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
