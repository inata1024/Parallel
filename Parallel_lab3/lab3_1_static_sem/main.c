//pthread��ͨ��˹��ȥ ��̬�߳�+ �ź���ͬ���汾
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include<pthread.h>
#include<semaphore.h>
#define NUM_THREADS 4
LARGE_INTEGER frequency;


float **m;//��������̬����
int N=0;

typedef struct {
    int t_id;
}threadParam_t;

//�����ź���
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS]; //ÿ���߳����Լ�ר�����ź���
sem_t sem_workerend[NUM_THREADS];

void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id; //�̱߳��

    for(int k=0;k<N;k++)
    {
        sem_wait(&sem_workerstart[t_id]);
        //ѭ����������
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS)
        {
            //��ȥ
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
        sem_init(&sem_main, 0, 0);
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }


        //�����߳�
        pthread_t handles[NUM_THREADS];// ������Ӧ��Handle
        threadParam_t param[NUM_THREADS];// ������Ӧ���߳����ݽṹ
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

            //���ѹ����߳�
            for(int t_id=0;t_id<NUM_THREADS;++t_id)
                sem_post(&sem_workerstart[t_id]);

            //���߳�˯��
            //�ź�����ȥNUM_THREADS ÿ�������߳���󶼻� sem_post(&sem_main)
            for(int t_id=0;t_id<NUM_THREADS;++t_id)
                sem_wait(&sem_main);

            //�������ѹ����߳�
            for(int t_id=0;t_id<NUM_THREADS;++t_id)
                sem_post(&sem_workerend[t_id]);
        }

        for(int t_id=0;t_id<NUM_THREADS;++t_id)
            pthread_join(handles[t_id],NULL);

        sem_destroy(&sem_main);
        sem_destroy(sem_workerend);
        sem_destroy(sem_workerstart);


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
