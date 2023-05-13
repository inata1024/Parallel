//pthread��ͨ��˹��ȥ ��̬�߳�
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include<pthread.h>
LARGE_INTEGER frequency;

float **m;//��������̬����
int N=0;

typedef struct {
    int k;
    int t_id;
}threadParam_t;

void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int k = p -> k; //��ȥ���ִ�
    int t_id = p -> t_id; //�̱߳��
    int i = k + t_id + 1; //��ȡ�Լ��ļ�������

    for(int j=k+1;j<N;++j)
        m[i][j]=m[i][j]-m[i][k]*m[k][j];
    m[i][k]=0;
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
        for(int k=0;k<N;k++)
        {
            for(int j=k+1;j<N;j++)
                m[k][j]=m[k][j]/m[k][k];
            m[k][k]=1.0;

            //���������߳�
            int worker_count=N-1-k;

            pthread_t *handles=malloc(sizeof(pthread_t)*worker_count);//handle ָ�빲��
            threadParam_t *param=malloc(sizeof(threadParam_t)*worker_count);//�߳����ݽṹ

            //��������
            for(int t_id=0;t_id<worker_count;t_id++)
            {
                param[t_id].k=k;
                param[t_id].t_id=t_id;
            }
            //�����߳�
            for(int i=0;i<worker_count;i++)
                pthread_create(&handles[i],NULL,threadFunc,(void*)&param[i]);

            //���̹߳���
            for(int t_id=0;t_id<worker_count;t_id++)
                pthread_join(handles[t_id],NULL);


        }

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

//        for(int i=0;i<N;i++)
//            free(m[i]);
//
//        free(m);
    }
	return 0;
}
