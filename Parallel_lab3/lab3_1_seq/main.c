//��ͨ��˹��ȥ����
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include<pthread.h>
LARGE_INTEGER frequency;

float **m;//��������̬����
int N=0;

int main(){

    int a[12]={128,256,384,512,640,768,896,1024,1280,1536,1792,2048};
    for(int tms=0;tms<7;tms++)
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

        //���и�˹��ȥ
        for (int k = 0;k < N;k++)
        {
            for (int j = k + 1;j < N;j++)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;
            for (int i = k + 1;i < N;i++)
            {
                for (int j = k + 1;j < N;j++)
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                m[i][k] = 0;
            }
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

        for(int i=0;i<N;i++)
            free(m[i]);

        free(m);
    }
	return 0;
}
