//普通高斯消去串行
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include<pthread.h>
LARGE_INTEGER frequency;

float **m;//主函数动态分配
int N=0;

int main(){

    int a[12]={128,256,384,512,640,768,896,1024,1280,1536,1792,2048};
    for(int tms=0;tms<7;tms++)
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

        //串行高斯消去
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
