//��ͨ��˹��ȥ openMP
#include <iostream>
#include<cstdlib>
#include <chrono>
#include<random>
#include<omp.h>
#define NUM_THREADS 8
using namespace std;
using std::default_random_engine;
int N=0;//�����ģ
int main()
{
    int a[12]={128,256,384,512,640,768,896,1024,1280,1536,1792,2048};
    for(int tms=0;tms<12;tms++)
    {
        N=a[tms];//ָ�������ģ
        //N=10;
        //���ɲ�������
        default_random_engine e;
        srand(1);
        float **m=new float*[N];
        for(int i=0;i<N;i++)
        {
            m[i]=new float[N];
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

        auto t1 = std::chrono::high_resolution_clock::now();

        //��������
        int i,j,k;
        //�����߳�
        #pragma omp parallel num_threads(NUM_THREADS) private(i,j,k)
        {
            for(int k=0;k<N;k++)
            {
                #pragma omp single
                {
                    for(int j=k+1;j<N;j++)
                        m[k][j]=m[k][j]/m[k][k];
                    m[k][k]=1.0;
                }

                //ѭ�����Ȳ��� dynamicЧ����� staticû��
                #pragma omp for
                for(int i=k+1;i<N;i++)
                {
                    for(int j=k+1;j<N;j++)
                        m[i][j]=m[i][j]-m[k][j]*m[i][k];
                    m[i][k]=0;
                }
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        cout << "�����ģ��"<<N<<"  ��ʱ��"<<fp_ms.count() << "ms" << endl;
//        for(int k=0;k<N;k++)
//        {
//            for(int j=0;j<N;j++)
//            {
//                printf("%10.2f ",m[k][j]);
//            }
//            printf("\n");
//        }
    }
	return 0;
}