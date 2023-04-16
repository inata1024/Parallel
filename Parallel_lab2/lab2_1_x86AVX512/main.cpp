//��ͨ��˹��ȥx86ƽ̨AVX512
//������֧��AVX512
#include <iostream>
#include<cstdlib>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2
using namespace std;
int N=1000;//�����ģ
LARGE_INTEGER frequency;
int main()
{
    //���ɲ�������
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
//    //��ӡ��Ԫǰ����
//    for(int k=0;k<N;k++)
//    {
//        for(int j=0;j<N;j++)
//            cout<<m[k][j]<<" ";
//        cout<<endl;
//    }


    //ʱ�����
    double dff, begin_, _end, time;
	QueryPerformanceFrequency(&frequency);//���ʱ��Ƶ��
	dff = (double)frequency.QuadPart;//ȡ��Ƶ��
	QueryPerformanceCounter(&frequency);
	begin_ = frequency.QuadPart;//��ó�ʼֵ

	//SSE��˹��ȥ
	__m512 va,vt;
	for(int k=0;k<N;k++)
    {
        //�������㷨����
        vt=_mm512_set_ps(m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k]);
        int j=0;
        for(j=k+1;j<=N-16;j+=16)
        {
            va=_mm512_loadu_ps(m[k]+j);
            va=_mm512_div_ps(va,vt);
            _mm512_storeu_ps(m[k]+j,va);
        }
        //����ʣ���±�
        while(j<N)
        {
            m[k][j]=m[k][j]/m[k][k];
            j++;
        }

        m[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            __m512 vaik,vakj,vaij,vx;
            vaik=_mm512_set_ps(m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k]);
            for(j=k+1;j<=N-16;j+=16)
            {
                vakj=_mm512_loadu_ps(m[k]+j);
                vaij=_mm512_loadu_ps(m[i]+j);
                vx=_mm512_mul_ps(vakj,vaik);
                vaij=_mm512_sub_ps(vaij,vx);
                _mm512_storeu_ps(m[i]+j,vaij);
            }
            while(j<N)
            {
                m[i][j]=m[i][j]-m[k][j]*m[i][k];
                j++;
            }
            m[i][k]=0;
        }
    }

	QueryPerformanceCounter(&frequency);
	_end = frequency.QuadPart;//�����ֵֹ
	time = (_end - begin_) / dff;//��ֵ����Ƶ�ʵõ�ʱ��(ms)
	cout << time *1000 << "ms" << endl;
//	for(int k=0;k<N;k++)
//    {
//        for(int j=0;j<N;j++)
//            cout<<m[k][j]<<" ";
//        cout<<endl;
//    }
	return 0;

}

