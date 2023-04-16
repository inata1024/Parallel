//普通高斯消去x86平台AVX512
//本机不支持AVX512
#include <iostream>
#include<cstdlib>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
using namespace std;
int N=1000;//矩阵规模
LARGE_INTEGER frequency;
int main()
{
    //生成测试数据
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
//    //打印消元前矩阵
//    for(int k=0;k<N;k++)
//    {
//        for(int j=0;j<N;j++)
//            cout<<m[k][j]<<" ";
//        cout<<endl;
//    }


    //时间测量
    double dff, begin_, _end, time;
	QueryPerformanceFrequency(&frequency);//获得时钟频率
	dff = (double)frequency.QuadPart;//取得频率
	QueryPerformanceCounter(&frequency);
	begin_ = frequency.QuadPart;//获得初始值

	//SSE高斯消去
	__m512 va,vt;
	for(int k=0;k<N;k++)
    {
        //不对齐算法策略
        vt=_mm512_set_ps(m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k]);
        int j=0;
        for(j=k+1;j<=N-16;j+=16)
        {
            va=_mm512_loadu_ps(m[k]+j);
            va=_mm512_div_ps(va,vt);
            _mm512_storeu_ps(m[k]+j,va);
        }
        //处理剩余下标
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
	_end = frequency.QuadPart;//获得终止值
	time = (_end - begin_) / dff;//差值除以频率得到时间(ms)
	cout << time *1000 << "ms" << endl;
//	for(int k=0;k<N;k++)
//    {
//        for(int j=0;j<N;j++)
//            cout<<m[k][j]<<" ";
//        cout<<endl;
//    }
	return 0;

}

