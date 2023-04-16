//普通高斯消去x86平台SSE
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
#include <chrono>
#include<random>
using namespace std;
using std::default_random_engine;
int N = 0;//矩阵规模
int main()
{
    int a[12] = { 128,256,384,512,640,768,896,1024,1280,1536,1792,2048 };
    for (int tms = 0;tms < 12;tms++)
    {
        N = a[tms];//指定矩阵规模

        //生成测试数据
        default_random_engine e;
        srand(1);
        float** m = new float* [N];
        for (int i = 0;i < N;i++)
        {
            m[i] = new float[N];
            for (int j = 0;j < N;j++)
                m[i][j] = 0;
        }
        for (int i = 0;i < N;i++)
        {
            m[i][i] = 1.0;
            for (int j = i + 1;j < N;j++)
                m[i][j] = rand();
        }
        for (int k = 0;k < N;k++)
            for (int i = k + 1;i < N;i++)
                for (int j = 0;j < N;j++)
                    m[i][j] += m[k][j];

        auto t1 = std::chrono::high_resolution_clock::now();
        __m128 va, vt;
        for (int k = 0;k < N;k++)
        {
            //不对齐算法策略
            vt = _mm_set_ps(m[k][k], m[k][k], m[k][k], m[k][k]);
            int j = 0;
            for (j = k + 1;j <= N - 4;j += 4)
            {
                va = _mm_loadu_ps(m[k] + j);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(m[k] + j, va);
            }
            //处理剩余下标
            while (j < N)
            {
                m[k][j] = m[k][j] / m[k][k];
                j++;
            }

            m[k][k] = 1.0;
            for (int i = k + 1;i < N;i++)
            {
                __m128 vaik, vakj, vaij, vx;
                vaik = _mm_set_ps(m[i][k], m[i][k], m[i][k], m[i][k]);
                for (j = k + 1;j <= N - 4;j += 4)
                {
                    vakj = _mm_loadu_ps(m[k] + j);
                    vaij = _mm_loadu_ps(m[i] + j);
                    vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_storeu_ps(m[i] + j, vaij);
                }
                while (j < N)
                {
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                    j++;
                }
                m[i][k] = 0;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        cout << "矩阵规模：" << N << "  用时：" << fp_ms.count() << "ms" << endl;
    }
    return 0;
}
