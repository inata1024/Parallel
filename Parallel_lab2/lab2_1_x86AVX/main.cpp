//��ͨ��˹��ȥx86ƽ̨AVX
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
#include <chrono>
#include<random>
using namespace std;
using std::default_random_engine;
int N = 0;//�����ģ
int main()
{
    int a[12] = { 128,256,384,512,640,768,896,1024,1280,1536,1792,2048 };
    for (int tms = 0;tms < 12;tms++)
    {
        N = a[tms];//ָ�������ģ

        //���ɲ�������
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
        __m256 va, vt;
        for (int k = 0;k < N;k++)
        {
            //�������㷨����
            vt = _mm256_set_ps(m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]);
            int j = 0;
            for (j = k + 1;j <= N - 8;j += 8)
            {
                va = _mm256_loadu_ps(m[k] + j);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(m[k] + j, va);
            }
            //����ʣ���±�
            while (j < N)
            {
                m[k][j] = m[k][j] / m[k][k];
                j++;
            }

            m[k][k] = 1.0;
            for (int i = k + 1;i < N;i++)
            {
                __m256 vaik, vakj, vaij, vx;
                vaik = _mm256_set_ps(m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]);
                for (j = k + 1;j <= N - 8;j += 8)
                {
                    vakj = _mm256_loadu_ps(m[k] + j);
                    vaij = _mm256_loadu_ps(m[i] + j);
                    vx = _mm256_mul_ps(vakj, vaik);
                    vaij = _mm256_sub_ps(vaij, vx);
                    _mm256_storeu_ps(m[i] + j, vaij);
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
        cout << "�����ģ��" << N << "  ��ʱ��" << fp_ms.count() << "ms" << endl;
    }
    return 0;
}
