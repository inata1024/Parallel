//普通高斯消去 openMP
#include <iostream>
#include<cstdlib>
#include <chrono>
#include<random>
#include<omp.h>
#include<arm_neon.h>
#define NUM_THREADS 4
using namespace std;
using std::default_random_engine;
int N = 0;//矩阵规模
int main()
{
    int a[12] = { 128,256,384,512,640,768,896,1024,1280,1536,1792,2048 };
    for (int tms = 0;tms < 12;tms++)
    {
        N = a[tms];//指定矩阵规模
        //N=10;
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

        //声明变量
        int i, j, k;
        //创建线程
#pragma omp parallel num_threads(NUM_THREADS) private(i,j,k)
        {
            //串行高斯消去
            for (int k = 0;k < N;k++)
            {
#pragma omp single
                {
                    float32x4_t va, vt;
                    vt = vmovq_n_f32(m[k][k]);
                    int j = 0;
                    for (j = k + 1;j <= N - 4;j += 4)
                    {
                        va = vld1q_f32(m[k] + j);
                        //arm32不支持Neon浮点除法，用乘以倒数代替
                        //float32x4_t reciprocal = vrecpeq_f32(vt);
                        //reciprocal = vmulq_f32(vrecpsq_f32(vt, reciprocal), reciprocal);
                        //va = vmulq_f32(va,reciprocal);
                        va = vdivq_f32(va, vt);
                        vst1q_f32(m[k] + j, va);
                    }
                    //处理剩余下标
                    while (j < N)
                    {
                        m[k][j] = m[k][j] / m[k][k];
                        j++;
                    }

                    m[k][k] = 1.0;
                }

                //循环调度策略 dynamic效果最好 static没用
#pragma omp for
                for (int i = k + 1;i < N;i++)
                {
                    float32x4_t vaik, vakj, vaij, vx;
                    vaik = vld1q_dup_f32(m[i] + k);
                    for (j = k + 1;j <= N - 4;j += 4)
                    {
                        vakj = vld1q_f32(m[k] + j);
                        vaij = vld1q_f32(m[i] + j);
                        vx = vmulq_f32(vakj, vaik);
                        vaij = vsubq_f32(vaij, vx);
                        vst1q_f32(m[i] + j, vaij);
                    }
                    while (j < N)
                    {
                        m[i][j] = m[i][j] - m[k][j] * m[i][k];
                        j++;
                    }
                    m[i][k] = 0;
                }
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        cout << "矩阵规模：" << N << "  用时：" << fp_ms.count() << "ms" << endl;
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
