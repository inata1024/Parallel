//普通高斯消去串行unroll算法
#include <iostream>
#include<cstdlib>
#include <chrono>
#include<random>
using namespace std;
using std::default_random_engine;
int N = 0;//矩阵规模
int main()
{
    int a[12] = { 128,256,384,512,640,768,896,1024,1280,1536,1792,2048 };
    for (int tms = 0;tms < 7;tms++)
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
        //串行unroll高斯消去
        for (int k = 0;k < N;k++)
        {
            int j = 0;
            for (j = k + 1;j < N - 1;j += 2)
            {
                m[k][j] = m[k][j] / m[k][k];
                m[k][j + 1] = m[k][j + 1] / m[k][k];
            }
            if (j == N - 1)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;
            for (int i = k + 1;i < N;i++)
            {
                for (j = k + 1;j < N - 1;j += 2)
                {
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                    m[i][j + 1] = m[i][j + 1] - m[k][j + 1] * m[i][k];
                }
                if (j == N - 1)
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];

                m[i][k] = 0;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        cout << "矩阵规模：" << N << "  用时：" << fp_ms.count() << "ms" << endl;
    }
    return 0;
}
