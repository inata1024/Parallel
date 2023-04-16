//普通高斯消去arm平台Neon 内联汇编
#include <iostream>
#include <chrono>
#include<arm_neon.h>
using namespace std;
int N = 3000;//矩阵规模
int main()
{
    int a[12] = { 128,256,384,512,640,768,896,1024,1280,1536,1792,2048 };
    for (int tms = 0;tms < 7;tms++)
    {
        N = a[tms];//指定矩阵规模
        //生成测试数据
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
        //时间测量
        auto t1 = std::chrono::high_resolution_clock::now();

        //Neon高斯消去
        for (int k = 0;k < N;k++)
        {
            int j = 0;
            for (j = k + 1;j <= N - 4;j += 4)
            {
                float* addr = m[k] + j;
                float* addr1 = m[k] + k;
                __asm__ __volatile__(
                    "mov x0, %[addr] \n"//加载内存地址
                    "ld1 {v0.4s}, [x0] \n" //从内存中装四个float到v0
                    "mov x0, %[addr1] \n"//加载被除数m[k][k]地址
                    "ld1r {v1.4s}, [x0] \n"//将被除数重复加载到v1 ld1r指令
                    "fdiv v2.4s,v0.4s,v1.4s \n "//向量相除
                    "mov x0, %[addr] \n"//加载内存地址
                    "st1 {v2.4s}, [x0] \n"//结果存入内存
                    :                                          // 输出寄存器列表
                : [addr] "r"(addr), [addr1]"r"(addr1)        // 输入寄存器列表
                    : "v0", "v1", "v2", "x0", "memory", "cc"       // 被破坏的寄存器列表
                    );
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
                for (int j = k + 1;j < N;j++)
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                m[i][k] = 0;
            }

        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        cout << fp_ms.count() << "ms" << endl;

    }
    return 0;
}
