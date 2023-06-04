//普通高斯消去MPI 
#include<iostream>
#include <stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<omp.h>
#include<arm_neon.h>
using namespace std;
int N = 1024;//矩阵规模
float** m;

//串行
double MPI_Func0(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI高斯消去
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
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;
}

//行划分
double MPI_Func1(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI高斯消去
    for (int k = 0;k < N;k++)
    {
        //模划分 不能全划给0号进程 因为除法前必须保证该进程的这一行存的是更新过的数据
        if (rank == k % size)
        {
            for (int j = k + 1;j < N;j++)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;
            for (int i = 0;i < size;i++)
            {
                if (i == rank)
                    continue;
                MPI_Send((void*)m[k], N, MPI_FLOAT, i, k, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv((void*)m[k], N, MPI_FLOAT, k % size, k, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }
        //方法一
        //MPI_Bcast((void*)m[k], N, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (int i = k + 1;i < N;i++)
        {
            if (rank == i % size)
            {
                for (int j = k + 1;j < N;j++)
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                m[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;

}


//列块划分
double MPI_Func2_0(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //每个进程负责的范围，左闭右开
    int len = N / size, l = rank * len, h = (rank == size - 1) ? N : rank * len + len;//避免h越界

    //MPI高斯消去
    for (int k = 0;k < N;k++)
    {

        //for (int j = k;j < N;j++)
        //{
        //    MPI_Bcast((void*)(m[j] + k), 1, MPI_FLOAT, (k / len) >= size ? size - 1 : k / len, MPI_COMM_WORLD);
        //    MPI_Barrier(MPI_COMM_WORLD);
        //}
        //int* load = new int[N - k];
        int root = (k / len) >= size ? size - 1 : k / len;
        if (rank == root)
        {
            for (int i = 0;i < size;i++)
            {
                if (i == rank)
                    continue;
                for (int j = k;j < N;j++)
                    MPI_Send((void*)(m[j] + k), 1, MPI_FLOAT, i, j, MPI_COMM_WORLD);

                //for (int j = k;j < N;j++)
                //    load[j] = m[j][k];
                //MPI_Send((void*)load, N - k, MPI_FLOAT, i, k, MPI_COMM_WORLD);
            }
        }
        else
        {
            //MPI_Recv((void*)load, N - k, MPI_FLOAT, root, k, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            //for (int j = k;j < N;j++)
            //    m[j][k] = load[j];
            for (int j = k;j < N;j++)
                MPI_Recv((void*)(m[j] + k), 1, MPI_FLOAT, root, j, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        }
        MPI_Barrier(MPI_COMM_WORLD);
        //各自进行除法

        for (int j = l;j < h;j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;

        //各自进行消去
        for (int i = k + 1;i < N;i++)
        {
            //对拥有对角线元素的线程，这里应该从k + 1开始，否则一开始就把k+1列消成0，后面就没得消了 
            //这里很细节
            for (int j = (rank == root) ? k + 1 : l;j < h;j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
    //汇总到0号进程
    if (rank == 0)
    {
        for (int i = 1;i < size;i++)
        {
            int i_l = i * len, i_h = (i == size - 1) ? N : i * len + len;
            for (int j = 0;j < N;j++)
            {
                MPI_Recv((void*)(m[j] + i_l), i_h - i_l, MPI_FLOAT, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else
    {
        for (int i = 0;i < N;i++)
        {
            MPI_Send((void*)(m[i] + l), h - l, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;

}

//列块划分
double MPI_Func2(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //每个进程负责的范围，左闭右开
    int len = N / size, l = rank * len, h = (rank == size - 1) ? N : rank * len + len;//避免h越界
    float* load = new float[N];
    //MPI高斯消去
    for (int k = 0;k < N;k++)
    {
        //for (int j = k;j < N;j++)
        //{
        //    MPI_Bcast((void*)(m[j] + k), 1, MPI_FLOAT, (k / len) >= size ? size - 1 : k / len, MPI_COMM_WORLD);
        //    MPI_Barrier(MPI_COMM_WORLD);
        //}
        int root = (k / len) >= size ? size - 1 : k / len;
        if (rank == root)
        {
            for (int i = 0;i < size;i++)
            {
                if (i == rank)
                    continue;
 /*               for (int j = k;j < N;j++)
                    MPI_Send((void*)(m[j] + k), 1, MPI_FLOAT, i, j, MPI_COMM_WORLD);*/
                
                for (int j = k;j < N;j++)
                    load[j] = m[j][k];
                MPI_Send((void*)load, N - k, MPI_FLOAT, i, k, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv((void*)load, N - k, MPI_FLOAT, root, k, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            for (int j = k;j < N;j++)
                m[j][k] = load[j];
            //for (int j = k;j < N;j++)
            //    MPI_Recv((void*)(m[j] + k), 1, MPI_FLOAT, root, j, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        }
        MPI_Barrier(MPI_COMM_WORLD);
        //各自进行除法

        for (int j = l;j < h;j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;

        //各自进行消去
        for (int i = k + 1;i < N;i++)
        {
            //对拥有对角线元素的线程，这里应该从k + 1开始，否则一开始就把k+1列消成0，后面就没得消了 
            //这里很细节
            for (int j = (rank == root) ? k + 1 : l;j < h;j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
    //汇总到0号进程
    if (rank == 0)
    {
        for (int i = 1;i < size;i++)
        {
            int i_l = i * len, i_h = (i == size - 1) ? N : i * len + len;
            for (int j = 0;j < N;j++)
            {
                MPI_Recv((void*)(m[j] + i_l), i_h - i_l, MPI_FLOAT, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else
    {
        for (int i = 0;i < N;i++)
        {
            MPI_Send((void*)(m[i] + l), h - l, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;

}


//流水线算法
double MPI_Func3(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI高斯消去
    for (int k = 0;k < N;k++)
    {
        if (rank == k % size)
        {
            for (int j = k + 1;j < N;j++)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;

            //点对点转发
            MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
        }
        else
        {
            //这里不能直接用rank - 1，因为模完可能是负数
            MPI_Recv((void*)m[k], N, MPI_FLOAT, (rank + size - 1) % size, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if ((rank + 1) % size != k % size)
                MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
        }

        for (int i = k + 1;i < N;i++)
        {
            if (rank == i % size)
            {
                for (int j = k + 1;j < N;j++)
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                m[i][k] = 0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;

}


//流水线算法 + openMP
double MPI_Func4(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI高斯消去
#pragma omp parallel num_threads(4)
    for (int k = 0;k < N;k++)
    {
#pragma omp single
        {
            if (rank == k % size)
            {
                for (int j = k + 1;j < N;j++)
                    m[k][j] = m[k][j] / m[k][k];
                m[k][k] = 1.0;

                //点对点转发
                MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
            }
            else
            {
                //这里不能直接用rank - 1，因为模完可能是负数
                MPI_Recv((void*)m[k], N, MPI_FLOAT, (rank + size - 1) % size, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if ((rank + 1) % size != k % size)
                    MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
            }
        }
#pragma omp for
        for (int i = k + 1;i < N;i++)
        {
            if (rank == i % size)
            {
                for (int j = k + 1;j < N;j++)
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                m[i][k] = 0;
            }
        }
        //不用广播消去结果
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;
}

//流水线算法 + neon
double MPI_Func5(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    for (int k = 0;k < N;k++)
    {
    
        if (rank == k % size)
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

            //点对点转发
            MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
        }
        else
        {
            //这里不能直接用rank - 1，因为模完可能是负数
            MPI_Recv((void*)m[k], N, MPI_FLOAT, (rank + size - 1) % size, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if ((rank + 1) % size != k % size)
                MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
        }
        
        for (int i = k + 1;i < N;i++)
        {
            if (rank == i % size)
            {
                int j = 0;
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

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;
}

//流水线算法 + openMP + neon
double MPI_Func6(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI高斯消去
#pragma omp parallel num_threads(4)
    for (int k = 0;k < N;k++)
    {
#pragma omp single
        {
            if (rank == k % size)
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

                //点对点转发
                MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
            }
            else
            {
                //这里不能直接用rank - 1，因为模完可能是负数
                MPI_Recv((void*)m[k], N, MPI_FLOAT, (rank + size - 1) % size, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if ((rank + 1) % size != k % size)
                    MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
            }
        }

#pragma omp for
        for (int i = k + 1;i < N;i++)
        {
            if (rank == i % size)
            {
                int j = 0;
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

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;

}
int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int a[3] = { 512,1024,2048 }, tms = 1;
    N = 1024;//指定矩阵规模
    //生成测试数据
    float** m = (float**)malloc(N * sizeof(float*));
    for (int i = 0;i < N;i++)
    {
        m[i] = (float*)malloc(N * sizeof(float));
        for (int j = 0;j < N;j++)
            m[i][j] = 0;
    }
    for (int i = 0;i < N;i++)
    {
        m[i][i] = 1.0;
        for (int j = i + 1;j < N;j++)
            m[i][j] = rand() % 1000;
    }
    for (int k = 0;k < N;k++)
        for (int i = k + 1;i < N;i++)
            for (int j = 0;j < N;j++)
                m[i][j] += m[k][j];

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Func5(rank, size, m);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("矩阵规模:%d 时间: % fms\n", a[tms], time * 1000);
        //for (int k = 0;k < N;k++)
        //{
        //    for (int j = 0;j < N;j++)
        //    {
        //        printf("%10.2f ", m[k][j]);
        //    }
        //    printf("\n");
        //}
    }

    

    MPI_Finalize();
    return 0;
}
