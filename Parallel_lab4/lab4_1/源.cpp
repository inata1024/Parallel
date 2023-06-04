//��ͨ��˹��ȥMPI 
#include<iostream>
#include <stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<omp.h>
#include<arm_neon.h>
using namespace std;
int N = 1024;//�����ģ
float** m;

//����
double MPI_Func0(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI��˹��ȥ
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

//�л���
double MPI_Func1(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI��˹��ȥ
    for (int k = 0;k < N;k++)
    {
        //ģ���� ����ȫ����0�Ž��� ��Ϊ����ǰ���뱣֤�ý��̵���һ�д���Ǹ��¹�������
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
        //����һ
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


//�п黮��
double MPI_Func2_0(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //ÿ�����̸���ķ�Χ������ҿ�
    int len = N / size, l = rank * len, h = (rank == size - 1) ? N : rank * len + len;//����hԽ��

    //MPI��˹��ȥ
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
        //���Խ��г���

        for (int j = l;j < h;j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;

        //���Խ�����ȥ
        for (int i = k + 1;i < N;i++)
        {
            //��ӵ�жԽ���Ԫ�ص��̣߳�����Ӧ�ô�k + 1��ʼ������һ��ʼ�Ͱ�k+1������0�������û������ 
            //�����ϸ��
            for (int j = (rank == root) ? k + 1 : l;j < h;j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
    //���ܵ�0�Ž���
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

//�п黮��
double MPI_Func2(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //ÿ�����̸���ķ�Χ������ҿ�
    int len = N / size, l = rank * len, h = (rank == size - 1) ? N : rank * len + len;//����hԽ��
    float* load = new float[N];
    //MPI��˹��ȥ
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
        //���Խ��г���

        for (int j = l;j < h;j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;

        //���Խ�����ȥ
        for (int i = k + 1;i < N;i++)
        {
            //��ӵ�жԽ���Ԫ�ص��̣߳�����Ӧ�ô�k + 1��ʼ������һ��ʼ�Ͱ�k+1������0�������û������ 
            //�����ϸ��
            for (int j = (rank == root) ? k + 1 : l;j < h;j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
    //���ܵ�0�Ž���
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


//��ˮ���㷨
double MPI_Func3(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI��˹��ȥ
    for (int k = 0;k < N;k++)
    {
        if (rank == k % size)
        {
            for (int j = k + 1;j < N;j++)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;

            //��Ե�ת��
            MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
        }
        else
        {
            //���ﲻ��ֱ����rank - 1����Ϊģ������Ǹ���
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


//��ˮ���㷨 + openMP
double MPI_Func4(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI��˹��ȥ
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

                //��Ե�ת��
                MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
            }
            else
            {
                //���ﲻ��ֱ����rank - 1����Ϊģ������Ǹ���
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
        //���ù㲥��ȥ���
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;
}

//��ˮ���㷨 + neon
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
                //arm32��֧��Neon����������ó��Ե�������
                //float32x4_t reciprocal = vrecpeq_f32(vt);
                //reciprocal = vmulq_f32(vrecpsq_f32(vt, reciprocal), reciprocal);
                //va = vmulq_f32(va,reciprocal);
                va = vdivq_f32(va, vt);
                vst1q_f32(m[k] + j, va);
            }
            //����ʣ���±�
            while (j < N)
            {
                m[k][j] = m[k][j] / m[k][k];
                j++;
            }

            m[k][k] = 1.0;

            //��Ե�ת��
            MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
        }
        else
        {
            //���ﲻ��ֱ����rank - 1����Ϊģ������Ǹ���
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

//��ˮ���㷨 + openMP + neon
double MPI_Func6(int rank, int size, float** m)
{
    double start = MPI_Wtime();

    //MPI��˹��ȥ
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
                    //arm32��֧��Neon����������ó��Ե�������
                    //float32x4_t reciprocal = vrecpeq_f32(vt);
                    //reciprocal = vmulq_f32(vrecpsq_f32(vt, reciprocal), reciprocal);
                    //va = vmulq_f32(va,reciprocal);
                    va = vdivq_f32(va, vt);
                    vst1q_f32(m[k] + j, va);
                }
                //����ʣ���±�
                while (j < N)
                {
                    m[k][j] = m[k][j] / m[k][k];
                    j++;
                }

                m[k][k] = 1.0;

                //��Ե�ת��
                MPI_Send((void*)m[k], N, MPI_FLOAT, (rank + 1) % size, k, MPI_COMM_WORLD);
            }
            else
            {
                //���ﲻ��ֱ����rank - 1����Ϊģ������Ǹ���
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
    N = 1024;//ָ�������ģ
    //���ɲ�������
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
        printf("�����ģ:%d ʱ��: % fms\n", a[tms], time * 1000);
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
