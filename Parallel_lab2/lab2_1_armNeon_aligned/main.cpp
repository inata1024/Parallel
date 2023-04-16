//��ͨ��˹��ȥarmƽ̨Neon ����
#include <iostream>
#include <chrono>
#include<arm_neon.h>
using namespace std;
int N = 3000;//�����ģ
int main()
{
    int a[12] = { 128,256,384,512,640,768,896,1024,1280,1536,1792,2048 };
    for (int tms = 0;tms < 7;tms++)
    {
        N = a[tms];//ָ�������ģ
        //���ɲ�������
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
        //ʱ�����
        auto t1 = std::chrono::high_resolution_clock::now();

        //Neon��˹��ȥ
        float32x4_t va, vt;
        for (int k = 0;k < N;k++)
        {
            //�����㷨����
            vt = vmovq_n_f32(m[k][k]);
            int j = k + 1;
            //���д�������߽�
            while (j % 4 != 0)
            {
                m[k][j] = m[k][j] / m[k][k];
                j++;
            }
            for (;j <= N - 4;j += 4)
            {
                va = vld1q_f32(m[k] + j);//��ȡ����4�������ȸ������������Ĵ���

                //arm32��֧��Neon����������ó��Ե�������
                //float32x4_t reciprocal = vrecpeq_f32(vt);
                //reciprocal = vmulq_f32(vrecpsq_f32(vt, reciprocal), reciprocal);
                //va = vmulq_f32(va,reciprocal);
                va = vdivq_f32(va, vt);//arm64֧��
                vst1q_f32(m[k] + j, va);
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
                float32x4_t vaik, vakj, vaij, vx;
                vaik = vld1q_dup_f32(m[i] + k);
                j = k + 1;
                while (j % 4 != 0)
                {
                    m[i][j] = m[i][j] - m[k][j] * m[i][k];
                    j++;
                }
                for (;j <= N - 4;j += 4)
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

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        cout << fp_ms.count() << "ms" << endl;

    }
    return 0;
}
