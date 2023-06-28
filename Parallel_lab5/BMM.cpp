%%writefile lab/vector_add.cpp
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//普通高斯消去GPU Buffer Memory Model
#include <sycl/sycl.hpp>
#include <iostream>
#include<cstdlib>
#include <chrono>
#include<random>
using std::default_random_engine;
using namespace sycl;
int const N=10;//矩阵规模
int main()
{
    int a[12]={128,256,384,512,640,768,896,1024,1280,1536,1792,2048};
    queue q;
    for(int tms=0;tms<1;tms++)
    {
        //N=a[tms];//指定矩阵规模
        //N=10;
        //生成测试数据
        std::default_random_engine e;
        std::srand(1);
        buffer<float,2> m_buf(range(N,N));
        host_accessor m_host(m_buf,read_write);
        for(int i=0;i<N;i++)
        {
            //m[i]=new float[N];
            for(int j=0;j<N;j++)
                m_host[i][j]=0;
        }
        for(int i=0;i<N;i++)
        {
            m_host[i][i]=1.0;
            for(int j=i+1;j<N;j++)
                m_host[i][j]=rand()%1000;
        }
        for(int k=0;k<N;k++)
            for(int i=k+1;i<N;i++)
                for(int j=0;j<N;j++)
                    m_host[i][j]+=m_host[k][j];
        
        printf("OK\n");
        auto t1 = std::chrono::high_resolution_clock::now();
        //GPU高斯消去
        buffer<int,1> k_buf(range(1));
        for(int k=0;k<N-2;k++)
        {
            //除法操作
            host_accessor k_host(k_buf,read_write);
            k_host[0]=k;
            q.submit([&](auto &h){
                accessor m(m_buf,h,read_write);
                accessor acc_k(k_buf,h,read_only);
                int k=acc_k[0];
                q.parallel_for(N-k-1,[=](auto index){
                    m[k][index+k+1]=m[k][index+k+1]/m[k][k];
                }).wait();
                m[k][k]=1.0;
            }).wait();

            printf("OK\n");
            
            q.submit([&](auto &h){
                accessor m(m_buf,h,read_write);
                accessor acc_k(k_buf,h,read_only);
                int k=acc_k[0];
                q.parallel_for(range(N-k-1,N-k-2),[=](auto index){
                    int row=index[0]+k+1;
                    int col=index[1]+k+2;
                
                    m[row][col]=m[row][col]-m[k][col]*m[row][k];
                    //for(int j=k+1;j<N;j++)
                    //    m[i+k+1][j]=m[i+k+1][j]-m[k][j]*m[i+k+1][k];
                    m[row][k]=0;
                }).wait();
                
            });
            
        }
        for (int k = 0;k < N;k++)
        {
          for (int j = 0;j < N;j++)
            {
                printf("%10.2f ", m_host[k][j]);
            }
           printf("\n");
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        printf("矩阵规模：%d 用时：%f ms",N,fp_ms.count());
        //std::cout << "矩阵规模："<<N<<"  用时："<<fp_ms.count() << "ms" << endl;
    }
	return 0;
}