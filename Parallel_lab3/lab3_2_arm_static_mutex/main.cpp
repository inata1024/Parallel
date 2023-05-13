//arm特殊高斯消去 pthread 静态线程 mutex加锁
#include <stdlib.h>
#include<pthread.h>
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>
using namespace std;
//鲲鹏服务器八核
#define NUM_THREADS 8
//int C_raw = 1011, R1 = 539, R2 = 263, task_id = 0;//原始矩阵列数,非零消元子,被消元行,
//int C_raw = 130, R1 = 22, R2 = 8, task_id = 0;//原始矩阵列数,非零消元子,被消元行,

typedef struct {
    int t_id;
}threadParam_t;

//创建互斥锁
pthread_mutex_t amutex = PTHREAD_MUTEX_INITIALIZER;

//定义全局变量
int C_raw,R1,R2,C;//原始矩阵列数,非零消元子,被消元行,位向量矩阵列数
int **a,**b,*ini;//消元子、被消元行、记录消元子首项数组

//定义线程函数
void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id; //线程编号

    for (int r = t_id;r < R2;r+=NUM_THREADS)//对每个被消元行
    {
        for (int c = C - 1;c >= 0;c--)//每一列位向量
        {
            //位向量不为0才需要消去
            int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
            bool end = false;//当b[r]成为消元子时，end = true
            while (b[r][c] != 0 && !end)//从后往前消去
            {
                while(!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
                {
                    col--;
                    continue;
                }
                //当前线程加锁
                pthread_mutex_lock(&amutex);
                int temp = ini[32 * c + col];
                pthread_mutex_unlock(&amutex);
                if (temp > -1)//若存在消元行
                {
                    int* er = temp < R1 ? a[temp] : b[temp - R1];
                    for (int i = 0;i < C;i++)
                        b[r][i] ^= er[i];
                }
                else//否则将b[r]加入消元子
                {
                    //当前线程加锁
                    pthread_mutex_lock(&amutex);
                    ini[32 * c + col] = R1 + r;
                    end = true;
                    pthread_mutex_unlock(&amutex);

                }
                col--;
            }
            if (end)//若该行已进入消元子，则不再消元
                break;
        }
    }
    pthread_exit(NULL);
}

int main()
{
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
   "6_3799_2759_1953","7_8399_6375_4535", "8_23075_18748_14325","9_37960_29304_14291","10_43577_39477_54274","11_85401_5724_756"};
	struct Size {
		int a;
		int b;
		int c;//分别为矩阵列数，消元子个数和被消元行个数
	} fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 263}, {2362, 1226, 453},
	{3799, 2759, 1953},{8399, 6375, 4535},{23075,18748,14325},{37960,29304,14291},{43577,39477,54274} ,{85401,5724,756} };


	for (int tms = 0;tms < 7;tms++)
	{
		C_raw = fileSize[tms].a;
		R1 = fileSize[tms].b;
		R2 = fileSize[tms].c;

		C = ceil(C_raw / 32.0);//位向量矩阵列数
		//为消元子、被消元行分配空间
		a = new int* [R1];
        b = new int* [R2];
		ini = new int[C_raw];//记录该位置首项所在行数 在a中：0~R1-1 在b中：R1~R1+R2-1
		for (int i = 0;i < C_raw;i++)
			ini[i] = -1;
		for (int i = 0;i < R1;i++)
			a[i] = new int[C] {0};
		for (int i = 0;i < R2;i++)
			b[i] = new int[C] {0};
		//读入数据集并构建矩阵
		ifstream in1("/home/data/Groebner/" + Folders[tms] + "/1.txt");
		ifstream in2("/home/data/Groebner/" + Folders[tms] + "/2.txt");
		string str;
		int row = 0;//行数
		//处理每行数据
		while (getline(in1, str))
		{
			istringstream ss(str);
			string tmp;
			int count = 0;
			int n = 0;
			while (ss >> n) {
				a[row][n >> 5] += 1 << (31 - n % 32);//模操作可用按位与31代替
				if (!count)//第一次循环获取首项
					ini[n] = row;//该位置有消元首项，在row行
				count++;
			}
			row++;
		}
		row = 0;
		while (getline(in2, str))
		{
			istringstream ss(str);
			string tmp;
			int n = 0;
			while (ss >> n) {
				b[row][n >> 5] += 1 << (31 - n % 32);//模操作可用按位与31代替
			}
			row++;
		}

		auto t1 = std::chrono::high_resolution_clock::now();

        //创建线程
        pthread_t handles[NUM_THREADS];// 创建对应的Handle
        threadParam_t param[NUM_THREADS];// 创建对应的线程数据结构
        for(int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
        }


        for(int t_id=0;t_id<NUM_THREADS;++t_id)
            pthread_join(handles[t_id],NULL);


		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
		cout << tms << ": " << fp_ms.count() << "ms" << endl;

//        for (int r = 0;r < R2;r++)//对每个被消元行
//		{
//			for (int c = C - 1;c >= 0;c--)//每一列位向量
//			{
//				//位向量不为0才需要消去
//				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
//				while (b[r][c] != 0 && col >= 0)//从后往前消去
//				{
//					if (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
//					{
//						col--;
//						continue;
//					}
//					cout<<32*c+col<<" ";
//					col--;
//				}
//			}
//			cout<<endl;
//		}
	}
	pthread_mutex_destroy(&amutex);
	return 0;
}
