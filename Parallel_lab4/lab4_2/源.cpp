//特殊高斯消去MPI 
#include<iostream>
#include <stdio.h>
#include<fstream>
#include<sstream>
#include<stdlib.h>
#include<mpi.h>
#include<algorithm>
#include<math.h>
#include<omp.h>
//#include<arm_neon.h>
using namespace std;

int main(int argc, char* argv[])
{

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	//string Folders[] = { "测试样例1 矩阵列数130，非零消元子22，被消元行8", "测试样例2 矩阵列数254，非零消元子106，被消元行53",
	//					"测试样例3 矩阵列数562，非零消元子170，被消元行53",
	//					"测试样例4 矩阵列数1011，非零消元子539，被消元行263", "测试样例5 矩阵列数2362，非零消元子1226，被消元行453",
	//					"测试样例6 矩阵列数3799，非零消元子2759，被消元行1953","测试样例7 矩阵列数8399，非零消元子6375，被消元行4535",
	//					"测试样例8 矩阵列数23075，非零消元子18748，被消元行14325" };

	string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
   "6_3799_2759_1953","7_8399_6375_4535", "8_23075_18748_14325","9_37960_29304_14291","10_43577_39477_54274","11_85401_5724_756" };
	struct Size {
		int a;
		int b;
		int c;//分别为矩阵列数，消元子个数和被消元行个数
	} fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 263}, {2362, 1226, 453},
	{3799, 2759, 1953},{8399, 6375, 4535},{23075,18748,14325},{37960,29304,14291},{43577,39477,54274} ,{85401,5724,756} };
	int tms = 0;

	int C_raw = fileSize[tms].a, R1 = fileSize[tms].b, R2 = fileSize[tms].c;//原始矩阵列数,非零消元子,被消元行,

	int C = ceil(C_raw / 32.0);//位向量矩阵列数
	//为消元子、被消元行分配空间
	int** a = new int* [R1], ** b = new int* [R2];
	int* ini;

	MPI_Alloc_mem(C_raw * sizeof(int), MPI_INFO_NULL, &ini);
	//int* ini = new int[C_raw];//记录该位置首项所在行数 在a中：0~R1-1 在b中：R1~R1+R2-1
	for (int i = 0;i < C_raw;i++)
		ini[i] = -1;
	for (int i = 0;i < R1;i++)
		MPI_Alloc_mem(C * sizeof(int), MPI_INFO_NULL, &a[i]);

	//a[i] = new int[C] {0};
	for (int i = 0;i < R2;i++)
		MPI_Alloc_mem(C * sizeof(int), MPI_INFO_NULL, &b[i]);

	//b[i] = new int[C] {0};

//读入数据集并构建矩阵
//ifstream in1("D:/学习/并行程序设计/Groebner data/" + Folders[tms] + "/消元子.txt");
//ifstream in2("D:/学习/并行程序设计/Groebner data/" + Folders[tms] + "/被消元行.txt");
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
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();


	MPI_Win* win, win0;
	MPI_Alloc_mem(R2 * sizeof(MPI_Win), MPI_INFO_NULL, &win);
	MPI_Win_create(ini, C_raw * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win0);
	for (int i = 0;i < R2;i++)
	{
		MPI_Win_create(b[i], C * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win[i]);
	}
	//将0号线程数据共享

	//每个线程处理一些被消元行，每一轮循环进行
	for (int r = 0;r < R2;r++)//对每个被消元行
	{
		if (rank == 0 || rank - 1 != r % (size - 1))
			continue;
		bool end = false;//当b[r]成为消元子时，end = true
		for (int c = C - 1;c >= 0;c--)//每一列位向量
		{
			//位向量不为0才需要消去
			int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标

			while (b[r][c] != 0 && !end)//从后往前消去
			{
				while (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
				{
					col--;
					continue;
				}
				//实现思路：使用RMA lock/unlock 实现类似openMP的共享内存
				//共享0号进程的数据
				//每个进程负责自己的行
				int temp = 0;
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win0);
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win[r]);
				//int temp = ini[32 * c + col];
				MPI_Get(&temp, 1, MPI_INT, 0, (32 * c + col), 1, MPI_INT, win0);
				//MPI_Get_accumulate(&temp, 1, MPI_INT, &temp, 1, MPI_INT, 0, (32 * c + col), 1, MPI_INT, MPI_NO_OP, win0);
				MPI_Win_flush(0, win0);

				if (temp == -1)//将b[r]加入消元子
				{
					int temp_r = R1 + r;
					MPI_Put(&temp_r, 1, MPI_INT, 0, (32 * c + col), 1, MPI_INT, win0);
					//MPI_Accumulate(&temp_r, 1, MPI_INT, 0, (32 * c + col), 1, MPI_INT, MPI_REPLACE, win0);
					MPI_Win_flush(0, win0);
					//ini[32 * c + col] = R1 + r;
					MPI_Put(b[r], C, MPI_INT, 0, (C_raw + r * C), C, MPI_INT, win[r]);
					//MPI_Accumulate(b[r], C, MPI_INT, 0, (C_raw + r * C), C, MPI_INT, MPI_REPLACE, win[r]);
					MPI_Win_flush(0, win[r]);
					end = true;
				}
				MPI_Win_unlock(0, win0);
				MPI_Win_unlock(0, win[r]);

				if (temp > -1)//若存在消元行
				{
					int* er = nullptr;
					if (temp < R1)
						er = a[temp];
					else
					{
						er = new int[C];
						MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win[temp - R1]);
						MPI_Get(er, C, MPI_INT, 0, (C_raw + (temp - R1) * C), C, MPI_INT, win[temp - R1]);
						MPI_Win_unlock(0, win[temp - R1]);

						//MPI_Get_accumulate(er, C, MPI_INT, er, C, MPI_INT, 0, (C_raw + (temp - R1) * C), C, MPI_INT, MPI_NO_OP, win[temp - R1]);

					}
					//int* er = temp < R1 ? a[temp] : b[temp - R1];
					for (int i = 0;i < C;i++)
						b[r][i] ^= er[i];
				}

				col--;
			}
			if (end)//若该行已进入消元子，则不再消元
				break;
		}
		//消去完成的被消元行，发给0号进程
		if (!end)
		{
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win[r]);
			MPI_Put(b[r], C, MPI_INT, 0, (C_raw + r * C), C, MPI_INT, win[r]);
			MPI_Win_unlock(0, win[r]);

			//MPI_Accumulate(b[r], C, MPI_INT, 0, (C_raw + r * C), C, MPI_INT, MPI_REPLACE, win[r]);
		}

	}

	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Win_free(&win0);
	//for (int i = 0;i < R2;i++)
	//	MPI_Win_free(&win[i]);

	double end = MPI_Wtime();
	if (rank == 0)
	{
		printf("矩阵规模:%d 时间: % fms\n", a[tms], (end - start) * 1000);
		//for (int r = 0;r < R2;r++)//对每个被消元行
		//{
		//	for (int c = C - 1;c >= 0;c--)//每一列位向量
		//	{
		//		//位向量不为0才需要消去
		//		int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
		//		while (b[r][c] != 0 && col >= 0)//从后往前消去
		//		{
		//			if (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
		//			{
		//				col--;
		//				continue;
		//			}
		//			cout<<32*c+col<<" ";
		//			col--;
		//		}
		//	}
		//	cout<<endl;
		//}
	}


	MPI_Finalize();
	return 0;
}

