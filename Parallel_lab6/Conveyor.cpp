//特殊高斯消去 MPI Conveyor算法
//本来就是对齐的
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>
#include<mpi.h>
using namespace std;
#define NUM_THREADS 4
//int C_raw = 1011, R1 = 539, R2 = 263, task_id = 0;//原始矩阵列数,非零消元子,被消元行,
//int C_raw = 130, R1 = 22, R2 = 8, task_id = 0;//原始矩阵列数,非零消元子,被消元行,
int main(int argc, char* argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	string Folders[] = { "测试样例1 矩阵列数130，非零消元子22，被消元行8", "测试样例2 矩阵列数254，非零消元子106，被消元行53",
						"测试样例3 矩阵列数562，非零消元子170，被消元行53",
						"测试样例4 矩阵列数1011，非零消元子539，被消元行263", "测试样例5 矩阵列数2362，非零消元子1226，被消元行453",
						"测试样例6 矩阵列数3799，非零消元子2759，被消元行1953","测试样例7 矩阵列数8399，非零消元子6375，被消元行4535",
						"测试样例8 矩阵列数23075，非零消元子18748，被消元行14325" };
	//string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
 //  "6_3799_2759_1953","7_8399_6375_4535", "8_23075_18748_14325","9_37960_29304_14291","10_43577_39477_54274","11_85401_5724_756" };
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
	int* ini = new int[C_raw];//记录该位置首项所在行数 在a中：0~R1-1 在b中：R1~R1+R2-1
	for (int i = 0;i < C_raw;i++)
		ini[i] = -1;
	for (int i = 0;i < R1;i++)
		a[i] = new int[C] {0};
	for (int i = 0;i < R2;i++)
		b[i] = new int[C] {0};
	//读入数据集并构建矩阵
	//ifstream in1("/home/data/Groebner/" + Folders[tms] + "/1.txt");
	//ifstream in2("/home/data/Groebner/" + Folders[tms] + "/2.txt");
	ifstream in1("D:/学习/并行程序设计/Groebner data/" + Folders[tms] + "/消元子.txt");
	ifstream in2("D:/学习/并行程序设计/Groebner data/" + Folders[tms] + "/被消元行.txt");
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
	//任务划分
	int block_size = R2 / NUM_THREADS;
	int l = rank * block_size, h = rank == NUM_THREADS - 1 ? R2 : (rank + 1) * block_size;
	block_size = l - h;

	for (int r = l;r < h;r++)//对所负责的每个被消元行
	{
		for (int c = C - 1;c >= 0;c--)//每一列位向量
		{
			//位向量不为0才需要消去
			int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
			bool end = false;//当b[r]成为消元子时，end = true
			while (b[r][c] != 0 && !end)//从后往前消去
			{
				if (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
				{
					col--;
					continue;
				}
				int temp = ini[32 * c + col];
				if (temp > -1)//若存在消元行
				{
					int* er = temp < R1 ? a[temp] : b[temp - R1];
					for (int i = 0;i < C;i++)
						b[r][i] ^= er[i];
				}
				else//否则将b[r]加入消元子
				{
					ini[32 * c + col] = R1 + r;
					end = true;
				}
				col--;
			}
			if (end)//若该行已进入消元子，则不再消元
				break;
		}
	}

	//第二轮消去
	if (rank == 0)
	{
		for (int i = 1;i < NUM_THREADS;i++)
		{
			//算出进程i负责的范围
			int block_size = R2 / NUM_THREADS;
			int l = i * block_size, h = i == NUM_THREADS - 1 ? R2 : (i + 1) * block_size;
			block_size = l - h;
			//接收第i个进程的消去结果
			for (int j = l;j < h;j++)
				MPI_Recv(b[j], C, MPI_INT, i, j, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
		//对于0号进程的第二轮消去，从h开始消去即可
		//即用P_1到P_{i-1}消去P_i
		for (int r = h;r < R2;r++)//对所负责的每个被消元行
		{
			for (int c = C - 1;c >= 0;c--)//每一列位向量
			{
				//位向量不为0才需要消去
				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
				bool end = false;//当b[r]成为消元子时，end = true
				while (b[r][c] != 0 && !end)//从后往前消去
				{
					if (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
					{
						col--;
						continue;
					}
					int temp = ini[32 * c + col];
					if (temp > -1)//若存在消元行
					{
						int* er = temp < R1 ? a[temp] : b[temp - R1];
						for (int i = 0;i < C;i++)
							b[r][i] ^= er[i];
					}
					else//否则将b[r]加入消元子
					{
						ini[32 * c + col] = R1 + r;
						end = true;
					}
					col--;
				}
				if (end)//若该行已进入消元子，则不再消元
					break;
			}
		}

		for (int r = 0;r < R2;r++)//对每个被消元行
		{
			for (int c = C - 1;c >= 0;c--)//每一列位向量
			{
				//位向量不为0才需要消去
				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
				while (b[r][c] != 0 && col >= 0)//从后往前消去
				{
					if (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
					{
						col--;
						continue;
					}
					cout<<32*c+col<<" ";
					col--;
				}
			}
			cout<<endl;
		}
	}
	else
	{
		//向0号进程发送消去结果
		for (int i = l;i < h;i++)
			MPI_Send(b[i], C, MPI_INT, 0, i, MPI_COMM_WORLD);

	}
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
	std::cout << tms << ": " << fp_ms.count() << "ms" << endl;
	MPI_Finalize();
	return 0;
}
