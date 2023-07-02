//MPI Pipeline算法
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>
#include<omp.h>
#include<mpi.h>
#define NUM_THREADS 4
using namespace std;
int main(int argc,char* argv[])
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
	//f是接收到的被消元行
	int** a = nullptr, ** b = nullptr, * ini = nullptr, * f = nullptr;
	//0号进程存被消元行
	if (rank == 0)
	{
		b = new int* [R2];
		for (int i = 0;i < R2;i++)
			b[i] = new int[C] {0};
	}
	//其他进程存消元子
	else {
		f = new int [R1];
		a = new int* [R1];
		ini = new int[C_raw];//记录该位置首项所在行数 在a中：0~R1-1 在b中：R1~R1+R2-1
		for (int i = 0;i < C_raw;i++)
			ini[i] = -1;
		for (int i = 0;i < R1;i++)
			a[i] = new int[C] {0};
	}

	//读入数据集并构建矩阵
	ifstream in1("D:/学习/并行程序设计/Groebner data/" + Folders[tms] + "/消元子.txt");
	ifstream in2("D:/学习/并行程序设计/Groebner data/" + Folders[tms] + "/被消元行.txt");
	string str;
	int row = 0;//行数
	//处理每行数据
	if (rank == 0)
	{
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
	}
	//其他进程存消元子
	else {
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
	}


	auto t1 = std::chrono::high_resolution_clock::now();


	if (rank == 0)
	{
		for (int r = 0;r < R2;r++)//向1号进程发送被消元行
		{
			MPI_Send(b[r], C, MPI_INT, 1, r, MPI_COMM_WORLD);
		}
		//发出去的被消元行，要么消为0，要么升格留在某个进程
		//简单来自规整，默认所有进程都收发n次
		for (int r = 0;r < R2;r++)//接收被消元行
		{
			MPI_Recv(b[r], C, MPI_INT, NUM_THREADS - 1, r, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
	}
	else
	{
		//该进程负责的消元子范围[l,h)，以位向量为单位
		int block_size = C / NUM_THREADS;
		int l = (rank - 1) * block_size, h = rank == (NUM_THREADS - 1) ? C : rank * block_size;
		block_size = 32 * (h - l);//h-l个位向量，32*(h-l)个元素
		int** temp_a = new int* [block_size];//储存升格后的消元子
		for (int i = 0;i < block_size;i++)
		{
			temp_a[i] = new int[C];
		}

		for (int r = 0;r < R2;r++)//对每个被消元行
		{
			MPI_Recv(f, C, MPI_INT, rank - 1, r, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

			for (int c = h - 1;c >= l;c--)//[l,h)每一列位向量
			{
				//位向量不为0才需要消去
				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
				bool end = false;//当f成为消元子时，end = true
				while (f[c] != 0 && !end)//从后往前消去
				{
					while (!((f[c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
					{
						col--;
						continue;
					}
					int temp = ini[32 * c + col];
					if (temp > -1)//若存在消元行
					{
						int* er = temp < R1 ? a[temp] : temp_a[temp - R1];
						for (int i = 0;i < C;i++)
							f[i] ^= er[i];
					}
					else//否则将f加入消元子
					{
						ini[32 * c + col] = R1 + (32 * c + col - l * 32);
						//f停留在该进程，不会在此后进行消去
						for (int i = 0;i < C;i++)
							f[i] = 0;
						end = true;
					}
					col--;
				}
				if (end)//若该行已进入消元子，则不再消元
					break;
			}
			MPI_Send(f, C, MPI_INT, (rank + 1) % NUM_THREADS, r, MPI_COMM_WORLD);//发送给下一个进程
		}
	}
	
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
	//cout << tms << ": " << fp_ms.count() << "ms" << endl;
	MPI_Finalize();
	return 0;
}
