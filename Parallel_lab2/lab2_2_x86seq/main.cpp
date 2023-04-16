//特殊高斯消去x86平台串行
//本来就是对齐的
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>

using namespace std;
//int C_raw = 1011, R1 = 539, R2 = 263, task_id = 0;//原始矩阵列数,非零消元子,被消元行,
//int C_raw = 130, R1 = 22, R2 = 8, task_id = 0;//原始矩阵列数,非零消元子,被消元行,
int main()
{
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

	for (int tms = 0;tms < 5;tms++)
	{
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

		for (int r = 0;r < R2;r++)//对每个被消元行
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
		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
		cout << tms << ": " << fp_ms.count() << "ms" << endl;
	}
	return 0;
}
