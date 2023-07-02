#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include<fstream>
#include<sstream>
#include <string.h>
using namespace sycl;

int main()
{
	std::string Folders[] = { "测试样例1 矩阵列数130，非零消元子22，被消元行8", "测试样例2 矩阵列数254，非零消元子106，被消元行53",
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

	for (int tms = 3; tms < 4; tms++)
	{
		int C_raw = fileSize[tms].a, R1 = fileSize[tms].b, R2 = fileSize[tms].c;//原始矩阵列数,非零消元子,被消元行,
		int C = ceil(C_raw / 32.0);//位向量矩阵列数
		
		//为消元子、被消元行分配空间
		queue my_gpu_queue(gpu_selector{});
		std::cout << "Selected GPU device: " <<
			my_gpu_queue.get_device().get_info<info::device::name>() << "\n";

		int** a = malloc_host<int*>(R1, my_gpu_queue);
		int** b = malloc_host<int*>(R2, my_gpu_queue);
		int* ini = malloc_host<int>(C_raw, my_gpu_queue);
		//初始化cpu数据
		for (int i = 0; i < C_raw; i++)
			ini[i] = -1;
		for (int i = 0; i < R1; i++)
			a[i] = malloc_host<int>(C, my_gpu_queue);
		for (int i = 0; i < R2; i++)
			b[i] = malloc_host<int>(C, my_gpu_queue);


		int* device_er = malloc_device<int>(C, my_gpu_queue);
		int* device_b = malloc_device<int>(C, my_gpu_queue);

		//读入数据集并构建矩阵
		std::ifstream in1("D:/学习/Groebner data/" + Folders[tms] + "/消元子.txt");//如果想要运行，请修改此处的文件路径
		std::ifstream in2("D:/学习/Groebner data/" + Folders[tms] + "/被消元行.txt");//如果想要运行，请修改此处的文件路径
		std::string str;
		int row = 0;//行数
		//处理每行数据
		while (getline(in1, str))
		{
			std::istringstream ss(str);
			std::string tmp;
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
			std::istringstream ss(str);
			std::string tmp;
			int n = 0;
			while (ss >> n) {
				b[row][n >> 5] += 1 << (31 - n % 32);//模操作可用按位与31代替
			}
			row++;
		}

		auto t1 = std::chrono::high_resolution_clock::now();

		for (int r = 0; r < R2; r++)//对每个被消元行
		{
			for (int c = C - 1; c >= 0; c--)//每一列位向量
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
					if (temp > -1)//若存在消元行，使用GPU计算
					{
						int* er = temp < R1 ? a[temp] : b[temp - R1];
						// Copy from host(CPU) to device(GPU)
						my_gpu_queue.memcpy(device_er, er, C * sizeof(int)).wait();
						my_gpu_queue.memcpy(device_b, b[r], C * sizeof(int)).wait();

						// submit the content to the queue for execution
						my_gpu_queue.submit([&](handler& h) {

							// Parallel Computation      
							h.parallel_for(range{ C }, [=](id<1> item) {
								device_b[item] ^= er[item];
								});

							});
						// wait the computation done
						my_gpu_queue.wait();
						// Copy back from GPU to CPU
						my_gpu_queue.memcpy(b[r], device_b, C * sizeof(int)).wait();
							
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
		std::cout << tms << ": " << fp_ms.count() << "ms" << std::endl;
	}
	return 0;
}



//串行算法
//int main()
//{
//	std::string Folders[] = { "测试样例1 矩阵列数130，非零消元子22，被消元行8", "测试样例2 矩阵列数254，非零消元子106，被消元行53",
//							"测试样例3 矩阵列数562，非零消元子170，被消元行53",
//							"测试样例4 矩阵列数1011，非零消元子539，被消元行263", "测试样例5 矩阵列数2362，非零消元子1226，被消元行453",
//							"测试样例6 矩阵列数3799，非零消元子2759，被消元行1953","测试样例7 矩阵列数8399，非零消元子6375，被消元行4535",
//							"测试样例8 矩阵列数23075，非零消元子18748，被消元行14325" };
//	struct Size {
//		int a;
//		int b;
//		int c;//分别为矩阵列数，消元子个数和被消元行个数
//	} fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 263}, {2362, 1226, 453},
//	{3799, 2759, 1953},{8399, 6375, 4535},{23075,18748,14325},{37960,29304,14291},{43577,39477,54274} ,{85401,5724,756} };
//
//	for (int tms = 3; tms < 4; tms++)
//	{
//		int C_raw = fileSize[tms].a, R1 = fileSize[tms].b, R2 = fileSize[tms].c;//原始矩阵列数,非零消元子,被消元行,
//
//		int C = ceil(C_raw / 32.0);//位向量矩阵列数
//		//为消元子、被消元行分配空间
//		int** a = new int* [R1], ** b = new int* [R2];
//		int* ini = new int[C_raw];//记录该位置首项所在行数 在a中：0~R1-1 在b中：R1~R1+R2-1
//		for (int i = 0; i < C_raw; i++)
//			ini[i] = -1;
//		for (int i = 0; i < R1; i++)
//			a[i] = new int[C] {0};
//		for (int i = 0; i < R2; i++)
//			b[i] = new int[C] {0};
//		//读入数据集并构建矩阵
//		std::ifstream in1("D:/学习/Groebner data/" + Folders[tms] + "/消元子.txt");
//		std::ifstream in2("D:/学习/Groebner data/" + Folders[tms] + "/被消元行.txt");
//		std::string str;
//		int row = 0;//行数
//		//处理每行数据
//		while (getline(in1, str))
//		{
//			std::istringstream ss(str);
//			std::string tmp;
//			int count = 0;
//			int n = 0;
//			while (ss >> n) {
//				a[row][n >> 5] += 1 << (31 - n % 32);//模操作可用按位与31代替
//				if (!count)//第一次循环获取首项
//					ini[n] = row;//该位置有消元首项，在row行
//				count++;
//			}
//			row++;
//		}
//		row = 0;
//		while (getline(in2, str))
//		{
//			std::istringstream ss(str);
//			std::string tmp;
//			int n = 0;
//			while (ss >> n) {
//				b[row][n >> 5] += 1 << (31 - n % 32);//模操作可用按位与31代替
//			}
//			row++;
//		}
//
//		auto t1 = std::chrono::high_resolution_clock::now();
//
//		for (int r = 0; r < R2; r++)//对每个被消元行
//		{
//			for (int c = C - 1; c >= 0; c--)//每一列位向量
//			{
//				//位向量不为0才需要消去
//				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
//				bool end = false;//当b[r]成为消元子时，end = true
//				while (b[r][c] != 0 && !end)//从后往前消去
//				{
//					if (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
//					{
//						col--;
//						continue;
//					}
//					int temp = ini[32 * c + col];
//					if (temp > -1)//若存在消元行
//					{
//						int* er = temp < R1 ? a[temp] : b[temp - R1];
//						for (int i = 0; i < C; i++)
//							b[r][i] ^= er[i];
//					}
//					else//否则将b[r]加入消元子
//					{
//						ini[32 * c + col] = R1 + r;
//						end = true;
//					}
//					col--;
//				}
//				if (end)//若该行已进入消元子，则不再消元
//					break;
//			}
//		}
//		auto t2 = std::chrono::high_resolution_clock::now();
//		std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
//		std::cout << tms << ": " << fp_ms.count() << "ms" << std::endl;
//	}
//	return 0;
//}