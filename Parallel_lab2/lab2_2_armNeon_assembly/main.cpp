//特殊高斯消去arm平台Neon  assembly
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>
#include<arm_neon.h>

using namespace std;
int C_raw = 1011, R1 = 539, R2 = 263, task_id = 0;//原始矩阵列数,非零消元子,被消元行,
//int C_raw = 130, R1 = 22, R2 = 8, task_id = 0;//原始矩阵列数,非零消元子,被消元行,
int main()
{

	int C = ceil(C_raw / 32.0);//位向量矩阵列数
	//为消元子、被消元行分配空间
	int **a = new int* [R1], **b = new int* [R2];
	int* ini = new int[C_raw];//记录该位置首项所在行数 在a中：0~R1-1 在b中：R1~R1+R2-1
	for (int i = 0;i < C_raw;i++)
		ini[i] = -1;
	for (int i = 0;i < R1;i++)
		a[i] = new int[C] {0};
	for (int i = 0;i < R2;i++)
		b[i] = new int[C] {0};
	//读入数据集并构建矩阵
	ifstream in1("/home/data/Groebner/4_1011_539_263/1.txt");
	ifstream in2("/home/data/Groebner/4_1011_539_263/2.txt");
	string str;
	int row = 0;//行数
	//处理每行数据
	while (getline(in1, str))
	{
		istringstream ss(str);
		string tmp;
		int count = 0;
		int n=0;
		while (ss>>n) {
			a[row][n >> 5] += 1 << (31 - n % 32);//模操作可用按位与31代替
			if(!count)//第一次循环获取首项
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
		int n=0;
		while (ss>>n) {
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
				    //Neon并行
				    int i=0;
				    for(i=0;i<C-4;i+=4)
                    {
                        int* addr = b[r]+i;
                        int* addr1 = er+i;
                        __asm__ __volatile__(
								"mov x0, %[addr] \n"//加载内存地址
								"ld1 {v0.4s}, [x0] \n" //从内存中装四个float到v0
								"mov x0, %[addr1] \n"//加载被除数m[k][k]地址
								"ld1r {v1.4s}, [x0] \n"//将被除数重复加载到v1 ld1r指令
								"fdiv v2.4s,v0.4s,v1.4s \n "//向量相除
								"mov x0, %[addr] \n"//加载内存地址
								"st1 {v2.4s}, [x0] \n"//结果存入内存
								:                                          // 输出寄存器列表
							: [addr] "r"(addr), [addr1]"r"(addr1)        // 输入寄存器列表
								: "v0", "v1", "v2", "x0", "memory", "cc"       // 被破坏的寄存器列表
								);

//                        vbri=vld1q_u32((uint32_t *)&b[r][i]);
//                        ver=vld1q_u32((uint32_t *)&er[i]);
//                        vbri=veorq_u32(vbri,ver);
//                        vst1q_u32((uint32_t *)&b[r][i],vbri);
                    }
                    //串行处理剩下的
					for (;i < C;i++)
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
	cout << fp_ms.count() << "ms" << endl;
//
//	//输出消元结果
//	for (int r = 0;r < R2;r++)
//	{
//		for (int c = C - 1;c >= 0;c--)
//		{
//			int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//位向量中尾坐标
//			while (b[r][c] != 0 && col >= 0)//从后往前输出
//			{
//				if (!((b[r][c] >> (31 - col)) & 1))//先将col对应位移到末尾，再和1按位与，判断该位是否为1
//				{
//					col--;
//					continue;
//				}
//				cout << 32 * c + col << " ";
//				col--;
//			}
//		}
//		cout << endl;
//	}
	return 0;
}
