#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include<fstream>
#include<sstream>
#include <string.h>
using namespace sycl;

int main()
{
	std::string Folders[] = { "��������1 ��������130��������Ԫ��22������Ԫ��8", "��������2 ��������254��������Ԫ��106������Ԫ��53",
							"��������3 ��������562��������Ԫ��170������Ԫ��53",
							"��������4 ��������1011��������Ԫ��539������Ԫ��263", "��������5 ��������2362��������Ԫ��1226������Ԫ��453",
							"��������6 ��������3799��������Ԫ��2759������Ԫ��1953","��������7 ��������8399��������Ԫ��6375������Ԫ��4535",
							"��������8 ��������23075��������Ԫ��18748������Ԫ��14325" };
	struct Size {
		int a;
		int b;
		int c;//�ֱ�Ϊ������������Ԫ�Ӹ����ͱ���Ԫ�и���
	} fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 263}, {2362, 1226, 453},
	{3799, 2759, 1953},{8399, 6375, 4535},{23075,18748,14325},{37960,29304,14291},{43577,39477,54274} ,{85401,5724,756} };

	for (int tms = 3; tms < 4; tms++)
	{
		int C_raw = fileSize[tms].a, R1 = fileSize[tms].b, R2 = fileSize[tms].c;//ԭʼ��������,������Ԫ��,����Ԫ��,
		int C = ceil(C_raw / 32.0);//λ������������
		
		//Ϊ��Ԫ�ӡ�����Ԫ�з���ռ�
		queue my_gpu_queue(gpu_selector{});
		std::cout << "Selected GPU device: " <<
			my_gpu_queue.get_device().get_info<info::device::name>() << "\n";

		int** a = malloc_host<int*>(R1, my_gpu_queue);
		int** b = malloc_host<int*>(R2, my_gpu_queue);
		int* ini = malloc_host<int>(C_raw, my_gpu_queue);
		//��ʼ��cpu����
		for (int i = 0; i < C_raw; i++)
			ini[i] = -1;
		for (int i = 0; i < R1; i++)
			a[i] = malloc_host<int>(C, my_gpu_queue);
		for (int i = 0; i < R2; i++)
			b[i] = malloc_host<int>(C, my_gpu_queue);


		int* device_er = malloc_device<int>(C, my_gpu_queue);
		int* device_b = malloc_device<int>(C, my_gpu_queue);

		//�������ݼ�����������
		std::ifstream in1("D:/ѧϰ/Groebner data/" + Folders[tms] + "/��Ԫ��.txt");//�����Ҫ���У����޸Ĵ˴����ļ�·��
		std::ifstream in2("D:/ѧϰ/Groebner data/" + Folders[tms] + "/����Ԫ��.txt");//�����Ҫ���У����޸Ĵ˴����ļ�·��
		std::string str;
		int row = 0;//����
		//����ÿ������
		while (getline(in1, str))
		{
			std::istringstream ss(str);
			std::string tmp;
			int count = 0;
			int n = 0;
			while (ss >> n) {
				a[row][n >> 5] += 1 << (31 - n % 32);//ģ�������ð�λ��31����
				if (!count)//��һ��ѭ����ȡ����
					ini[n] = row;//��λ������Ԫ�����row��
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
				b[row][n >> 5] += 1 << (31 - n % 32);//ģ�������ð�λ��31����
			}
			row++;
		}

		auto t1 = std::chrono::high_resolution_clock::now();

		for (int r = 0; r < R2; r++)//��ÿ������Ԫ��
		{
			for (int c = C - 1; c >= 0; c--)//ÿһ��λ����
			{
				//λ������Ϊ0����Ҫ��ȥ
				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//λ������β����
				bool end = false;//��b[r]��Ϊ��Ԫ��ʱ��end = true
				while (b[r][c] != 0 && !end)//�Ӻ���ǰ��ȥ
				{
					if (!((b[r][c] >> (31 - col)) & 1))//�Ƚ�col��Ӧλ�Ƶ�ĩβ���ٺ�1��λ�룬�жϸ�λ�Ƿ�Ϊ1
					{
						col--;
						continue;
					}
					int temp = ini[32 * c + col];
					if (temp > -1)//��������Ԫ�У�ʹ��GPU����
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
					else//����b[r]������Ԫ��
					{
						ini[32 * c + col] = R1 + r;
						end = true;
					}
					col--;
				}
				if (end)//�������ѽ�����Ԫ�ӣ�������Ԫ
					break;
			}
		}
		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
		std::cout << tms << ": " << fp_ms.count() << "ms" << std::endl;
	}
	return 0;
}



//�����㷨
//int main()
//{
//	std::string Folders[] = { "��������1 ��������130��������Ԫ��22������Ԫ��8", "��������2 ��������254��������Ԫ��106������Ԫ��53",
//							"��������3 ��������562��������Ԫ��170������Ԫ��53",
//							"��������4 ��������1011��������Ԫ��539������Ԫ��263", "��������5 ��������2362��������Ԫ��1226������Ԫ��453",
//							"��������6 ��������3799��������Ԫ��2759������Ԫ��1953","��������7 ��������8399��������Ԫ��6375������Ԫ��4535",
//							"��������8 ��������23075��������Ԫ��18748������Ԫ��14325" };
//	struct Size {
//		int a;
//		int b;
//		int c;//�ֱ�Ϊ������������Ԫ�Ӹ����ͱ���Ԫ�и���
//	} fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 263}, {2362, 1226, 453},
//	{3799, 2759, 1953},{8399, 6375, 4535},{23075,18748,14325},{37960,29304,14291},{43577,39477,54274} ,{85401,5724,756} };
//
//	for (int tms = 3; tms < 4; tms++)
//	{
//		int C_raw = fileSize[tms].a, R1 = fileSize[tms].b, R2 = fileSize[tms].c;//ԭʼ��������,������Ԫ��,����Ԫ��,
//
//		int C = ceil(C_raw / 32.0);//λ������������
//		//Ϊ��Ԫ�ӡ�����Ԫ�з���ռ�
//		int** a = new int* [R1], ** b = new int* [R2];
//		int* ini = new int[C_raw];//��¼��λ�������������� ��a�У�0~R1-1 ��b�У�R1~R1+R2-1
//		for (int i = 0; i < C_raw; i++)
//			ini[i] = -1;
//		for (int i = 0; i < R1; i++)
//			a[i] = new int[C] {0};
//		for (int i = 0; i < R2; i++)
//			b[i] = new int[C] {0};
//		//�������ݼ�����������
//		std::ifstream in1("D:/ѧϰ/Groebner data/" + Folders[tms] + "/��Ԫ��.txt");
//		std::ifstream in2("D:/ѧϰ/Groebner data/" + Folders[tms] + "/����Ԫ��.txt");
//		std::string str;
//		int row = 0;//����
//		//����ÿ������
//		while (getline(in1, str))
//		{
//			std::istringstream ss(str);
//			std::string tmp;
//			int count = 0;
//			int n = 0;
//			while (ss >> n) {
//				a[row][n >> 5] += 1 << (31 - n % 32);//ģ�������ð�λ��31����
//				if (!count)//��һ��ѭ����ȡ����
//					ini[n] = row;//��λ������Ԫ�����row��
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
//				b[row][n >> 5] += 1 << (31 - n % 32);//ģ�������ð�λ��31����
//			}
//			row++;
//		}
//
//		auto t1 = std::chrono::high_resolution_clock::now();
//
//		for (int r = 0; r < R2; r++)//��ÿ������Ԫ��
//		{
//			for (int c = C - 1; c >= 0; c--)//ÿһ��λ����
//			{
//				//λ������Ϊ0����Ҫ��ȥ
//				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//λ������β����
//				bool end = false;//��b[r]��Ϊ��Ԫ��ʱ��end = true
//				while (b[r][c] != 0 && !end)//�Ӻ���ǰ��ȥ
//				{
//					if (!((b[r][c] >> (31 - col)) & 1))//�Ƚ�col��Ӧλ�Ƶ�ĩβ���ٺ�1��λ�룬�жϸ�λ�Ƿ�Ϊ1
//					{
//						col--;
//						continue;
//					}
//					int temp = ini[32 * c + col];
//					if (temp > -1)//��������Ԫ��
//					{
//						int* er = temp < R1 ? a[temp] : b[temp - R1];
//						for (int i = 0; i < C; i++)
//							b[r][i] ^= er[i];
//					}
//					else//����b[r]������Ԫ��
//					{
//						ini[32 * c + col] = R1 + r;
//						end = true;
//					}
//					col--;
//				}
//				if (end)//�������ѽ�����Ԫ�ӣ�������Ԫ
//					break;
//			}
//		}
//		auto t2 = std::chrono::high_resolution_clock::now();
//		std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
//		std::cout << tms << ": " << fp_ms.count() << "ms" << std::endl;
//	}
//	return 0;
//}