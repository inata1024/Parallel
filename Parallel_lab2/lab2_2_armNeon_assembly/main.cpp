//�����˹��ȥarmƽ̨Neon  assembly
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>
#include<arm_neon.h>

using namespace std;
int C_raw = 1011, R1 = 539, R2 = 263, task_id = 0;//ԭʼ��������,������Ԫ��,����Ԫ��,
//int C_raw = 130, R1 = 22, R2 = 8, task_id = 0;//ԭʼ��������,������Ԫ��,����Ԫ��,
int main()
{

	int C = ceil(C_raw / 32.0);//λ������������
	//Ϊ��Ԫ�ӡ�����Ԫ�з���ռ�
	int **a = new int* [R1], **b = new int* [R2];
	int* ini = new int[C_raw];//��¼��λ�������������� ��a�У�0~R1-1 ��b�У�R1~R1+R2-1
	for (int i = 0;i < C_raw;i++)
		ini[i] = -1;
	for (int i = 0;i < R1;i++)
		a[i] = new int[C] {0};
	for (int i = 0;i < R2;i++)
		b[i] = new int[C] {0};
	//�������ݼ�����������
	ifstream in1("/home/data/Groebner/4_1011_539_263/1.txt");
	ifstream in2("/home/data/Groebner/4_1011_539_263/2.txt");
	string str;
	int row = 0;//����
	//����ÿ������
	while (getline(in1, str))
	{
		istringstream ss(str);
		string tmp;
		int count = 0;
		int n=0;
		while (ss>>n) {
			a[row][n >> 5] += 1 << (31 - n % 32);//ģ�������ð�λ��31����
			if(!count)//��һ��ѭ����ȡ����
				ini[n] = row;//��λ������Ԫ�����row��
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
			b[row][n >> 5] += 1 << (31 - n % 32);//ģ�������ð�λ��31����
		}
		row++;
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	for (int r = 0;r < R2;r++)//��ÿ������Ԫ��
	{
		for (int c = C - 1;c >= 0;c--)//ÿһ��λ����
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
				if (temp > -1)//��������Ԫ��
				{
				    int* er = temp < R1 ? a[temp] : b[temp - R1];
				    //Neon����
				    int i=0;
				    for(i=0;i<C-4;i+=4)
                    {
                        int* addr = b[r]+i;
                        int* addr1 = er+i;
                        __asm__ __volatile__(
								"mov x0, %[addr] \n"//�����ڴ��ַ
								"ld1 {v0.4s}, [x0] \n" //���ڴ���װ�ĸ�float��v0
								"mov x0, %[addr1] \n"//���ر�����m[k][k]��ַ
								"ld1r {v1.4s}, [x0] \n"//���������ظ����ص�v1 ld1rָ��
								"fdiv v2.4s,v0.4s,v1.4s \n "//�������
								"mov x0, %[addr] \n"//�����ڴ��ַ
								"st1 {v2.4s}, [x0] \n"//��������ڴ�
								:                                          // ����Ĵ����б�
							: [addr] "r"(addr), [addr1]"r"(addr1)        // ����Ĵ����б�
								: "v0", "v1", "v2", "x0", "memory", "cc"       // ���ƻ��ļĴ����б�
								);

//                        vbri=vld1q_u32((uint32_t *)&b[r][i]);
//                        ver=vld1q_u32((uint32_t *)&er[i]);
//                        vbri=veorq_u32(vbri,ver);
//                        vst1q_u32((uint32_t *)&b[r][i],vbri);
                    }
                    //���д���ʣ�µ�
					for (;i < C;i++)
						b[r][i] ^= er[i];
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
	cout << fp_ms.count() << "ms" << endl;
//
//	//�����Ԫ���
//	for (int r = 0;r < R2;r++)
//	{
//		for (int c = C - 1;c >= 0;c--)
//		{
//			int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//λ������β����
//			while (b[r][c] != 0 && col >= 0)//�Ӻ���ǰ���
//			{
//				if (!((b[r][c] >> (31 - col)) & 1))//�Ƚ�col��Ӧλ�Ƶ�ĩβ���ٺ�1��λ�룬�жϸ�λ�Ƿ�Ϊ1
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
