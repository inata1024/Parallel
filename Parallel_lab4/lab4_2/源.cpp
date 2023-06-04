//�����˹��ȥMPI 
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


	//string Folders[] = { "��������1 ��������130��������Ԫ��22������Ԫ��8", "��������2 ��������254��������Ԫ��106������Ԫ��53",
	//					"��������3 ��������562��������Ԫ��170������Ԫ��53",
	//					"��������4 ��������1011��������Ԫ��539������Ԫ��263", "��������5 ��������2362��������Ԫ��1226������Ԫ��453",
	//					"��������6 ��������3799��������Ԫ��2759������Ԫ��1953","��������7 ��������8399��������Ԫ��6375������Ԫ��4535",
	//					"��������8 ��������23075��������Ԫ��18748������Ԫ��14325" };

	string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
   "6_3799_2759_1953","7_8399_6375_4535", "8_23075_18748_14325","9_37960_29304_14291","10_43577_39477_54274","11_85401_5724_756" };
	struct Size {
		int a;
		int b;
		int c;//�ֱ�Ϊ������������Ԫ�Ӹ����ͱ���Ԫ�и���
	} fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 263}, {2362, 1226, 453},
	{3799, 2759, 1953},{8399, 6375, 4535},{23075,18748,14325},{37960,29304,14291},{43577,39477,54274} ,{85401,5724,756} };
	int tms = 0;

	int C_raw = fileSize[tms].a, R1 = fileSize[tms].b, R2 = fileSize[tms].c;//ԭʼ��������,������Ԫ��,����Ԫ��,

	int C = ceil(C_raw / 32.0);//λ������������
	//Ϊ��Ԫ�ӡ�����Ԫ�з���ռ�
	int** a = new int* [R1], ** b = new int* [R2];
	int* ini;

	MPI_Alloc_mem(C_raw * sizeof(int), MPI_INFO_NULL, &ini);
	//int* ini = new int[C_raw];//��¼��λ�������������� ��a�У�0~R1-1 ��b�У�R1~R1+R2-1
	for (int i = 0;i < C_raw;i++)
		ini[i] = -1;
	for (int i = 0;i < R1;i++)
		MPI_Alloc_mem(C * sizeof(int), MPI_INFO_NULL, &a[i]);

	//a[i] = new int[C] {0};
	for (int i = 0;i < R2;i++)
		MPI_Alloc_mem(C * sizeof(int), MPI_INFO_NULL, &b[i]);

	//b[i] = new int[C] {0};

//�������ݼ�����������
//ifstream in1("D:/ѧϰ/���г������/Groebner data/" + Folders[tms] + "/��Ԫ��.txt");
//ifstream in2("D:/ѧϰ/���г������/Groebner data/" + Folders[tms] + "/����Ԫ��.txt");
//�������ݼ�����������
	ifstream in1("/home/data/Groebner/" + Folders[tms] + "/1.txt");
	ifstream in2("/home/data/Groebner/" + Folders[tms] + "/2.txt");
	string str;
	int row = 0;//����
	//����ÿ������
	while (getline(in1, str))
	{
		istringstream ss(str);
		string tmp;
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
		istringstream ss(str);
		string tmp;
		int n = 0;
		while (ss >> n) {
			b[row][n >> 5] += 1 << (31 - n % 32);//ģ�������ð�λ��31����
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
	//��0���߳����ݹ���

	//ÿ���̴߳���һЩ����Ԫ�У�ÿһ��ѭ������
	for (int r = 0;r < R2;r++)//��ÿ������Ԫ��
	{
		if (rank == 0 || rank - 1 != r % (size - 1))
			continue;
		bool end = false;//��b[r]��Ϊ��Ԫ��ʱ��end = true
		for (int c = C - 1;c >= 0;c--)//ÿһ��λ����
		{
			//λ������Ϊ0����Ҫ��ȥ
			int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//λ������β����

			while (b[r][c] != 0 && !end)//�Ӻ���ǰ��ȥ
			{
				while (!((b[r][c] >> (31 - col)) & 1))//�Ƚ�col��Ӧλ�Ƶ�ĩβ���ٺ�1��λ�룬�жϸ�λ�Ƿ�Ϊ1
				{
					col--;
					continue;
				}
				//ʵ��˼·��ʹ��RMA lock/unlock ʵ������openMP�Ĺ����ڴ�
				//����0�Ž��̵�����
				//ÿ�����̸����Լ�����
				int temp = 0;
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win0);
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win[r]);
				//int temp = ini[32 * c + col];
				MPI_Get(&temp, 1, MPI_INT, 0, (32 * c + col), 1, MPI_INT, win0);
				//MPI_Get_accumulate(&temp, 1, MPI_INT, &temp, 1, MPI_INT, 0, (32 * c + col), 1, MPI_INT, MPI_NO_OP, win0);
				MPI_Win_flush(0, win0);

				if (temp == -1)//��b[r]������Ԫ��
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

				if (temp > -1)//��������Ԫ��
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
			if (end)//�������ѽ�����Ԫ�ӣ�������Ԫ
				break;
		}
		//��ȥ��ɵı���Ԫ�У�����0�Ž���
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
		printf("�����ģ:%d ʱ��: % fms\n", a[tms], (end - start) * 1000);
		//for (int r = 0;r < R2;r++)//��ÿ������Ԫ��
		//{
		//	for (int c = C - 1;c >= 0;c--)//ÿһ��λ����
		//	{
		//		//λ������Ϊ0����Ҫ��ȥ
		//		int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//λ������β����
		//		while (b[r][c] != 0 && col >= 0)//�Ӻ���ǰ��ȥ
		//		{
		//			if (!((b[r][c] >> (31 - col)) & 1))//�Ƚ�col��Ӧλ�Ƶ�ĩβ���ٺ�1��λ�룬�жϸ�λ�Ƿ�Ϊ1
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

