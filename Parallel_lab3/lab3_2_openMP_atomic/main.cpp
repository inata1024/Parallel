//�����˹��ȥx86ƽ̨ openMP
//�������Ƕ����
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>
#include<omp.h>
#define NUM_THREADS 4
using namespace std;
//int C_raw = 1011, R1 = 539, R2 = 263, task_id = 0;//ԭʼ��������,������Ԫ��,����Ԫ��,
//int C_raw = 130, R1 = 22, R2 = 8, task_id = 0;//ԭʼ��������,������Ԫ��,����Ԫ��,
int main()
{
	string Folders[] = { "��������1 ��������130��������Ԫ��22������Ԫ��8", "��������2 ��������254��������Ԫ��106������Ԫ��53",
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

	for (int tms = 5;tms < 6;tms++)
	{
		int C_raw = fileSize[tms].a, R1 = fileSize[tms].b, R2 = fileSize[tms].c;//ԭʼ��������,������Ԫ��,����Ԫ��,

		int C = ceil(C_raw / 32.0);//λ������������
		//Ϊ��Ԫ�ӡ�����Ԫ�з���ռ�
		int** a = new int* [R1], ** b = new int* [R2];
		int* ini = new int[C_raw];//��¼��λ�������������� ��a�У�0~R1-1 ��b�У�R1~R1+R2-1
		for (int i = 0;i < C_raw;i++)
			ini[i] = -1;
		for (int i = 0;i < R1;i++)
			a[i] = new int[C] {0};
		for (int i = 0;i < R2;i++)
			b[i] = new int[C] {0};
		//�������ݼ�����������
		ifstream in1("D:/ѧϰ/���г������/Groebner data/" + Folders[tms] + "/��Ԫ��.txt");
		ifstream in2("D:/ѧϰ/���г������/Groebner data/" + Folders[tms] + "/����Ԫ��.txt");
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

		auto t1 = std::chrono::high_resolution_clock::now();

		#pragma omp parallel num_threads(NUM_THREADS)
		{
		    #pragma omp for
		    for (int r = 0;r < R2;r++)//��ÿ������Ԫ��
            {
                for (int c = C - 1;c >= 0;c--)//ÿһ��λ����
                {
                    //λ������Ϊ0����Ҫ��ȥ
                    int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//λ������β����
                    bool end = false;//��b[r]��Ϊ��Ԫ��ʱ��end = true
                    while (b[r][c] != 0 && !end)//�Ӻ���ǰ��ȥ
                    {
                        while(!((b[r][c] >> (31 - col)) & 1))//�Ƚ�col��Ӧλ�Ƶ�ĩβ���ٺ�1��λ�룬�жϸ�λ�Ƿ�Ϊ1
                        {
                            col--;
                            continue;
                        }
                        int temp=0;
                        #pragma omp atomic

                            temp += ini[32 * c + col];

                        if (temp > -1)//��������Ԫ��
                        {
                            int* er = temp < R1 ? a[temp] : b[temp - R1];
                            for (int i = 0;i < C;i++)
                                b[r][i] ^= er[i];
                        }
                        else//����b[r]������Ԫ��
                        {
                            end = true;
                            #pragma omp atomic
                                ini[32 * c + col] += R1 + r + 1;//��Ϊini��ֵΪ-1����ϣ����+=,���Զ����һ��1


                        }
                        col--;
                    }
                    if (end)//�������ѽ�����Ԫ�ӣ�������Ԫ
                        break;
                }
            }
		}
		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
		cout << tms << ": " << fp_ms.count() << "ms" << endl;
//        for (int r = 0;r < R2;r++)//��ÿ������Ԫ��
//		{
//			for (int c = C - 1;c >= 0;c--)//ÿһ��λ����
//			{
//				//λ������Ϊ0����Ҫ��ȥ
//				int col = min(32 * (c + 1) - 1, C_raw - 1) % 32;//λ������β����
//				while (b[r][c] != 0 && col >= 0)//�Ӻ���ǰ��ȥ
//				{
//					if (!((b[r][c] >> (31 - col)) & 1))//�Ƚ�col��Ӧλ�Ƶ�ĩβ���ٺ�1��λ�룬�жϸ�λ�Ƿ�Ϊ1
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
	return 0;
}
