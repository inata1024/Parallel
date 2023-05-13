//arm�����˹��ȥ pthread ��̬�߳� mutex����
#include <stdlib.h>
#include<pthread.h>
#include <iostream>
#include<fstream>
#include <sstream>
#include<string>
#include<math.h>
#include <chrono>
using namespace std;
//�����������˺�
#define NUM_THREADS 8
//int C_raw = 1011, R1 = 539, R2 = 263, task_id = 0;//ԭʼ��������,������Ԫ��,����Ԫ��,
//int C_raw = 130, R1 = 22, R2 = 8, task_id = 0;//ԭʼ��������,������Ԫ��,����Ԫ��,

typedef struct {
    int t_id;
}threadParam_t;

//����������
pthread_mutex_t amutex = PTHREAD_MUTEX_INITIALIZER;

//����ȫ�ֱ���
int C_raw,R1,R2,C;//ԭʼ��������,������Ԫ��,����Ԫ��,λ������������
int **a,**b,*ini;//��Ԫ�ӡ�����Ԫ�С���¼��Ԫ����������

//�����̺߳���
void *threadFunc(void *param){
    threadParam_t *p = (threadParam_t*)param;
    int t_id = p -> t_id; //�̱߳��

    for (int r = t_id;r < R2;r+=NUM_THREADS)//��ÿ������Ԫ��
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
                //��ǰ�̼߳���
                pthread_mutex_lock(&amutex);
                int temp = ini[32 * c + col];
                pthread_mutex_unlock(&amutex);
                if (temp > -1)//��������Ԫ��
                {
                    int* er = temp < R1 ? a[temp] : b[temp - R1];
                    for (int i = 0;i < C;i++)
                        b[r][i] ^= er[i];
                }
                else//����b[r]������Ԫ��
                {
                    //��ǰ�̼߳���
                    pthread_mutex_lock(&amutex);
                    ini[32 * c + col] = R1 + r;
                    end = true;
                    pthread_mutex_unlock(&amutex);

                }
                col--;
            }
            if (end)//�������ѽ�����Ԫ�ӣ�������Ԫ
                break;
        }
    }
    pthread_exit(NULL);
}

int main()
{
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
   "6_3799_2759_1953","7_8399_6375_4535", "8_23075_18748_14325","9_37960_29304_14291","10_43577_39477_54274","11_85401_5724_756"};
	struct Size {
		int a;
		int b;
		int c;//�ֱ�Ϊ������������Ԫ�Ӹ����ͱ���Ԫ�и���
	} fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 263}, {2362, 1226, 453},
	{3799, 2759, 1953},{8399, 6375, 4535},{23075,18748,14325},{37960,29304,14291},{43577,39477,54274} ,{85401,5724,756} };


	for (int tms = 0;tms < 7;tms++)
	{
		C_raw = fileSize[tms].a;
		R1 = fileSize[tms].b;
		R2 = fileSize[tms].c;

		C = ceil(C_raw / 32.0);//λ������������
		//Ϊ��Ԫ�ӡ�����Ԫ�з���ռ�
		a = new int* [R1];
        b = new int* [R2];
		ini = new int[C_raw];//��¼��λ�������������� ��a�У�0~R1-1 ��b�У�R1~R1+R2-1
		for (int i = 0;i < C_raw;i++)
			ini[i] = -1;
		for (int i = 0;i < R1;i++)
			a[i] = new int[C] {0};
		for (int i = 0;i < R2;i++)
			b[i] = new int[C] {0};
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

		auto t1 = std::chrono::high_resolution_clock::now();

        //�����߳�
        pthread_t handles[NUM_THREADS];// ������Ӧ��Handle
        threadParam_t param[NUM_THREADS];// ������Ӧ���߳����ݽṹ
        for(int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
        }


        for(int t_id=0;t_id<NUM_THREADS;++t_id)
            pthread_join(handles[t_id],NULL);


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
	pthread_mutex_destroy(&amutex);
	return 0;
}
