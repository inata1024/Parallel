���еĽű�
�����漰�˱����ļ���ڵ���ļ�

#neon_mm.sh
#!/bin/sh
#PBS -N test
#PBS -l nodes=1
pssh -h $PBS_NODEFILE mkdir -p /home/ss2113098 1>&2
scp master:/home/ss2113098/lab_2/neon_mm /home/ss2113098
pscp -h $PBS_NODEFILE /home/ss2113098/lab_2/neon_ /home/ss2113098 1>&2
/home/ss2113098/neon_mm
