mkdir ./checkpoint
srun -p SenseMediaF python train.py 2>&1 | tee -a ./checkpoint/log.txt