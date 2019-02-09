mkdir -p ./checkpoint
srun -p SenseMediaF --gres gpu:1 python train.py 2>&1 | tee -a ./checkpoint/log.txt
