mkdir -p ./checkpoint2
srun -p SenseMediaF --gres gpu:1 python train.py 2>&1 | tee -a ./checkpoint2/log.txt
