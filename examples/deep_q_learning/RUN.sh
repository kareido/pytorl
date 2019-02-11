mkdir -p ./checkpoint
python learning.py 2>&1 | tee -a ./checkpoint/log.txt
