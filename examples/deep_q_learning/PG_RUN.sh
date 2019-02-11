prefix='rl'
name=$1
partition=$2

srun_cmd="srun --job-name ${name} -p ${partition} --gres gpu:1 python learning.py 2>&1 | tee -a ../log.txt"

echo '________________________________________________________________________________'
echo "pg-run -rn ${prefix}/${name} -c"
echo "${srun_cmd}"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'

pg-run -rn ${prefix}/${name} -c "${srun_cmd}"

echo"RUN_NAME: ${prefix}/${name}"
echo"CMD_LINE: sh PG_RUN.sh ${name} ${partition}"
echo"          DEEP Q-LEARNING COMPLETED"
