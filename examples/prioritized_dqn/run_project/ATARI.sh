prefix='gorila'
alg='ATARI PRIORITIZED DQN'
name=$1
partition=$2

cd ..

srun_cmd="srun -J ${name} -p ${partition} --gres gpu:1"
py_cmd='python learn.py 2>&1 | tee -a ../log.txt'

echo '________________________________________________________________________________'
echo -e "\e[42mSTARTING\e[0m:\e[43m ${alg} \e[0m"
echo           "pg-run -rn ${prefix}/${name} -c"
echo           "${srun_cmd}"
echo           "${py_cmd}"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'

pg-run -rn ${prefix}/${name} -c "${srun_cmd} ${py_cmd}"

echo '________________________________________________________________________________'
echo          "RUN_NAME: ${prefix}/${name}"
echo          "CMD_LINE: sh ${0} ${name} ${partition}"
echo -e "\e[41mFINISHED\e[0m:\e[43m ${alg} \e[0m"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'
