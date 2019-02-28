prefix='gorila'
alg='ATARI GORILA DQN'
name=$1
partition=$2
nodes=$3
gpus=$4

cd ..
echo ''

if [ -z "${nodes}" ]
then
    echo -e "\e[46mNUM OF NODES NOT SPECIFIED, USE DEFAULT: 8 \e[0m"
    nodes=8
fi

if [ -z "${gpus}" ]
then
    echo -e "\e[46mNUM OF GPUS NOT SPECIFIED, USE DEFAULT: ${nodes} \e[0m"
    gpus=${nodes}
fi

N=$(( ${nodes}+0 ))

srun_cmd="srun -J ${name} -p ${partition} -n${N} --gres gpu:${gpus}"
py_cmd='python launcher.py 2>&1 | tee -a ../log.txt'

echo '________________________________________________________________________________'
echo -e "\e[42mSTARTING:\e[0m\e[43m ${alg} \e[0m"
echo           "pg-run -rn ${prefix}/${name} -c"
echo           "${srun_cmd}"
echo           "${py_cmd}"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'

pg-run -rn ${prefix}/${name} -c "${srun_cmd} ${py_cmd}"

echo '________________________________________________________________________________'
echo          "RUN_NAME: ${prefix}/${name}"
echo          "CMD_LINE: sh ${0} ${name} ${partition}"
echo -e "\e[41mFINISHED:\e[0m\e[43m ${alg} \e[0m"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'
