prefix='torl'
alg='ATARI GORILA DQN'
name=$1
partition=$2
nodes=$3

cd ..

if [ -z "${nodes}" ]
then
    echo ''
    echo -e "\e[46mNUM OF NODES NOT SPECIFIED, USE DEFAULT: 1 \e[0m"
    nodes=1
fi

N=$(( ${nodes}+0 ))

srun_cmd="srun -J ${name} -p ${partition} -n${N} --gres gpu:${nodes}"
py_cmd='python launcher.py 2>&1 | tee -a ../log.txt'

echo '________________________________________________________________________________'
echo -e "\e[42mSTARTING\e[0m: \e[43m${alg}\e[0m"
echo           "pg-run -rn ${prefix}/${name} -c"
echo           "${srun_cmd}"
echo           "${py_cmd}"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'

pg-run -rn ${prefix}/${name} -c "${srun_cmd} ${py_cmd}"

echo '________________________________________________________________________________'
echo          "RUN_NAME: ${prefix}/${name}"
echo          "CMD_LINE: sh ${0} ${name} ${partition}"
echo -e "\e[41mFINISHED\e[0m: \e[43m${alg}\e[0m"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'
