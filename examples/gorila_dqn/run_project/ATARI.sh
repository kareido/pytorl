#!/usr/bin/bash 
OPTSPEC=":GLNchnp-:"

alg='ATARI GORILA DQN' # name of algorithm
py_filename='launcher.py'
local=false
rn_prefix='gorila' # a prefix of the run name to help manage experiment dir
rn_suffix='default'
default_num_tasks=4
default_cpus_per_task=1

echo

while getopts "$OPTSPEC" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
                cpus-per-task)
                    cpus="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                cpus-per-task=*)
                    cpus=${OPTARG#*=}
                    ;;
                gpu)
                    gpus="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                gpu=*)
                    gpus=${OPTARG#*=}
                    ;;
                partition)
                    partition="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                partition=*)
                    partition=${OPTARG#*=}
                    ;;
                prefix)
                    rn_prefix="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                prefix=*)
                    rn_prefix=${OPTARG#*=}
                    ;;
                name)
                    rn_suffix="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                name=*)
                    rn_suffix=${OPTARG#*=}
                    ;;
                ntasks)
                    tasks="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                ntasks=*)
                    tasks=${OPTARG#*=}
                    ;;
                ntasks-per-node)
                    ntasks_per_node="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                    ;;
                ntasks-per-node=*)
                    ntasks_per_node=${OPTARG#*=}
                    ;;
                local)
                    local=true
                    echo -e "\e[46m [lrun] LOCAL RUN (NON SRUN) MODE SPECIFIED \e[0m"
                    ;;
                *)
                    if [ "$OPTERR" = 1 ]; then
                        echo -e "\e[41m [ERROR] UNKNOWN ARGUMENT '-${OPTARG}' SPECIFIED \e[0m" >&2
                        exit 2
                    fi
                    ;;
            esac;;
        G)
            gpus="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
        L)
            local=true
            echo -e "\e[46m [lrun] LOCAL RUN (NON SRUN) MODE SPECIFIED \e[0m"
            ;;
        N)
            rn_suffix="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
        c)
            cpus="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
        h)
            echo "USAGE: $0 <OPTIONS...>" >&2
            echo "          [-h][-G, --gpu[=]<corresp. to srun gres gpu:>]" >&2
            echo "          [-h][-L, --local    using lrun instead of srun]" >&2
            echo "          [-h][-c, --cpus-per-task[=]<num cpus per task>]" >&2
            echo "          [-h][-n, --ntasks[=]<total num tasks>]" >&2
            echo "          [-h][    --ntasks-per-node[=]<num tasks per node>]" >&2
            echo "          [-h][    --prefix[=]<run name prefix>]" >&2
            echo "          [-h][-N, --name[=]<run name suffix>]" >&2
            echo "          [-h][-p, --partition=<srun partition>]" >&2
            echo
            exit 2
            ;;
        n)
            ntasks="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
        # lower case p
        p)
            partition="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
        *)
            if [ "$OPTERR" != 1 ] || [ "${OPTSPEC:0:1}" = ":" ]; then
                echo -e "\e[41m [ERROR] UNKNOWN ARGUMENT '-${OPTARG}' SPECIFIED \e[0m" >&2
                exit 2
            fi
            ;;
    esac
done

cd ..

if [ -z "${tasks}" ]; then
    tasks=${default_num_tasks}
fi

if [ -z "${ntasks_per_node}" ]; then
    ntasks_per_node=${tasks}
fi

if [ -z "${cpus}" ]; then
    cpus=${default_cpus_per_task}
fi

if [ -z "${gpus}" ]; then
    gpus=${ntasks_per_node}
fi

local_cmd="lrun -n${tasks}"

run_cmd="srun -J ${rn_suffix} -p ${partition} -n${tasks} --gres gpu:${gpus} --ntasks-per-node ${ntasks_per_node}"
    
py_cmd="python ${py_filename} 2>&1 | tee -a ../log.txt"

if [ ${local} == true ]; then
    run_cmd=${local_cmd}
else
    if [ -z "${partition}" ]; then
        echo -e "\e[41m [ERROR] PARTITION(--partition or -p) NOT SPECIFIED \e[0m"  >&2
        exit 2
    fi
fi

echo -e "\e[46m [ntasks] USING DEFAULT VALUE [${tasks}] (AS SPECIFIED IN THIS SCRIPT)\e[0m"
if [ ${local} == false ]; then
    echo -e "\e[46m [ntasks_per_node] USING DEFAULT VALUE [${ntasks_per_node}] (EQUALS TO [ntasks]) \e[0m"
    echo -e "\e[46m [cpus_per_task] USING DEFAULT VALUE [${cpus}] (AS SPECIFIED IN THIS SCRIPT) \e[0m"
    echo -e "\e[46m [gpu] USING DEFAULT VALUE [${gpus}] (EQUALS TO [ntasks_per_node]) \e[0m"
fi

echo '________________________________________________________________________________'
echo -e "\e[42m STARTING \e[0m\e[43m ${alg} \e[0m"
echo           "rl-run -rn ${rn_prefix}/${rn_suffix} -c"
echo           "${run_cmd}"
echo           "${py_cmd}"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'

rl-run -rn ${rn_prefix}/${rn_suffix} -c "${run_cmd} ${py_cmd}"

echo '________________________________________________________________________________'
echo          "RUN_NAME: ${rn_prefix}/${rn_suffix}"
echo          "CMD_LINE: sh $0 $@"
echo -e "\e[44m FINISHED \e[0m\e[43m ${alg} \e[0m"
echo '‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾'


