#!/usr/bin/bash 
OPTSPEC=":Nh-:"

alg='TEST LEARNING OUTCOME'
py_filename='test.py'
rn_prefix='torl' # a prefix of the run name to help manage experiment dir
rn_suffix='default'

echo

while getopts "$OPTSPEC" optchar; do
    case "${optchar}" in
        -)
            case "${OPTARG}" in
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
                *)
                    if [ "$OPTERR" = 1 ]; then
                        echo -e "\e[41m [ERROR] UNKNOWN ARGUMENT '-${OPTARG}' SPECIFIED \e[0m" >&2
                        exit 2
                    fi
                    ;;
            esac;;
        N)
            rn_suffix="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
        h)
            echo "USAGE: $0 <OPTIONS...>" >&2
            echo "          [-h][    --prefix[=]<run name prefix>]" >&2
            echo "          [-h][-N, --name[=]<run name suffix>]" >&2
            echo
            exit 2
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

run_cmd="lrun -n1" # note that test only runs under local env
py_cmd="python ${py_filename} 2>&1 | tee -a ../log.txt"

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


