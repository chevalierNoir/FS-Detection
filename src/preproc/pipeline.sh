#! /bin/bash

data=data/fswild
step=1
job=""
set=ChicagoFSWild

while test $# -gt 0;do
    case "$1" in
        -h|--help)
            echo "Training pipeline"
            echo " "
            echo "./pipeline.sh [options] [arguments]"
            echo " "
            echo "options:"
            echo "-h, --help show help"
            echo "-d, --data data dir"
            echo "-t, --set set"
            echo "-j, --job job"
            echo "-s, --step step"
            exit 0
            ;;
        -d|--data)
            shift
            if test $# -gt 0;then
                data=$1
            fi
            echo "data dir: "$data
            shift
            ;;
        -s|--step)
            shift
            if test $# -gt 0;then
                step=$1
            fi
            echo "step: "$step
            shift
            ;;
        -t|--set)
            shift
            if test $# -gt 0;then
                set=$1
            fi
            echo "set: "$set
            shift
            ;;
        -j|--job)
            shift
            if test $# -gt 0;then
                job=$1
            fi
            echo "job id: "$job
            shift
            ;;
        *)
            break
            ;;
        esac
done

if [ -z $job ];then
    python preproc/pipeline.py --data $data --set $set --step $step
else
    python preproc/pipeline.py --data $data --set $set --step $step --job $job
fi
