#! /bin/bash

exp_dir=data/exp/
data_dir=data/fswild/loader/
stage=1
rec_path=data/eval_ckpt/fswild.pth # path of a trained recognizer

while test $# -gt 0;do
    case "$1" in
        -h|--help)
            echo "Evaluation pipeline"
            echo " "
            echo "./scripts/eval.sh [options] [arguments]"
            echo " "
            echo "options:"
            echo "-h, --help show help"
            echo "-d, --data data dir"
            echo "-e, --exp exp dir"
            echo "-s, --stage stage (stage model to evaluate: 1, 2)"
            echo "-r, --rec recognition model (required for MSA/AP@Acc evaluation)"
            exit 0
            ;;
        -e|--exp)
            shift
            if test $# -gt 0;then
                exp_dir=$1
            fi
            echo "exp dir: "$exp_dir
            shift
            ;;
        -d|--data)
            shift
            if test $# -gt 0;then
                data_dir=$1
            fi
            echo "data dir: "$data_dir
            shift
            ;;
        -s|--stage)
            shift
            if test $# -gt 0;then
                stage=$1
            fi
            echo "stage: "$stage
            shift
            ;;
        -r|--rec)
            shift
            if test $# -gt 0;then
                rec=$1
            fi
            echo "recognition model path: "$rec
            shift
            ;;
        *)
            break
            ;;
        esac
done

exp=$exp_dir/stage$stage
eval_label=$data_dir/test.json
eval_scp=$data_dir/video/test/scp

echo "Generating proposals"
python -B evaluate.py --config $exp/train_conf.yaml --eval_scp $eval_scp --eval_label $eval_label --output_fn $exp/test-proposal.pkl --eval_type pred || exit 1;

echo "AP@IoU"
python -B measure.py -p $exp/test-proposal.pkl -t iou || exit 1;

echo "Evaluating with recognizer"
python -B measure.py -p $exp/test-proposal.pkl -m $rec_path -l $eval_label -s $eval_scp -t rec || exit 1;

echo "AP@Acc"
python -B measure.py -p $exp/test-proposal.pkl -t acc || exit 1;

echo "MSA"
python -B measure.py -p $exp/test-proposal.pkl -t msa || exit 1;
