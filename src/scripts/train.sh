#! /bin/bash

exp_dir=data/exp/
data_dir=data/fswild/loader/
step=1

while test $# -gt 0;do
    case "$1" in
        -h|--help)
            echo "Training pipeline"
            echo " "
            echo "./scripts/train.sh [options] [arguments]"
            echo " "
            echo "options:"
            echo "-h, --help show help"
            echo "-d, --data data dir"
            echo "-e, --exp exp dir"
            echo "-s, --step step"
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
        -s|--step)
            shift
            if test $# -gt 0;then
                step=$1
            fi
            echo "step: "$step
            shift
            ;;
        *)
            break
            ;;
        esac
done

stage1_exp=$exp_dir/stage1
stage2_exp=$exp_dir/stage2

if [ $step -eq 1 ];then
    echo "Step 1: stage-1 training"
    python train.py  --pose_coef 0 --fsr_coef 0.1 --det_coef 1 --reward_coef 0 --det_sample_rate 1 --info_interval 1000  --det_interval 1000 --epoch 8 --train_label $data_dir/train.json --dev_label $data_dir/dev.json --train_scp $data_dir/video/train/scp --dev_scp $data_dir/video/dev/scp --output $stage1_exp --stage 1 || exit 1;
fi
if [ $step -eq 2 ];then
    echo "Step 2: prepare bounding box for stage-2 training"
    for part in {train,dev,test};do
        python evaluate.py --config $stage1_exp/train_conf.yaml --eval_scp $data_dir/video/$part/scp --eval_label $data_dir/$part.json --output_fn $stage2_exp/bbox.pkl --eval_type bbox || exit 1;
    done
fi
if [ $step -eq 3 ];then
    echo "Step 3: stage-2 training"
    python train.py  --pose_coef 0 --fsr_coef 0.1 --det_coef 1 --reward_coef 0 --det_sample_rate 2  --info_interval 1000 --det_interval 1000 --epoch 8 --stage 2 --train_label $data_dir/train.json --dev_label $data_dir/dev.json --train_scp $data_dir/video/train/scp --dev_scp $data_dir/video/dev/scp  --output $stage2_exp --bbox_file $stage2_exp/bbox.pkl || exit 1;
fi
