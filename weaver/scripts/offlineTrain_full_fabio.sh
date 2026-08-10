#!/bin/bash -x

RUN=$1
GPUS=$2

if [ -z $GPUS ]; then
    echo "Usage: $0 <ngpu>"
    exit 1
fi
NGPUS=$(echo $GPUS | tr "," "\n" | wc -l)

# print number of GPUS to screen
echo $GPUS

cmdlineopts="${@:3}"

current_dir=`pwd`
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

# changed to v3beta4 default command
## remember: remove all single-quote characters --io-test 
ARG="--run-mode train --train-mode hybrid \
-o num_nodes 750 -o num_cls_nodes 374 -o use_swiglu_config True -o use_pair_norm_config True \
-o fc_params [(2048,0.1)] -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 12 \
-o reg_kw {'gamma':5.,'composed_split_reg':[True,False],'as_resid_of':[0]} \
--use-amp --batch-size 512 --start-lr 7e-4 --num-epochs 100 --optimizer ranger \
--num-workers 8 --fetch-step 1. --data-split-num 250 \
--network-config networks/example_ParticleTransformer2024PlusTagger_unified2SV.py \
--data-train \
t_qcd170to300:./datasets/v10_ul_train/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/*.root \
--data-test \
znunu2jpt150to250:./datasets/v10_ul_train_v2/inferv2/Z2JetsToNuNu_M-50_LHEFilterPtZ-150To250/*.root \
znunu2jpt250to400:./datasets/v10_ul_train_v2/inferv2/Z2JetsToNuNu_M-50_LHEFilterPtZ-250To400/*.root \
znunu2jpt50to150:./datasets/v10_ul_train_v2/inferv2/Z2JetsToNuNu_M-50_LHEFilterPtZ-50To150/*.root \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} \
--model-prefix model/${PREFIX}/net \
--predict-output /scratch/sewuchte/predict/$PREFIX/pred.root "

# --copy-inputs \

# znunu1jpt150to250:./datasets/v10_ul_train_v2/inferv2/Z1JetsToNuNu_M-50_LHEFilterPtZ-150To250/*.root \
# znunu1jpt250to400:./datasets/v10_ul_train_v2/inferv2/Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400/*.root \
# znunu1jpt400toinf:./datasets/v10_ul_train_v2/inferv2/Z1JetsToNuNu_M-50_LHEFilterPtZ-400ToInf/*.root \
# znunu1jpt50to150:./datasets/v10_ul_train_v2/inferv2/Z1JetsToNuNu_M-50_LHEFilterPtZ-50To150/*.root \

if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ] || [ $RUN == "autorecover" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

if [ $GPUS == "cpu" ]; then
    echo "Running CPU mode"
    cmd="python3 train.py $ARG $cmdlineopts --gpus \"\" "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
    # cmd="python train.py --gpus $GPUS $ARG $cmdlineopts "
    cmd="python3 train.py --gpus $GPUS $ARG $cmdlineopts "
else
    # GPU list is separated by comma
    export CUDA_VISIBLE_DEVICES=$GPUS
    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS train.py --backend nccl $ARG $cmdlineopts "
fi

echo Run command: $cmd

if [ $RUN == "dryrun" ] || [ $RUN == "run" ]; then
    $cmd
elif [ $RUN == "autorecover" ]; then
    # epochopts=""
    # # if the training is halted, resume from the last epoch
    # while true; do
    #     $cmd $epochopts
    #     ret=$?
    #     if [ $ret -eq 0 ]; then
    #         break
    #     fi
    #     echo "Error: return code $ret"
    #     # match model/${PREFIX}/net_epoch-(\d+)_state.pt and extract the maximum epoch number
    #     maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
    #     if [ -z $maxepoch ]; then
    #         epochopts=""
    #     else
    #         epochopts="--load-epoch $maxepoch"
    #         echo "Resuming from epoch $maxepoch"
    #     fi
    #     sleep 10
    # done

    maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
    if [ -z $maxepoch ]; then
        epochopts=""
    else
        epochopts="--load-epoch $maxepoch"
        echo "Resuming from epoch $maxepoch"
    fi

    # echo $epochopts
    # echo $cmd

    $cmd $epochopts

fi