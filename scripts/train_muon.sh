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

# current_dir=`pwd`
# if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
#     echo "Please run this script from the weaver directory"
#     exit 1
# fi

DATADIR=/eos/cms/store/cmst3/group/deepjet/leptonid/fromjules_v2/muon/parTsplit/


ARG="--run-mode train \
--network-config networks/ParT.py \
--data-train \
"fake_hi:${DATADIR}/*/*/fake_ptGe35/*.parquet" \
"fake_lo:${DATADIR}/*/*/fake_ptLt35/*.parquet" \
"heavy_hi:${DATADIR}/*/*/heavy_ptGe35/*.parquet" \
"heavy_lo:${DATADIR}/*/*/heavy_ptLt35/*.parquet" \
"light_hi:${DATADIR}/*/*/light_ptGe35/*.parquet" \
"light_lo:${DATADIR}/*/*/light_ptLt35/*.parquet" \
"tau_hi:${DATADIR}/*/*/tau_ptGe35/*.parquet" \
"tau_lo:${DATADIR}/*/*/tau_ptLt35/*.parquet" \
"prompt_hi:${DATADIR}/*/*/prompt_ptGe35/*.parquet" \
"prompt_lo:${DATADIR}/*/*/prompt_ptLt35/*.parquet" \
--data-test \
"test:TestLeptonCMSSW/tree.root" \
--lr-scheduler warmup+cos \
--optimizer adamW \
--use-amp --auto-clean --compile \
--prefetch-factor 5 \
--samples-per-epoch $((150000 * 512 / $NGPUS)) --samples-per-epoch-val $((10000 * 512)) \
--data-config ${config} \
--model-prefix model/${PREFIX}/net \
--tensorboard muon_${model}_{auto}${suffix} \
--predict-output predict/$PREFIX/pred.root "



if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ] || [ $RUN == "autorecover" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

if [ $GPUS == "cpu" ]; then
    echo "Running CPU mode"
    cmd="python3 weaver/weaver/train.py $ARG $cmdlineopts --gpus \"\" "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
    # cmd="python train.py --gpus $GPUS $ARG $cmdlineopts "
    cmd="python3 weaver/weaver/train.py --gpus $GPUS $ARG $cmdlineopts "
else
    # GPU list is separated by comma
    export CUDA_VISIBLE_DEVICES=$GPUS
    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS weaver/weaver/train.py --backend nccl $ARG $cmdlineopts "
fi

echo Run command: $cmd

if [ $RUN == "dryrun" ] || [ $RUN == "run" ]; then
    $cmd
elif [ $RUN == "autorecover" ]; then
    maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
    if [ -z $maxepoch ]; then
        epochopts=""
    else
        epochopts="--load-epoch $maxepoch"
        echo "Resuming from epoch $maxepoch"
    fi

    $cmd $epochopts

fi