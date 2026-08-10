#!/bin/bash -x

RUN=$1
GPUS=$2

if [ -z $GPUS ]; then
    echo "Usage: $0 <ngpu>"
    exit 1
fi
NGPUS=$(echo $GPUS | tr "," "\n" | wc -l)

cmdlineopts="${@:3}"

current_dir=`pwd`
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

DATADIR=/eos/cms/store/group/cmst3/group/deepjet/leptonid/fromjules/muon/

# hadded!
## remember: remove all single-quote characters
ARG="--run-mode train \
--use-amp --optimizer ranger \
--network-config networks_leptonID/ParT.py \
--data-train \
"fake_hi:${DATADIR}/class_fake_highpt/*/*/*.parquet" \
"fake_lo:${DATADIR}/class_fake_lowpt/*/*/*.parquet" \
"heavy_hi:${DATADIR}/class_heavy_highpt/*/*/*.parquet" \
"heavy_lo:${DATADIR}/class_heavy_lowpt/*/*/*.parquet" \
"light_hi:${DATADIR}/class_light_highpt/*/*/*.parquet" \
"light_lo:${DATADIR}/class_light_lowpt/*/*/*.parquet" \
"tau_hi:${DATADIR}/class_tau_highpt/*/*/*.parquet" \
"tau_lo:${DATADIR}/class_tau_lowpt/*/*/*.parquet" \
"prompt_hi:${DATADIR}/class_prompt_highpt/*/*/*.parquet" \
"prompt_lo:${DATADIR}/class_prompt_lowpt/*/*/*.parquet" \
--data-test \
"tt1l:${DATADIR}/*/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/fold*/*.parquet" \
"tttt:${DATADIR}/*/TTTT_TuneCP5_13p6TeV_amcatnlo-pythia8/fold*/*.parquet" \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "

if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ] || [ $RUN == "autorecover" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

if [ $GPUS == "cpu" ]; then
    cmd="python3 train.py $ARG $cmdlineopts "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
    unset CUDA_VISIBLE_DEVICES
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
    epochopts=""
    # if the training is halted, resume from the last epoch
    while true; do
        $cmd $epochopts
        ret=$?
        if [ $ret -eq 0 ]; then
            break
        fi
        echo "Error: return code $ret"
        # match model/${PREFIX}/net_epoch-(\d+)_state.pt and extract the maximum epoch number
        maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
        if [ -z $maxepoch ]; then
            epochopts=""
        else
            epochopts="--load-epoch $maxepoch"
            echo "Resuming from epoch $maxepoch"
        fi
        sleep 10
    done
fi
