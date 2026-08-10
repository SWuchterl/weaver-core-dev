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

# DATADIR=/eos/cms/store/group/cmst3/group/deepjet/leptonid/fromjules/ele/electron/
DATADIR=/eos/cms/store/group/cmst3/group/deepjet/leptonid/fromjules/ele/electron_v2/

# changed to v3beta4 default command
## remember: remove all single-quote characters --io-test 
# --num-workers 8 --fetch-step 1. --data-split-num 250 \


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
"test:TestLeptonCMSSW/tree.root" \
--samples-per-epoch $((150000 * 512 / $NGPUS / 2)) --samples-per-epoch-val $((10000 * 512 / 2)) \
--data-config ${config} \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "

# "tt1l:${DATADIR}/*/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/fold*/*.parquet" \
# "tttt:${DATADIR}/*/TTTT_TuneCP5_13p6TeV_amcatnlo-pythia8/fold*/*.parquet" \

# --samples-per-epoch $((150000 * 512 / $NGPUS)) --samples-per-epoch-val $((10000 * 512)) \


# --samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \


# "fake_hi:${DATADIR}/class_fake_highpt/*/*/*.parquet" \
# "fake_lo:${DATADIR}/class_fake_lowpt/*/*/*.parquet" \
# "heavy_hi:${DATADIR}/class_heavy_highpt/*/*/*.parquet" \
# "heavy_lo:${DATADIR}/class_heavy_lowpt/*/*/*1*.parquet" \
# "light_hi:${DATADIR}/class_light_highpt/*/*/*.parquet" \
# "light_lo:${DATADIR}/class_light_lowpt/*/*/*.parquet" \
# "tau_hi:${DATADIR}/class_tau_highpt/*/*/*.parquet" \
# "tau_lo:${DATADIR}/class_tau_lowpt/*/*/*.parquet" \
# "prompt_hi:${DATADIR}/class_prompt_highpt/*/*/*1*.parquet" \
# "prompt_lo:${DATADIR}/class_prompt_lowpt/*/*/*1*.parquet" \

# --copy-inputs \
# --model-prefix model/${PREFIX}_{auto}/net \


# "tt1l:${DATADIR}/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*2.root" \
# "tt2l:${DATADIR}/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*.root" \
# "tthbb:${DATADIR}/TTH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*.root" \
# "tthnotbb:${DATADIR}/TTH-HtoNon2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*2*.root" \
# "ttlllo:${DATADIR}/TTLL_Bin-MLL-4to50_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root" \
# "ttllhi:${DATADIR}/TTLL_Bin-MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root" \
# "ttlnu1j:${DATADIR}/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/*/*/*/*.root" \
# "ttlnu:${DATADIR}/TTLNu-EWK_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root" \
# "ttww:${DATADIR}/TTWW_TuneCP5_13p6TeV_madgraph-pythia8/*/*/*/*.root" \
# "ttwz:${DATADIR}/TTWZ_TuneCP5_13p6TeV_madgraph-pythia8/*/*/*/*.root" \
# "tzq:${DATADIR}/TZQB-Zto2L-4FS_Bin-MLL-30_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root" \
# "tttt:${DATADIR}/TTTT_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root" \

# test
# "tt1l:${DATADIR}/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*1*2*.root" \



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