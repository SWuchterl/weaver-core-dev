
# PREFIX=ak15_MD_incl_v10beta4_ul_full_manual.nlayer10.vispart_as_resid.ddp4-bs640-lr1p2e-3.nepoch100.testrun # 8GPU NGT
# PREFIX=muon_ParT.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v0 # 8GPU NGT
# PREFIX=muon_ParT.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v0_20260415-160021_ParT_ranger_lr8e-05_batch8192 # 8GPU NGT
# PREFIX=muon_ParT.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1 # 8GPU NGT
# PREFIX=muon_ParT_unw.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1 # 8GPU NGT
# PREFIX=muon_ParT_unw_pnet.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1 # 8GPU NGT
# PREFIX=muon_ParT_unw_full.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1 # 8GPU NGT
# PREFIX=muon_ParT_unwnone.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1 # 8GPU NGT
# PREFIX=muon_ParT_unwnone.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v2 # 8GPU NGT
# PREFIX=muon_ParT_unwnone.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v3 # 8GPU NGT



# PREFIX=muon_ParT_unwnonefixed.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v0 # 8GPU NGT
PREFIX=muon_ParT_unwnonefixed.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1 # 8GPU NGT
config=./data_leptonID/${PREFIX%%.*}.yaml




# ============ model options, general
# modelopts="-o num_layers 10 -o reg_kw {'gamma':5.,'composed_split_reg':[True,False],'as_resid_of':[1]} "
modelopts=""


# ============ train options
# trainopts="--num-workers 2 --fetch-step 1. --data-split-num 50 " # on NGT 8 GPU, worked
# trainopts="--num-workers 1 --fetch-step 1. --data-split-num 50 " # on NGT 8 GPU, worked
# trainopts="--num-workers 12 --fetch-step 1. --data-split-num 320 " # on NGT 8 GPU for weights
# 15 would be lowest nfiles for group

# trainopts="--num-workers 20 --fetch-step 1. --data-split-num 10 --batch-size 2200 "
# trainopts="--num-workers 20 --fetch-step 1. --data-split-num 10 --num-epochs 30 "
# trainopts="--num-workers 4 --fetch-step 1. --data-split-num 30 --num-epochs 40 " #4 gpu
# trainopts="--num-workers 8 --fetch-step 0.25 --data-split-num 25 --num-epochs 30 " #4 gpu
# trainopts="--num-workers 10 --fetch-step 0.15 --data-split-num 50 --num-epochs 50 " #2 gpu parquet
# trainopts="--num-workers 10 --fetch-step 0.25 --data-split-num 50 --num-epochs 50 " #2 gpu parquet
# trainopts="--num-workers 10 --fetch-step 0.35 --data-split-num 50 --num-epochs 100 " #2 gpu parquet #mega training
trainopts="--num-workers 10 --fetch-step 0.35 --data-split-num 50 --num-epochs 50 " #2 gpu parquet


# ============ val options
# valopts="--run-mode val --num-workers 5 --fetch-step 1. --data-split-num 50 --log-file logs/${PREFIX}/val.log "
# valopts="--run-mode val --num-workers 5 --fetch-step 1. --data-split-num 5 --log-file logs/${PREFIX}/val.log "
# valopts="--run-mode val --num-workers 20 --num-epochs 50 --fetch-step 0.1 --data-split-num 50 --log-file logs/${PREFIX}/val.log "
# valopts="--run-mode val --num-workers 20 --num-epochs 100 --fetch-step 0.25 --data-split-num 50 --log-file logs/${PREFIX}/val.log "
valopts="--run-mode val --num-workers 10 --num-epochs 50 --fetch-step 0.35 --data-split-num 50 --log-file logs/${PREFIX}/val.log "
valextopts=""


# ============ test options
testopts="--run-mode test --num-workers 1 --data-split-num 1 " # fetch-by-file


# =========== run
# ------------
bssettings="--batch-size 6144 --start-lr 0.0008 "

# source scripts/train_muon.sh run 0,1 --batch-size 8192 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh run 0,1 --batch-size 8192 --start-lr 0.0004 $modelopts $trainopts
# source scripts/train_muon.sh run 0,1 --batch-size 6144 --start-lr 0.0004 $modelopts $trainopts
# source scripts/train_muon.sh run 0,1 --batch-size 6144 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh autorecover 0,1 --batch-size 8192 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh autorecover 0,1 --batch-size 6144 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh autorecover 0,1 --batch-size 6144 --start-lr 0.0004 $modelopts $trainopts #mgea training

# source scripts/train_muon.sh run 0,1 $modelopts $trainopts $bssettings
# source scripts/train_muon.sh autorecover 0,1 $modelopts $trainopts $bssettings

# source scripts/train_muon.sh run 0 --batch-size 8192 --start-lr 2e-4 $valopts $valextopts $modelopts
# source scripts/train_muon.sh run 0 $valopts $valextopts $modelopts $bssettings

# source scripts/offlineTrain_full_fabio.sh run 0 --batch-size 2200 --start-lr 0.00228 $testopts $modelopts --use-last-model
# source scripts/train_muon.sh run 0 --batch-size 2200 --start-lr 2e-4 $testopts $modelopts
# source scripts/train_muon.sh run 0 --batch-size 500 --start-lr 2e-4 $testopts $modelopts
source scripts/train_muon.sh run 0 $testopts $modelopts $bssettings
