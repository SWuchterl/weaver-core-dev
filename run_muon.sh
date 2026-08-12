PREFIX=muon_ParT_unwnone.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1 # 8GPU NGT
config=./data/${PREFIX%%.*}.yaml


# ============ model options, general
modelopts=""


# ============ train options
# 15 would be lowest nfiles for group

trainopts="--num-workers 10 --fetch-step 0.45 --data-split-num 30 --num-epochs 50 "

# ============ val options
valopts="--run-mode val --num-workers 10 --num-epochs 50 --fetch-step 0.35 --data-split-num 50 --log-file logs/${PREFIX}/val.log "
valextopts=""

# ============ test options
testopts="--run-mode test --num-workers 1 --data-split-num 1 " # fetch-by-file


# =========== general options
bssettings="--batch-size 6144 --start-lr 0.0008 "

# =========== run training
# source scripts/train_muon.sh run 0,1 --batch-size 8192 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh run 0,1 --batch-size 8192 --start-lr 0.0004 $modelopts $trainopts
# source scripts/train_muon.sh run 0,1 --batch-size 6144 --start-lr 0.0004 $modelopts $trainopts
# source scripts/train_muon.sh run 0,1 --batch-size 6144 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh autorecover 0,1 --batch-size 8192 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh autorecover 0,1 --batch-size 6144 --start-lr 0.0008 $modelopts $trainopts
# source scripts/train_muon.sh autorecover 0,1 --batch-size 6144 --start-lr 0.0004 $modelopts $trainopts #mgea training

source scripts/train_muon.sh run 0,1,2,3 $modelopts $trainopts $bssettings
# source scripts/train_muon.sh autorecover 0,1 $modelopts $trainopts $bssettings


# =========== val training
# source scripts/train_muon.sh run 0 $valopts $valextopts $modelopts $bssettings

# =========== test training
# source scripts/train_muon.sh run 0 $testopts $modelopts $bssettings
