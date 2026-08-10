PREFIX=muon_ParT_unwnonefixed.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1
config=./data_leptonID/${PREFIX%%.*}.yaml

modelopts=""
# trainopts="--fetch-step 1. --data-split-num 1 --batch-size 700 --start-lr 3.2e-4"
trainopts="--num-workers 10 --fetch-step 1. --data-split-num 1 --num-epochs 30 --batch-size 4048 --start-lr 0.00065 "




## export with
source scripts/muonExport.sh run 0 $modelopts $trainopts --model-prefix model/muon_ParT_unwnonefixed.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1/net_best_epoch_state.pt --export-onnx model/muon_ParT_unwnonefixed.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1/muon_ParT_2024.onnx