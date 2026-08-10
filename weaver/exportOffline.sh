# PREFIX=ak8_MD_inclv10_scouting_2p_manual_pre2024_export.std.ddp4-bs750-lr3p2e-prenew
# PREFIX=ak15_MD_inclv10beta4_ul_scout_export.std.ddp4-bs750-lr3p2e-6-mass
PREFIX=ak15_MD_incl_v10beta4_ul_full_export.std.ddp4-bs750-lr3p2e-6-mass_run4_lastEpoch
# PREFIX=ak15_MD_incl_v10beta4_ul_full_manual.nlayer10.vispart_as_resid.ddp4-bs640-lr1p2e-3.nepoch100.testrun # 8GPU NGT


# config=./data_new/inclv10_aux/${PREFIX%%.*}.yaml
# config=./data_new/incl_v10_ak15_finetune/${PREFIX%%.*}.yaml
config=./data_new/incl_v10_ak15_full/${PREFIX%%.*}.yaml

modelopts="-o num_layers 10 -o reg_kw {'gamma':5.,'composed_split_reg':[True,False],'as_resid_of':[1]} "
trainopts="--fetch-step 1. --data-split-num 1 --batch-size 700 --start-lr 3.2e-4"

## export with
# source scripts/offlineExport.sh run 0 $modelopts $trainopts -o export_params '{"num_cls":22,"concat_hid":False}' --model-prefix model/ak15_MD_incl_v10beta4_ul_full_manual.nlayer10.vispart_as_resid.ddp4-bs640-lr1p2e-3.nepoch100.testrun/net_epoch-99_state.pt --export-onnx model/ak15_MD_incl_v10beta4_ul_full_export_lastEpoch/model_ak15.onnx
source scripts/offlineExport.sh run 0 $modelopts $trainopts -o export_params '{"num_cls":22,"concat_hid":False}' --model-prefix model/ak15_MD_incl_v10beta4_ul_full_manual.nlayer10.vispart_as_resid.ddp4-bs640-lr1p2e-3.nepoch100.testrun/net_epoch-99_state.pt --export-onnx model/ak15_MD_incl_v10beta4_ul_full_export_lastEpoch_try2/model_ak15.onnx