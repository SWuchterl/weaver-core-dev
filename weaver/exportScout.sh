# PREFIX=ak8_MD_inclv10_scouting_2p_manual_pre2024_export.std.ddp4-bs750-lr3p2e-prenew
# PREFIX=ak15_MD_inclv10beta4_ul_scout_export.std.ddp4-bs750-lr3p2e-6-mass
PREFIX=ak15_MD_inclv10beta4_ul_scout_export.std.ddp4-bs750-lr3p2e-6-mass_run4_lastEpoch

# config=./data_new/inclv10_aux/${PREFIX%%.*}.yaml
config=./data_new/incl_v10_ak15_finetune/${PREFIX%%.*}.yaml

modelopts=""
trainopts="--fetch-step 1. --data-split-num 1 --batch-size 700 --start-lr 3.2e-4"

## export with
# source scripts/aux/train_GloParT_v3beta4_scout_pre2024.sh run 0 $modelopts $trainopts -o export_params '{"num_cls":22,"concat_hid":False}'  --model-prefix model/ak8_MD_inclv10_scouting_2p_manual_pre2024.std.ddp4-bs750-lr3p2e-prenew/net_best_epoch_state.pt --export-onnx model/ak8_MD_inclv10_scouting_2p_manual_export_pre2024/scouting_model_2022to2023.onnx
# source scripts/scoutingTuneExport.sh run 3 $modelopts $trainopts -o export_params '{"num_cls":22,"concat_hid":False}' --model-prefix model/ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_tousebkp/net_best_epoch_state.pt --export-onnx model/ak15_MD_inclv10beta4_ul_scout_manual_export/scouting_model_ak15_2024.onnx
# source scripts/scoutingTuneExport.sh run 3 $modelopts $trainopts -o export_params '{"num_cls":22,"concat_hid":False}' --model-prefix model/ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_tousebkp2/net_best_epoch_state.pt --export-onnx model/ak15_MD_inclv10beta4_ul_scout_manual_export/scouting_model_ak15_2024.onnx
source scripts/scoutingTuneExport.sh run 3 $modelopts $trainopts -o export_params '{"num_cls":22,"concat_hid":False}' --model-prefix model/ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_run4_lastEpoch/net_epoch-29_state.pt --export-onnx model/ak15_MD_inclv10beta4_ul_scout_manual_export_run4_lastEpoch/scouting_model_ak15_2024.onnx
# source scripts/scoutingTuneExport.sh run 3 $modelopts $trainopts -o export_params '{"num_cls":22,"concat_hid":False}' --model-prefix model/ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_bkp/net_COPIED.pt --export-onnx model/ak15_MD_inclv10beta4_ul_scout_manual_export/scouting_model_ak15_2024.onnx
