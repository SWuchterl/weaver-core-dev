## include additional configs
# PREFIX=ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_run2
# PREFIX=ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_run3
# PREFIX=ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_run3_lastEpoch
# PREFIX=ak15_MD_inclv10beta4_ul_scout.std.ddp4-bs750-lr3p2e-6-mass_run2
# PREFIX=ak15_MD_inclv10beta4_ul_scout_simple.std.ddp4-bs750-lr3p2e-6-mass_run2
# PREFIX=ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_run4
PREFIX=ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_run4_lastEpoch
# PREFIX=ak15_MD_inclv10beta4_ul_scout_manual.std.ddp4-bs750-lr3p2e-6-mass_run5
config=./data_new/incl_v10_ak15_finetune/${PREFIX%%.*}.yaml

modelopts=""
# trainopts="--num-workers 2 --fetch-step 1. --data-split-num 100 --batch-size 750 --start-lr 3.2e-4"
# trainopts="--num-workers 2 --fetch-step 1. --data-split-num 50 --batch-size 750 --start-lr 3.2e-4"
# trainopts="--num-workers 6 --fetch-step 1. --data-split-num 100 --batch-size 750 --start-lr 3.2e-4"
# trainopts="--num-workers 3 --fetch-step 1. --data-split-num 100 --batch-size 750 --start-lr 3.2e-4 --in-memory"
# trainopts="--num-workers 2 --fetch-step 1. --data-split-num 1 --batch-size 750 --start-lr 3.2e-4 --in-memory"

# for ngt
# trainopts="--num-workers 2 --fetch-step 1. --data-split-num 10 --batch-size 1875 --start-lr 0.0000632455532"
# trainopts="--num-workers 1 --fetch-step 1. --data-split-num 1 --batch-size 1875 --start-lr 0.0000632455532 --in-memory " #local files?
# trainopts="--num-workers 1 --fetch-step 1. --data-split-num 1 --batch-size 1875 --start-lr 0.0005059644256 --in-memory " #local files?
# trainopts="--num-workers 1 --fetch-step 1. --data-split-num 1 --batch-size 2000 --start-lr 0.0001 --in-memory " #local files?
# trainopts="--num-workers 1 --fetch-step 1. --data-split-num 1 --batch-size 2000 --in-memory " #local files?
# trainopts="--num-workers 10 --fetch-step 1. --data-split-num 50 --batch-size 2200 "
trainopts="--num-workers 20 --fetch-step 1. --data-split-num 10 --batch-size 2200 "
# trainopts="--num-workers 10 --fetch-step 1. --data-split-num 10 --batch-size 2200 "



regexmatch='^part\.fc.*$'
# finetuneopts="--load-model-weights finetune_stage3beta4p1_scoutpretrainAK15.tillfc0 --optimizer-option lr_mult (\"$regexmatch\",1e1) --start-lr 2.4e-5 --num-epoch 30 -o reg_kw {'gamma':20.,'composed_split_reg':[True,False],'as_resid_of':[1]}"
# finetuneopts="--load-model-weights finetune_stage3beta4p1_scoutpretrainAK15.tillfc0 --optimizer-option lr_mult (\"$regexmatch\",1e1) --start-lr 0.000003 --num-epoch 50 -o reg_kw {'gamma':20.,'composed_split_reg':[True,False],'as_resid_of':[1]}"
# finetuneopts="--load-model-weights finetune_stage3beta4p1_scoutpretrainAK15.tillfc0 --optimizer-option lr_mult (\"$regexmatch\",1e1) --start-lr 0.000003 --num-epoch 50 -o reg_kw {'gamma':20.,'composed_split_reg':[True,False],'as_resid_of':[1]}"
# finetuneopts="--load-model-weights finetune_stage3beta4p1_scoutpretrainAK15.tillfc0 --optimizer-option lr_mult (\"$regexmatch\",1e1) --start-lr 0.0003 --num-epoch 30 -o reg_kw {'gamma':20.,'composed_split_reg':[True,False],'as_resid_of':[1]}"
# finetuneopts="--load-model-weights finetune_stage3beta4p1_scoutpretrainAK15.tillfc0 --optimizer-option lr_mult (\"$regexmatch\",1e1) --start-lr 0.0001 --num-epoch 30 -o reg_kw {'gamma':20.,'composed_split_reg':[True,False],'as_resid_of':[1]}"
finetuneopts="--load-model-weights finetune_stage3beta4p1_scoutpretrainAK15.tillfc0 --optimizer-option lr_mult (\"$regexmatch\",1e1) --start-lr 0.0002 --num-epoch 50 -o reg_kw {'gamma':20.,'composed_split_reg':[True,False],'as_resid_of':[1]}"

## train with
# source scripts/aux/train_GloParT_v3beta4_scout.sh run 0 $modelopts $trainopts $finetuneopts
# source scripts/scoutingTune.sh run 0 $modelopts $trainopts $finetuneopts
# source scripts/scoutingTune.sh run 0,1,2,3,4,5,6,7 $modelopts $trainopts $finetuneopts
# source scripts/scoutingTune.sh autorecover 0,1,2,3,4,5,6,7 $modelopts $trainopts $finetuneopts
# source scripts/scoutingTune.sh run 0 $modelopts $trainopts $finetuneopts

# valopts="--run-mode val --num-workers 4 --fetch-step 1. --data-split-num 50 --log-file logs/${PREFIX}/val.log --batch-size 750 --start-lr 3.2e-4 "
# valopts="--run-mode val --num-workers 1 --fetch-step 1. --data-split-num 1 --log-file logs/${PREFIX}/val.log --batch-size 1875 --start-lr 3.2e-4 --in-memory "
# valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-num 50 --log-file logs/${PREFIX}/val.log --batch-size 1875 --start-lr 3.2e-4 --in-memory "
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-num 50 --log-file logs/${PREFIX}/val.log --batch-size 1875 --start-lr 3.2e-4 "
valextopts="-o eval_kw {'roc_kw':{'comp_list':[('Xbb','QCD'),('Xcc','QCD'),('Xcc','Xbb'),('Wqq','QCD')],'label_inds_map':{'Xbb':[0],'Xcc':[1],'Wqq':[7,8,9,10],'QCD':[17,18,19,20,21]}}} "

# ## validation with
# source scripts/scoutingTune.sh run 0 $modelopts $valopts $valextopts

label_cls_nodes_v3beta3_scout="['label_H_bb','label_H_cc','label_H_ss','label_H_qq','label_Hp_bc','label_Hm_bc','label_H_bs','label_Hp_cs','label_Hm_cs','label_Hp_ud','label_Hm_ud','label_H_gg','label_H_ee','label_H_mm','label_H_tauhtaue','label_H_tauhtaum','label_H_tauhtauh','label_QCD_bb','label_QCD_cc','label_QCD_b','label_QCD_c','label_QCD_others']"
testopts="--run-mode test --num-workers 1 --data-split-num 1 -o label_cls_nodes $label_cls_nodes_v3beta3_scout " # fetch-by-file


## test with
source scripts/scoutingTune.sh run 0 $modelopts $testopts
# source scripts/scoutingTune.sh run 0 $modelopts $testopts --use-last-model