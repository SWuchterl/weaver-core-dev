# !#/bin/bash


DATADIR=$1

echo "Checking files in $DATADIR..."

tt1l=${DATADIR}/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*.root
tt2l=${DATADIR}/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*.root
tthbb=${DATADIR}/TTH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*.root
tthnotbb=${DATADIR}/TTH-HtoNon2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/*/*/*/*.root
ttlllo=${DATADIR}/TTLL_Bin-MLL-4to50_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root
ttllhi=${DATADIR}/TTLL_Bin-MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root
ttlnu1=:${DATADIR}/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/*/*/*/*.root
ttlnu=${DATADIR}/TTLNu-EWK_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root
tttt=${DATADIR}/TTTT_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root
ttww=${DATADIR}/TTWW_TuneCP5_13p6TeV_madgraph-pythia8/*/*/*/*.root
ttwz=${DATADIR}/TTWZ_TuneCP5_13p6TeV_madgraph-pythia8/*/*/*/*.root
tzq=${DATADIR}/TZQB-Zto2L-4FS_Bin-MLL-30_TuneCP5_13p6TeV_amcatnlo-pythia8/*/*/*/*.root


# now check how many files we have for each sample
count_tt1l=$(ls $tt1l | wc -l)
count_tt2l=$(ls $tt2l | wc -l)
count_tthbb=$(ls $tthbb | wc -l)
count_tthnotbb=$(ls $tthnotbb | wc -l)
count_ttlllo=$(ls $ttlllo | wc -l)
count_ttllhi=$(ls $ttllhi | wc -l)
count_ttlnu1j=$(ls $ttlnu1j | wc -l)
count_ttlnu=$(ls $ttlnu | wc -l)
count_tttt=$(ls $tttt | wc -l)
count_ttww=$(ls $ttww | wc -l)
count_ttwz=$(ls $ttwz | wc -l)
count_tzq=$(ls $tzq | wc -l)

# print all
echo "Number of files for tt1l: $count_tt1l"
echo "Number of files for tt2l: $count_tt2l"
echo "Number of files for tthbb: $count_tthbb"
echo "Number of files for tthnotbb: $count_tthnotbb"
echo "Number of files for ttlllo: $count_ttlllo"
echo "Number of files for ttllhi: $count_ttllhi"
echo "Number of files for ttlnu1j: $count_ttlnu1j"
echo "Number of files for ttlnu: $count_ttlnu"
echo "Number of files for tttt: $count_tttt"
echo "Number of files for ttww: $count_ttww"
echo "Number of files for ttwz: $count_ttwz"
echo "Number of files for tzq: $count_tzq"

echo "-------------------"

# sort them and print the min and max
counts=($count_tt1l $count_tt2l $count_tthbb $count_tthnotbb $count_ttlllo $count_ttllhi $count_ttlnu1j $count_ttlnu $count_tttt $count_ttww $count_ttwz $count_tzq)
sorted_counts=($(for i in "${counts[@]}"; do echo $i; done | sort -n))
min=${sorted_counts[0]}
Min_larger0=$(for i in "${sorted_counts[@]}"; do if [[ $i -gt 0 ]]; then echo $i; break; fi; done)
max=${sorted_counts[-1]}
echo "Min number of files: $min"
echo "Min number of files >0 : $Min_larger0"

echo "Max number of files: $max"

# # print the min and max number of files across all categories
# min=$(echo "$tt1l $tt2l $tthbb $tthnotbb $ttlllo $ttllhi $ttlnu1j $ttlnu $tttt $ttww $ttwz $tzq" | tr ' ' '\n' | xargs -n1 ls | wc -l | sort -n | head -1)
# max=$(echo "$tt1l $tt2l $tthbb $tthnotbb $ttlllo $ttllhi $ttlnu1j $ttlnu $tttt $ttww $ttwz $tzq" | tr ' ' '\n' | xargs -n1 ls | wc -l | sort -n | tail -1)
# echo "Min number of files: $min"
# echo "Max number of files: $max"