import os
import subprocess

basePath="/eos/cms/store/cmst3/group/deepjet/ScoutingAK15/ScoutingTrees/dnntuples/v10_ul_train/"

# newPath=basePath+"/hadded/"
newPath="/scratch/sewuchte/"+"/ScoutingAK15/ScoutingTrees/dnntuples/v10_ul_train//hadded/"

if not os.path.exists(newPath):
    print("Creating directory: ", newPath)
    os.makedirs(newPath)

for folder in os.listdir(basePath):   
    command = "hadd -fk -j 8 "+newPath+"/"+folder+".root "+basePath+"/"+folder+"/"+"/*.root"
    print (command)
    subprocess.call(command, shell=True)



# now same for testing
basePath="/eos/cms/store/cmst3/group/deepjet/ScoutingAK15/ScoutingTrees/dnntuples/v10_ul_train/test/"
newPath="/scratch/sewuchte/"+"/ScoutingAK15/ScoutingTrees/dnntuples/v10_ul_train/hadded/test/"
if not os.path.exists(newPath):
    print("Creating directory: ", newPath)
    os.makedirs(newPath)

for folder in os.listdir(basePath):   
    command = "hadd -fk -j 8 "+newPath+"/"+folder+".root "+basePath+"/"+folder+"/"+"/*.root"
    print (command)
    subprocess.call(command, shell=True)