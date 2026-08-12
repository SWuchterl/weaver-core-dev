# write a script that opens a root file and compares 5 output branches with each other. code it so far with placeholder names, but I want 5 2D plots and 5 histograms for the difference as plots, for a set of two times 5 branches.
import uproot
import matplotlib.pyplot as plt
import numpy as np

import mplhep as hep
plt.style.use(hep.style.CMS)

def read_branches(file_path, branches):
    with uproot.open(file_path) as f:
        tree = f["Events"]  # replace "tree" with the actual tree name if different
        data = {branch: tree[branch].array() for branch in branches}
    return data

def plot_2d(data1, data2, branch_name, outfolder):
    import os
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    plt.figure(figsize=(10, 10))
    # plt.hist2d(data1, data2, bins=50, cmap='viridis')
    # scatterplot
    plt.scatter(data1, data2, s=1, alpha=1.0)
    plt.colorbar(label='Entries')
    plt.xlabel(f'{branch_name} (CMSSW)')
    plt.ylabel(f'{branch_name} (Weaver)')
    plt.title(f'2D Comparison of {branch_name}')
    plt.grid()
    plt.savefig(f'{outfolder}/2d_{branch_name}.png')
    plt.savefig(f'{outfolder}/2d_{branch_name}.pdf')
    plt.tight_layout()
    plt.close()

def plot_1dDiff(data1, data2, branch_name, outfolder):
    import os
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    diff = data1 - data2
    plt.figure(figsize=(10, 10))
    plt.hist(diff, bins=500, color='orange', alpha=0.7)
    plt.xlabel(f'{branch_name} (CMSSW - Weaver)')
    plt.ylabel('Entries')
    plt.title(f'Difference of {branch_name}')
    # log y acis
    plt.yscale('log')
    plt.grid()
    plt.savefig(f'{outfolder}/diff_{branch_name}.png')
    plt.savefig(f'{outfolder}/diff_{branch_name}.pdf')
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    file_path_electron = "../predict/electron_ParT_unwnone.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1/pred_test.root"  # replace with your actual file path
    branches_cmssw = ["Lepton_ID_ParTLL_prompt", "Lepton_ID_ParTLL_heavy", "Lepton_ID_ParTLL_light", "Lepton_ID_ParTLL_tau", "Lepton_ID_ParTLL_fake"]  # replace with actual branch names for CMSSW
    branches_weaver = ["score_prompt", "score_heavy", "score_light", "score_tau", "score_fake"]  # replace with actual branch names for Weaver

    data_cmssw = read_branches(file_path_electron, branches_cmssw)
    data_weaver = read_branches(file_path_electron, branches_weaver)

    outfolder = "comparison_plots_electron"
    for cmssw_branch, weaver_branch in zip(branches_cmssw, branches_weaver):
        plot_2d(data_cmssw[cmssw_branch], data_weaver[weaver_branch], "ele_"+cmssw_branch.split("_")[-1], outfolder)
        plot_1dDiff(data_cmssw[cmssw_branch], data_weaver[weaver_branch], "ele_"+cmssw_branch.split("_")[-1], outfolder)

    file_path_muon = "../predict/muon_ParT_unwnonefixed.nlayerDef.Class.ddp4-bs8192-lr2p0e-3.nepoch20.testrun.v1/pred_test.root"  # replace with your actual file path
    branches_cmssw = ["Lepton_ID_ParTLL_prompt", "Lepton_ID_ParTLL_heavy", "Lepton_ID_ParTLL_light", "Lepton_ID_ParTLL_tau", "Lepton_ID_ParTLL_fake"]  # replace with actual branch names for CMSSW
    branches_weaver = ["score_prompt", "score_heavy", "score_light", "score_tau", "score_fake"]  # replace with actual branch names for Weaver

    data_cmssw = read_branches(file_path_muon, branches_cmssw)
    data_weaver = read_branches(file_path_muon, branches_weaver)

    outfolder = "comparison_plots_muon"
    for cmssw_branch, weaver_branch in zip(branches_cmssw, branches_weaver):
        plot_2d(data_cmssw[cmssw_branch], data_weaver[weaver_branch], "ele_"+cmssw_branch.split("_")[-1], outfolder)
        plot_1dDiff(data_cmssw[cmssw_branch], data_weaver[weaver_branch], "ele_"+cmssw_branch.split("_")[-1], outfolder)