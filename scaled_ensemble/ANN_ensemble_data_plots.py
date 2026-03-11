import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sklearn
from sklearn.model_selection import train_test_split
import copy
import mytools
from pathlib import Path
import re


# Print and store device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


FakeGen = False
QualCuts = False
Early = True

if FakeGen:
    str1 = 'FakeGen_1pt05_kaon'
else:
    str1 = 'phiKK'

if QualCuts:
    str2 = '_QualCuts'
else:
    str2 = '_limited'


if Early:
    data_dir = '/Users/mghrear/data/HPS_data/scaled/ensemble_NHP1_early'+str2+'_'+str1+'/'
else: 
    data_dir = '/Users/mghrear/data/HPS_data/scaled/ensemble_NHP1'+str2+'_'+str1+'/'


for p in Path(data_dir).iterdir():

    m = re.search(r'run(\d+)', p.name)
    run_number = int(m.group(1))

    df = pd.read_pickle(data_dir + p.name)

    plt.figure()

    for num in [400000,200000,100000,50000]:
        df_cut = df.nlargest(num, 'ANN_pred')
        plt.hist(df_cut.InvM*1000, bins=np.arange(980,1250,1), histtype='step', label=str(num), density=False)
        plt.xlabel('Invariant Mass (MeV)')
        plt.ylabel('Normalized Entries')
    plt.axvline(1019.461, color='k', linestyle='dashed', linewidth=1, label='PDG phi mass')
    plt.legend()
    if Early:
        plt.savefig('/Users/mghrear/Desktop/Ensemble_study/NHP1_scaled/'+str1+str2+'_early/data_plots/'+str(run_number)+'.png')
    else:
        plt.savefig('/Users/mghrear/Desktop/Ensemble_study/NHP1_scaled/'+str1+str2+'/data_plots/'+str(run_number)+'.png')


