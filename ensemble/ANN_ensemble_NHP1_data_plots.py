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

data_dir = '/Users/mghrear/data/HPS_data/ensemble_NHP1/'

for p in Path(data_dir).iterdir():

    m = re.search(r'run(\d+)', p.name)
    run_number = int(m.group(1))

    df = pd.read_pickle(data_dir + p.name)

    # Add positron cluster energy filter
    df = df.loc[df.pos_E_Ecal > 0.6]

    plt.figure()

    for num in [400000,200000,100000,50000]:
        df_cut = df.nlargest(num, 'ANN_pred')
        plt.hist(df_cut.InvM*1000, bins=np.arange(980,1250,1), histtype='step', label=str(num), density=False)
        plt.xlabel('Invariant Mass (MeV)')
        plt.ylabel('Normalized Entries')
    
    plt.axvline(1019.461, color='k', linestyle='dashed', linewidth=1, label='PDG phi mass')
    plt.legend()
    plt.savefig('/Users/mghrear/Desktop/Ensemble_study/NHP1/data_plots/'+str(run_number)+'.png')


