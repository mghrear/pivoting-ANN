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
import joblib 
import re


# Print and store device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Apply to MC to determine a selection that removes 99% of background

# Load MC data
df_tritrig = pd.read_pickle('/Users/mghrear/data/ML_data/patch/2021_v9_pass5_tritrig_limited.pk')
df_tritrig['PhiKK'] = 0.0
df_phiKK = pd.read_pickle('/Users/mghrear/data/ML_data/patch/2021_v9_pass5_phiKK_limited.pk')
df_phiKK['PhiKK'] = 1.0
df_wab = pd.read_pickle('/Users/mghrear/data/ML_data/patch/2021_v9_pass5_wab_limited.pk')
df_wab['PhiKK'] = 0.0 # Add label


# Make MC test set
# MUST BE CONSISTENT with BDT.ipynb and ANN.ipynb to avoid data leaks!!
tritrig_test = df_tritrig[20000:]
tritrig_test['type']= 'tritrig'
wab_test = df_wab[3000:]
wab_test['type']= 'wab'
phiKK_test = df_phiKK[3000:]
phiKK_test['type']= 'phiKK'

df_test = pd.concat([tritrig_test, phiKK_test,wab_test], ignore_index=True, sort=False)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True) # Now shuffle the combined dataframe


X_test = df_test.drop(columns=['PhiKK','type'])
Y_test = df_test['PhiKK']

df_test['InvM'] = mytools.get_InvM(df_test)


test_dataset = TensorDataset(torch.from_numpy(X_test.to_numpy().astype(np.float32)) )
test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)


model_dir = '/Users/mghrear/data/ML_data/patch/ensemble/'

ANN_slections = []
run_numbers = []

for p in Path(model_dir).iterdir():

    # Load model
    m = re.search(r'run(\d+)', model_dir+p.name)
    run_number = int(m.group(1))
    print("loading model: ", run_number)
    run_numbers.append(run_number)

    ANN = mytools.Classifier(in_features=X_test.shape[1]).to(device)
    ANN.load_state_dict(torch.load(model_dir+p.name, map_location=device))
    ANN.eval()

    #Inference step
    ANN_pred = mytools.test_final(test_loader, ANN, device)
    ANN_pred=torch.sigmoid(torch.tensor(ANN_pred)).numpy()
    df_test['ANN'] = ANN_pred

    # Split into tritrig, wab, background and phiKK dataframes
    test_df_tritrig = df_test[df_test['type']=="tritrig"].reset_index(drop=True)
    test_df_phiKK = df_test[df_test['type']=="phiKK"].reset_index(drop=True)
    test_df_wab = df_test[df_test['type']=="wab"].reset_index(drop=True)
    test_df_bkg = pd.concat([test_df_tritrig,test_df_wab]).reset_index(drop=True)


    ANN_selection = np.percentile(test_df_bkg['ANN'],[99.9])[0]
    print("ANN selection at 99.9% background rejection: ", ANN_selection)
    ANN_slections.append(ANN_selection)

    #Make mc plots
    plt.figure()
    for sel in np.percentile(test_df_bkg['ANN'],[0,10,20,30,40,50,60,70,80,90,99]):

        df_tritrig_cut = test_df_bkg[test_df_bkg['ANN']>sel].reset_index(drop=True)
        x_vals = 1000*df_tritrig_cut.InvM
        plt.hist(x_vals, bins=np.arange(980,1250,1), histtype='step', label=f'ANN > {sel:.4f}', density=True)

    plt.axvline(1019.461, color='k', linestyle='dashed', linewidth=1, label='PDG phi mass')
    plt.xlabel('Invariant Mass [MeV]')
    plt.ylabel('Normalized Counts')
    plt.legend()
    plt.savefig(f'/Users/mghrear/Desktop/Ensemble_study/original_QualCuts/MC_plots/ANN_ensemble_run{run_number}_mc_invmass.png')


# make pandas dataframe to store run numbers and selections
df_ANN_selections = pd.DataFrame({'run_number': run_numbers, 'ANN_selection': ANN_slections})


# Now apply to the 5% data


data_dir = '/Users/mghrear/data/HPS_data/2021_v9_pass5_processed/'
out_dir = '/Users/mghrear/data/HPS_data/ensemble_QualCuts_T1/'

def getmask(df, QualCuts):
    mask = (
        (df['pos_E_Ecal'] > QualCuts['pos_E_Ecal_low'] ) &
        (df['pos_Pz'] > QualCuts['pos_Pz_low']  ) &
        (df['pos_Px'] > QualCuts['pos_Px_low']  ) &
        (df['pos_Py'] > QualCuts['pos_Py_low'] ) & (df['pos_Py'] < QualCuts['pos_Py_high'] ) &
        (df['ele_Px'] > QualCuts['ele_Px_low'] ) &
        (df['ele_Py'] > QualCuts['ele_Py_low'] ) & (df['ele_Py'] < QualCuts['ele_Py_high'] ) &
        (df['pos_Ecal_x'] > QualCuts['pos_Ecal_x_low'] ) &
        (df['pos_Ecal_y'] > QualCuts['pos_Ecal_y_low'] ) & (df['pos_Ecal_y'] < QualCuts['pos_Ecal_y_high'] ) &
        (df['pos_Ecal_z'] > QualCuts['pos_Ecal_z_low'] ) &
        (df['ele_Ecal_x'] < QualCuts['ele_Ecal_x_high'] ) &
        ((df['ele_Ecal_z'] > QualCuts['ele_Ecal_z_low'] ) | (df['ele_Ecal_z'] < 0))  
    )
    return mask


QualCuts = {
    'pos_E_Ecal_low': 0.5,
    'pos_Pz_low': 0.65,
    'pos_Px_low': -0.06,
    'pos_Py_low': -0.12,
    'pos_Py_high': 0.13,
    'ele_Px_low': -0.12,
    'ele_Py_low': -0.14,
    'ele_Py_high': 0.16,
    'pos_Ecal_x_low': 100.0,
    'pos_Ecal_y_low': -85.0,
    'pos_Ecal_y_high': 90.0,
    'pos_Ecal_z_low': 1448.6,
    'ele_Ecal_x_high': 10.0,
    'ele_Ecal_z_low': 1448.6
}


# loop though dataframe
for index, row in df_ANN_selections.iterrows():
    run_number = int(row['run_number'])
    ANN_selection = row['ANN_selection']
    print(f'Run number: {run_number}, ANN selection: {ANN_selection}')

    # Load the correspinding model
    ANN = mytools.Classifier(in_features=X_test.shape[1]).to(device)
    ANN.load_state_dict(torch.load( '/Users/mghrear/data/ML_data/patch/ensemble/classifier_adv_2021_v9_pass5_run'+str(run_number)+'_limited.pt' , map_location=device))
    ANN.eval()

    model_preds = np.array([])
    InvMs = np.array([])
    pEcals = np.array([])

    for p in Path(data_dir).iterdir():

        # Load dataframe
        df = pd.read_pickle(data_dir+p.name)

        # Apply Quality Cuts
        df = df.loc[getmask(df, QualCuts)].reset_index(drop=True)

        # Get InvM
        InvM = mytools.get_InvM(df) 

        # Setup dataloader
        test_dataset = TensorDataset(torch.from_numpy(df.to_numpy().astype(np.float32)))
        test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)


        ANN_pred = mytools.test_clas(test_loader, ANN, device)
        ANN_pred = torch.sigmoid(torch.tensor(ANN_pred)).numpy()

        # append ANN preds to model
        df['ANN_pred'] = ANN_pred

        df = df [df.ANN_pred > ANN_selection]

        df['InvM'] = mytools.get_InvM(df)

        # Append to overall arrays
        model_preds = np.concatenate((model_preds, df['ANN_pred'].to_numpy()), axis=0)
        InvMs = np.concatenate((InvMs, df['InvM'].to_numpy()), axis=0)
        pEcals = np.concatenate((pEcals, df['pos_E_Ecal'].to_numpy()), axis=0)

    # Save the predictions and InvM to a pickle file
    out_df = pd.DataFrame({'InvM': InvMs, 'ANN_pred': model_preds, 'pos_E_Ecal': pEcals})
    out_df.to_pickle(out_dir+f'ANN_ensemble_run{run_number}_selected.pk')

