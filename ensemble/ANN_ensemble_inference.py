import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import copy
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import mytools
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


FakeGen = False
model_dir = '/Users/mghrear/data/ML_data/2019_pass2/models/scaled_ensemble/'


QualCuts = {
    'pos_E_Ecal_low': 0.6,
    'pos_Px_low': -0.06,
    'ele_Ecal_z_low': 1448.6,
    'ele_Ecal_y_low': -85.0,
    'ele_Ecal_y_high': 85.0,
}


df_tritrig = pd.read_pickle('/Users/mghrear/data/ML_data/2019_pass2/scaled/2019_pass2_tritrig.pk')
df_tritrig['PhiKK'] = 0.0
df_wab = pd.read_pickle('/Users/mghrear/data/ML_data/2019_pass2/scaled/2019_pass2_wab.pk')
df_wab['PhiKK'] = 0.0 # Add label
if FakeGen:
    df_phiKK = pd.read_pickle('/Users/mghrear/data/ML_data/2019_pass2/scaled/2019_pass2_FakeGen_1pt05_kaon.pk')
else:
    df_phiKK = pd.read_pickle('/Users/mghrear/data/ML_data/2019_pass2/scaled/2019_pass2_phiKK.pk')
df_phiKK['PhiKK'] = 1.0

df_data = pd.read_pickle('/Users/mghrear/data/ML_data/2019_pass2/scaled/2019_pass2_data_full.pk')


# Make MC test set
# MUST BE CONSISTENT with BDT.ipynb and ANN.ipynb to avoid data leaks!!
tritrig_test = df_tritrig[50000:]
tritrig_test['type']= 'tritrig'
wab_test = df_wab[6000:]
wab_test['type']= 'wab'
phiKK_test = df_phiKK[5000:]
phiKK_test['type']= 'phiKK'

df_test = pd.concat([tritrig_test, phiKK_test,wab_test], ignore_index=True, sort=False)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True) # Now shuffle the combined dataframe

X_test = df_test.drop(columns=['PhiKK','type'])
Y_test = df_test['PhiKK']

# Make data set
X_final = df_data


SENTINEL_COLS = ['ele_Ecal_x', 'ele_Ecal_y', 'ele_Ecal_z']

# Load the fitted pipeline (imputer + scaler) saved by ANN_NHP1.ipynb
pipeline = joblib.load("/Users/mghrear/data/ML_data/2019_pass2/scaler_2019_pass2.pkl")

# Create scaled copies for neural network inference
# BDT uses the original unscaled X_test / X_final
X_test_nn = X_test.copy()
X_test_nn[SENTINEL_COLS] = X_test_nn[SENTINEL_COLS].replace(-9999.0, np.nan)
X_test_nn = pd.DataFrame(pipeline.transform(X_test_nn), columns=X_test_nn.columns, index=X_test_nn.index)

X_final_nn = X_final.copy()
X_final_nn[SENTINEL_COLS] = X_final_nn[SENTINEL_COLS].replace(-9999.0, np.nan)
X_final_nn = pd.DataFrame(pipeline.transform(X_final_nn), columns=X_final_nn.columns, index=X_final_nn.index)



for seed in np.arange(1,101,1):
    print("loading model: ", seed)

    ANN = mytools.Classifier(in_features=X_test.shape[1]).to(device)
    if FakeGen:
        ANN.load_state_dict(torch.load(model_dir+"FakeGen_classifier_adv_2019_pass2_run"+str(seed)+".pt", map_location=device))
    else:
        ANN.load_state_dict(torch.load(model_dir+"classifier_adv_2019_pass2_run"+str(seed)+".pt", map_location=device))
    ANN.eval()

    # Get model predictions
    test_dataset = TensorDataset(torch.from_numpy(X_test_nn.to_numpy().astype(np.float32)) )
    test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)
    final_dataset = TensorDataset(torch.from_numpy(X_final_nn.to_numpy().astype(np.float32)) )
    final_loader = DataLoader(final_dataset, batch_size=2000, shuffle=False)

    ANN_pred = mytools.test_final(test_loader, ANN, device)
    ANN_pred=torch.sigmoid(torch.tensor(ANN_pred)).numpy()

    ANN_pred_final = mytools.test_final(final_loader, ANN, device)
    ANN_pred_final=torch.sigmoid(torch.tensor(ANN_pred_final)).numpy()


    df_test['ANN'] = ANN_pred
    df_test['InvM'] = mytools.get_InvM(df_test)

    df_data['ANN'] = ANN_pred_final
    df_data['InvM'] = mytools.get_InvM(df_data)

    ANN_selection = np.percentile(df_data['ANN'], 90)

    # Save the results (events passing ANN selection only)
    df_data_cut = df_data[df_data['ANN'] > ANN_selection].reset_index(drop=True)
    if FakeGen:
        df_data_cut.to_pickle('/Users/mghrear/data/ML_data/2019_pass2/ensemble_results/inference/scaled/2019_pass2_data_full_FakeGen_run'+str(seed)+'.pk')
    else:
        df_data_cut.to_pickle('/Users/mghrear/data/ML_data/2019_pass2/ensemble_results/inference/scaled/2019_pass2_data_full_run'+str(seed)+'.pk')


    # Now make a data / MC comparison plot 

    # Split into tritrig, wab, background and phiKK dataframes
    test_df_tritrig = df_test[df_test['type']=="tritrig"].reset_index(drop=True)
    test_df_phiKK = df_test[df_test['type']=="phiKK"].reset_index(drop=True)
    test_df_wab = df_test[df_test['type']=="wab"].reset_index(drop=True)
    test_df_bkg = pd.concat([test_df_tritrig,test_df_wab])

    # Combine dfs according to expected proportions 1470:228:1 (tritrig:wab:phiKK)
    # Binding constraint: floor(len(test_df_wab) / 228) = floor(10422/228) = 45
    k = min(len(test_df_tritrig) // 1470, len(test_df_wab) // 228, len(test_df_phiKK) // 1)
    n_tritrig, n_wab, n_phiKK = 1470 * k, 228 * k, 1 * k
    print(f"k={k}  tritrig={n_tritrig}  wab={n_wab}  phiKK={n_phiKK}  total={n_tritrig+n_wab+n_phiKK}")

    df_test_combined = pd.concat(
        [test_df_tritrig.iloc[:n_tritrig], test_df_wab.iloc[:n_wab], test_df_phiKK.iloc[:n_phiKK]],
        ignore_index=True, sort=False
    )


    bins = np.arange(0,1.01,0.01)

    mc   = df_test_combined['ANN'].to_numpy()
    data = df_data['ANN'].to_numpy()
    sig = test_df_phiKK['ANN'].to_numpy()


    # Use counts (not density) to form the ratio correctly
    mc_counts,   edges = np.histogram(mc,   bins=bins)
    data_counts, _     = np.histogram(data, bins=bins)

    mc_scale = data_counts.sum() / mc_counts.sum()

    centers = 0.5*(edges[:-1] + edges[1:])

    ratio = np.divide(data_counts, mc_counts * mc_scale,
                    out=np.full_like(data_counts, np.nan, dtype=float),
                    where=mc_counts > 0)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    # --- Top: MC scaled to match data total events ---
    ax_top.hist(mc,   bins=bins, histtype='step', density=False, weights=np.full(len(mc), mc_scale), label='MC (scaled)')
    ax_top.hist(data, bins=bins, histtype='step', density=False, label='data')
    ax_top.hist(sig, bins=bins, histtype='step', label='Signal')

    ax_top.set_ylabel('Count')
    ax_top.set_yscale('log')
    ax_top.legend()

    # --- Bottom: data / scaled MC ratio ---
    ax_bot.step(centers, ratio, where='mid', color='k')
    ax_bot.grid(True)
    ax_bot.set_ylabel('data / MC (scaled)')
    ax_bot.set_xlabel('ANN Output')
    ax_bot.axhline(y=1, color='k', linestyle='--')
    ax_bot.set_ylim(0, 10)  # adjust as you like
    if FakeGen:
        plt.savefig('/Users/mghrear/data/ML_data/2019_pass2/ensemble_results/plots/scaled/data_mc_comparison/'+'data_mc_comparison_FakeGen_run'+str(seed)+'.png')
    else:
        plt.savefig('/Users/mghrear/data/ML_data/2019_pass2/ensemble_results/plots/scaled/data_mc_comparison/'+'data_mc_comparison_run'+str(seed)+'.png')


    df_data_Qual = df_data[
        (df_data.pos_E_Ecal > QualCuts['pos_E_Ecal_low']) &
        (df_data.pos_Px > QualCuts['pos_Px_low']) &
        ( (df_data.ele_Ecal_z > QualCuts['ele_Ecal_z_low']) | (df_data['ele_Ecal_z'] < 0)) &
        (df_data.ele_Ecal_y > QualCuts['ele_Ecal_y_low']) &
        (df_data.ele_Ecal_y < QualCuts['ele_Ecal_y_high'])
    ]

    df_test_combined_Qual = df_test_combined[
        (df_test_combined.pos_E_Ecal > QualCuts['pos_E_Ecal_low']) &
        (df_test_combined.pos_Px > QualCuts['pos_Px_low']) &
        ( (df_test_combined.ele_Ecal_z > QualCuts['ele_Ecal_z_low']) | (df_test_combined['ele_Ecal_z'] < 0)) &
        (df_test_combined.ele_Ecal_y > QualCuts['ele_Ecal_y_low']) &
        (df_test_combined.ele_Ecal_y < QualCuts['ele_Ecal_y_high'])
    ]


    bins = np.arange(0, 1.01, 0.01)

    mc_ann_cut   = df_test_combined_Qual['ANN'].to_numpy()
    data_ann_cut = df_data_Qual['ANN'].to_numpy()
    sig = test_df_phiKK['ANN'].to_numpy()


    mc_counts,   edges = np.histogram(mc_ann_cut,   bins=bins)
    data_counts, _     = np.histogram(data_ann_cut, bins=bins)

    mc_scale = data_counts.sum() / mc_counts.sum()

    centers = 0.5 * (edges[:-1] + edges[1:])

    ratio = np.divide(data_counts, mc_counts * mc_scale,
                    out=np.full_like(data_counts, np.nan, dtype=float),
                    where=mc_counts > 0)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    ax_top.hist(mc_ann_cut,   bins=bins, histtype='step', density=False, weights=np.full(len(mc_ann_cut), mc_scale), label='MC (scaled)')
    ax_top.hist(data_ann_cut, bins=bins, histtype='step', density=False, label='data')
    ax_top.hist(sig, bins=bins, histtype='step', label='Signal')
    ax_top.set_ylabel('Count')
    ax_top.set_yscale('log')
    ax_top.set_title('ANN Output after quality cuts')
    ax_top.legend()

    ax_bot.step(centers, ratio, where='mid', color='k')
    ax_bot.axhline(y=1, color='k', linestyle='--')
    ax_bot.grid(True)
    ax_bot.set_ylabel('data / MC (scaled)')
    ax_bot.set_xlabel('ANN Output')
    ax_bot.set_ylim(0, 10)

    plt.tight_layout()

    if FakeGen:
        plt.savefig('/Users/mghrear/data/ML_data/2019_pass2/ensemble_results/plots/scaled/data_mc_comparison/'+'data_mc_comparison_QualCuts_FakeGen_run'+str(seed)+'.png')
    else:
        plt.savefig('/Users/mghrear/data/ML_data/2019_pass2/ensemble_results/plots/scaled/data_mc_comparison/'+'data_mc_comparison_QualCuts_run'+str(seed)+'.png')