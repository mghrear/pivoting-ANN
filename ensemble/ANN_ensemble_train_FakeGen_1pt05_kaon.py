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
import os, random


# Print and store device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Number of class in invariant mass used by the adversary
Num_classes = 5
# Hyper-parameter for adversarial training of the classifier
lambda_ = 100.0

# batch size
bsize = 2000

def seed_everything(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

def full_loss(output_clas, output_adv, target_clas, target_adv, w, lambda_):

    loss1 = BCE_loss(output_clas, target_clas )  # classification loss
    sample_losses2 = CE_loss(output_adv, target_adv)  # adversarial losses, since we set reduction='none', this is now a vector of losses, one per sample in the batch

    # Weighted mean loss over non-zero weights
    nonzero = w > 0
    loss2 = (sample_losses2[nonzero] * w[nonzero]).sum() / w[nonzero].sum()

    loss = loss1 - lambda_*loss2

    return loss

for seed in np.arange(100)+5:

    print("Starting run with seed: ", seed)
    seed_everything(int(seed))

    # Read tritrig, wab, phiKK, and data files
    df_tritrig = pd.read_pickle('/Users/mghrear/data/ML_data/patch/2021_v9_pass5_tritrig_QualCuts.pk')
    df_tritrig['PhiKK'] = 0.0
    df_phiKK = pd.read_pickle('/Users/mghrear/data/ML_data/patch/2021_v9_pass5_FakeGen_1pt05_kaon_QualCuts.pk')
    df_phiKK['PhiKK'] = 1.0
    df_wab = pd.read_pickle('/Users/mghrear/data/ML_data/patch/2021_v9_pass5_wab_QualCuts.pk')
    df_wab['PhiKK'] = 0.0 # Add label


    print("tritrig: ", len(df_tritrig))
    print("phiKK: ", len(df_phiKK))
    print("wab: ", len(df_wab))

    # Make Training set with 20000 tritrig, 4000 phiKK, 4000 wab
    tritrig_train = df_tritrig[0:20000]
    wab_train = df_wab[0:3000]
    phiKK_train = df_phiKK[0:3000]

    # Combine and shuffle all training data
    df_train = pd.concat([tritrig_train, phiKK_train, wab_train], ignore_index=True, sort=False)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Get edges in invariant mass classes for one-hot encoding
    # To be used for adverserial labels
    # The edges are chosen such that each bin has equal number of background events
    one_hot_edges = mytools.get_one_hot_edges(mytools.get_InvM( pd.concat([df_tritrig, df_wab], ignore_index=True, sort=False) ), n_bins=Num_classes)

    # Split the training data into training and validation sets
    df_train, df_val = train_test_split(df_train, test_size=0.33, random_state=42)

    # Make X and y for training and validation sets
    # the adverserial network has its own labels
    X_train = df_train.drop(columns=['PhiKK'])
    y_train = df_train['PhiKK']
    y_adv_train = mytools.get_adv_labels( mytools.get_InvM(df_train), one_hot_edges)
    X_val = df_val.drop(columns=['PhiKK'])
    y_val = df_val['PhiKK']
    y_adv_val = mytools.get_adv_labels( mytools.get_InvM(df_val), one_hot_edges)

    # Make testing set with remaining events
    tritrig_test = df_tritrig[20000:]
    tritrig_test['type']= 'tritrig'
    wab_test = df_wab[3000:]
    wab_test['type']= 'wab'
    phiKK_test = df_phiKK[3000:]
    phiKK_test['type']= 'phiKK'

    df_test = pd.concat([tritrig_test, phiKK_test,wab_test], ignore_index=True, sort=False)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True) # Now shuffle the combined dataframe

    X_test = df_test.drop(columns=['PhiKK','type'])
    y_test = df_test['PhiKK']
    y_adv_test = mytools.get_adv_labels( mytools.get_InvM(df_test), one_hot_edges)

    df_test["InvM"] = mytools.get_InvM(df_test) 

    # Convert to tensor dataset and dataloader
    train_dataset = TensorDataset(torch.from_numpy(X_train.to_numpy().astype(np.float32))  , torch.from_numpy(y_train.to_numpy().astype(np.float32)).unsqueeze(1)  )
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(X_val.to_numpy().astype(np.float32))  , torch.from_numpy(y_val.to_numpy().astype(np.float32)).unsqueeze(1)  )
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=True)

    test_dataset = TensorDataset(torch.from_numpy(X_test.to_numpy().astype(np.float32)))
    test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=False)

    clas = mytools.Classifier(in_features=X_train.shape[1]).to(device)
    criterion_clas = nn.BCEWithLogitsLoss()
    optimizer_clas = optim.Adam(clas.parameters(), lr=1e-5)
    scheduler_clas = torch.optim.lr_scheduler.ExponentialLR(optimizer_clas, gamma=0.9999)

    # Implement early stopping in training loop
    # Stop if validation loss has not decreased for the last [patience] epochs
    # The model with the lowest loss is stored
    patience = 10

    Training_losses = np.array([])
    Validation_losses = np.array([])

    epochs = 2500
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        Training_losses = np.append(Training_losses, mytools.train_clas(train_loader, clas, criterion_clas, optimizer_clas, scheduler_clas, device))
        Validation_losses = np.append(Validation_losses, mytools.validate_clas(val_loader, clas, criterion_clas, device))
        
        # Keep a running copy of the model with the lowest loss
        if Validation_losses[-1] == np.min(Validation_losses):
            final_classifier = copy.deepcopy(clas)
        
        if len(Validation_losses) > patience:
            if np.sum((Validation_losses[-1*np.arange(patience)-1] - Validation_losses[-1*np.arange(patience)-2]) < 0) == 0:
                print("Stopping early!")
                break
                
    adv = mytools.Adversary(n_classes=Num_classes).to(device)
    criterion_adv = nn.CrossEntropyLoss(reduction='none')  # returns loss per sample so we can weight it
    opt_adv = torch.optim.Adam(adv.parameters(), lr=1e-5)
    scheduler_adv = torch.optim.lr_scheduler.ExponentialLR(opt_adv, gamma=0.999)

    train_adv_dataset = TensorDataset(torch.from_numpy(X_train.to_numpy() .astype(np.float32))  , torch.from_numpy(y_adv_train), (torch.from_numpy(y_train.to_numpy().astype(np.float32))==0).float())
    train_adv_loader = DataLoader(train_adv_dataset, batch_size=bsize, shuffle=True)

    val_adv_dataset = TensorDataset(torch.from_numpy(X_val.to_numpy() .astype(np.float32))  , torch.from_numpy(y_adv_val), (torch.from_numpy(y_val.to_numpy().astype(np.float32))==0).float() )
    val_adv_loader = DataLoader(val_adv_dataset, batch_size=bsize, shuffle=True)

    # Weights are not relavent for testing
    test_adv_dataset = TensorDataset(torch.from_numpy(X_test.to_numpy() .astype(np.float32))  , torch.from_numpy(y_adv_test))
    test_adv_loader = DataLoader(test_adv_dataset, batch_size=bsize, shuffle=True)

    patience = 6

    Training_losses = np.array([])
    Validation_losses = np.array([])

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        Training_losses = np.append(Training_losses, mytools.train_adv(train_adv_loader, final_classifier, adv, criterion_adv, opt_adv, scheduler_adv, device))
        Validation_losses = np.append(Validation_losses, mytools.validate_adv(val_adv_loader, final_classifier, adv, criterion_adv, device))
        
        # Keep a running copy of the model with the lowest loss
        if Validation_losses[-1] == np.min(Validation_losses):
            final_adv = copy.deepcopy(adv)
        
        if len(Validation_losses) > patience:
            if np.sum((Validation_losses[-1*np.arange(patience)-1] - Validation_losses[-1*np.arange(patience)-2]) < 0) == 0:
                print("Stopping early!")
                break

    train_full_dataset = TensorDataset(torch.from_numpy(X_train.to_numpy().astype(np.float32))  , torch.from_numpy(y_train.to_numpy().astype(np.float32)).unsqueeze(1), torch.from_numpy(y_adv_train),  (torch.from_numpy(y_train.to_numpy().astype(np.float32))==0).float())
    train_full_loader = DataLoader(train_full_dataset, batch_size=bsize, shuffle=True)

    val_full_dataset = TensorDataset(torch.from_numpy(X_val.to_numpy().astype(np.float32))  , torch.from_numpy(y_val.to_numpy().astype(np.float32)).unsqueeze(1),  torch.from_numpy(y_adv_val),  (torch.from_numpy(y_val.to_numpy().astype(np.float32))==0).float())
    val_full_loader = DataLoader(val_full_dataset, batch_size=bsize, shuffle=True)

    BCE_loss = nn.BCEWithLogitsLoss()
    CE_loss = nn.CrossEntropyLoss(reduction='none') # returns loss per sample so we can weight it
    
    optimizer_clas = optim.Adam(final_classifier.parameters(), lr=1e-7,betas=(0.5, 0.999))
    optimizer_adv = optim.Adam(final_adv.parameters(), lr=1e-5,betas=(0.5, 0.999))

    scheduler_clas = torch.optim.lr_scheduler.ExponentialLR(optimizer_clas, gamma=0.999)
    scheduler_adv = torch.optim.lr_scheduler.ExponentialLR(optimizer_adv, gamma=0.999)

    Training_losses_clas = np.array([])
    Training_losses_adv = np.array([])
    Validation_losses_clas = np.array([])
    Validation_losses_adv = np.array([])
    diff_scores = np.array([])


    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        # Get validation losses
        E_clas_val_loss, E_adv_val_loss = mytools.validate_full(val_full_loader, final_classifier, final_adv, full_loss, lambda_, criterion_adv, device)
        Validation_losses_clas = np.append(Validation_losses_clas, E_clas_val_loss)
        Validation_losses_adv = np.append(Validation_losses_adv, E_adv_val_loss)

        # Train adversary and classifier
        E_clas_training_loss , E_adv_training_loss = mytools.train_full(train_full_loader, final_classifier, final_adv, full_loss, lambda_, criterion_adv, optimizer_clas, optimizer_adv, scheduler_clas, scheduler_adv, device)
        Training_losses_clas = np.append(Training_losses_clas, E_clas_training_loss)
        Training_losses_adv = np.append(Training_losses_adv, E_adv_training_loss)

        # Get the diff_score
        df_test["Class_adv"] = mytools.test_clas(test_loader, final_classifier, device)
        df_bkg = df_test.loc[ (df_test.type == 'tritrig') | (df_test.type == 'wab')]
        per = np.percentile(df_bkg['Class_adv'],99)
        df_cut = df_bkg[df_bkg['Class_adv']>per].reset_index(drop=True)
        x1 = 1000* df_bkg.InvM.values
        x2 = 1000*df_cut.InvM.values
        diff_scores = np.append(diff_scores, mytools.get_diff_score(x1,x2) )

        # Keep a running copy of the model with the lowest loss
        if diff_scores[-1] == np.min(diff_scores):
            final_clas_adv = copy.deepcopy(final_classifier)
            final_adv_adv = copy.deepcopy(final_adv)

    torch.save(final_clas_adv.state_dict(), "/Users/mghrear/data/ML_data/patch/ensemble_FakeGen_1pt05_kaon_QualCuts/classifier_adv_2021_v9_pass5_run"+str(seed)+"_limited.pt")
