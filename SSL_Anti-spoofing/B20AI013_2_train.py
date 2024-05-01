import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torch import Tensor
import os
import json
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from model import Model

###########################################
checkpoint_save_dir = 'checkpoint_save_dir'
experiment_save_name = 'train_eval_1'
# root_directory_path_rishabh_subset = r'C:\ai_sem_9\COURSES\supa3\Dataset_Speech_Assignment' # contains .wav and .mp3 files
# root_directory_path_testing = r'C:\ai_sem_9\COURSES\supa3\for-2seconds\testing'  # contains .wav files only.
root_directory_path_train = r'C:\ai_sem_9\COURSES\supa3\for-2seconds\training'  # contains .wav files only.
root_directory_path_validation = r'C:\ai_sem_9\COURSES\supa3\for-2seconds\validation'  # contains .wav files only.
batch_size = 2
nepochs = 5
learningrate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = None
random_seed = 42
torch.manual_seed(random_seed)
###########################################


# load the model
model = Model(args, 'cpu')

# load the best state dict .pth file
state_dict_path = r'C:\ai_sem_9\COURSES\supa3\models\Best_LA_model_for_DF.pth'
state_dict = torch.load(state_dict_path, map_location='cpu')
for key in list(state_dict.keys()):
    state_dict[key.replace('module.', '')] = state_dict.pop(key)

model.load_state_dict(state_dict)


# load the dataset
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def loader_rishabhsubset(samplepath):
    cut = 64600
    X, fs = librosa.load(samplepath, sr=16000)
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    return x_inp


train_dataset = DatasetFolder(root_directory_path_train, loader=loader_rishabhsubset, extensions=('wav', 'mp3', ))
eval_dataset  = DatasetFolder(root_directory_path_validation, loader=loader_rishabhsubset, extensions=('wav', 'mp3', ))

# load the dataloader. 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader  = DataLoader(eval_dataset , batch_size=batch_size, shuffle=True)



#define the eval utils
def compute_det_curve(target_scores, nontarget_scores):
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores
    return frr, far, thresholds



def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_eer_auc(target_scores, nontarget_scores):
    """ Returns equal error rate (EER), AUC, and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(1 - frr, far)

    return eer, auc, thresholds[min_index]


# define the eval script. it should output two lists:
# # 1. list_bonafide should contain the pred scores for REAL audios
# # 2. list_spoof should contain the pred scores for FAKE audios
def eval_script(model, loader, device, savepath='B20AI013_evaldir', savelogs=True, printlogs=True):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        correct = 0
        total = 0
        scores = []
        truths = []
        for xs, labels in tqdm(loader):
                xs, labels = xs.to(device), labels.to(device)
                outputs = model(xs)     
                scores.extend((outputs[:, 1]).data.cpu().numpy().ravel().tolist())
                truths.extend(labels.tolist())
    
    # #calculate EER, AUC
    # target_scores = [score for score, truth in zip(scores, truths) if truth == 0]
    # nontarget_scores = [score for score, truth in zip(scores, truths) if truth == 1]
    # eer_cm, _, _ = compute_eer_auc(target_scores, nontarget_scores)

    # calculate EER, AUC correctly
    fpr, tpr, thresholds = metrics.roc_curve(np.array(truths), np.array(scores), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    if printlogs:
        tqdm.write('FINAL SCORES BELOW!!!!!!!!!!')
        tqdm.write('eer:', eer, '        auc:', auc,  "    thresh:", thresh)
        tqdm.write('FINAL SCORES ABOVE!!!!!!!!!!')
        tqdm.write()
    
    if savelogs:
        with open(os.path.join(savepath, experiment_save_name), 'w') as f:
            json.dump(
                {
                    'eer':    float(f'{eer}'    ) ,
                    'auc':    float(f'{auc}'    ) ,
                    'thresh': float(f'{thresh}' ) ,
                }, f, indent=4
            )
    eer    = float(f'{eer}'    )
    auc    = float(f'{auc}'    )
    thresh = float(f'{thresh}' )
    return eer, auc, thresh


# eval_script(model, loader, device)

# after this, start doing shit inside train.py. figure out the training and then train for 1 epoch and then run eval straight after the training and recompute everything.

def train_script(model, train_loader, eval_loader, num_epochs, learning_rate, device, optimizer, criterion, trainloss=[], eer_train=[], eer_eval=[], auc_train=[], auc_eval=[], printlogs=True, savelogs=True):
    model.train()
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    criterion = criterion()

    for epoch in range(1, num_epochs+1):
        tqdm.write(f'begun training epoch#{epoch} out of {num_epochs}')
        tqdm.write(f'-' * 20)
        bar = tqdm(total=len(train_loader))
        for xs, labels in train_loader:
            optimizer.zero_grad()
            xs, labels = xs.to(device), labels.to(device)
            outputs = model(xs)     
            scores = torch.stack([outputs[:, 1], - outputs[:, 1]], dim=1)

            # print(scores.shape, labels.shape)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            trainloss.append(loss.item())
            bar.update(1)
            bar.set_postfix({
                'trainloss(tillnow)': np.mean(np.array(trainloss)),
            })
        bar.close()
        
        train_auc, train_eer, train_thresh = eval_script(model, train_loader, device, savelogs=False, printlogs=False)
        eval_auc, eval_eer, eval_thresh = eval_script(model, eval_loader, device, savelogs=False, printlogs=False)
        auc_train.append(train_auc )
        eer_train.append(train_eer )
        auc_eval.append (eval_auc  )
        eer_eval.append (eval_eer  )
        logs = {
            'checkpoint_save_dir': checkpoint_save_dir,
            'experiment_save_name': experiment_save_name,
            'save_location': os.path.join(checkpoint_save_dir, experiment_save_name),
            'trainloss': trainloss,
            'auc_train': auc_train,
            'eer_train': eer_train,
            'auc_eval': auc_eval,
            'eer_eval': eer_eval,
            'training_epochs': epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
        }

        if printlogs:
            print(json.dumps(obj=logs, indent=4))

        if savelogs:
            torch.save({
                'model':model.state_dict(), 
                'logs': logs
            }, os.path.join(checkpoint_save_dir, experiment_save_name + '_ckpt.pt'))
            with open(os.path.join(checkpoint_save_dir, experiment_save_name + '_logs.json'), 'w') as f:
                json.dump(logs, f, indent=4)
    return trainloss, auc_train, eer_train, auc_eval, eer_eval

trainloss, auc_train, eer_train, auc_eval, eer_eval = [], [], [], [], []
trainloss, auc_train, eer_train, auc_eval, eer_eval = train_script(model, train_loader, eval_loader, nepochs, learningrate, device, torch.optim.Adam, torch.nn.CrossEntropyLoss, trainloss, eer_train, eer_eval, auc_train, auc_eval)

print('trainloss: ', trainloss[-1])
print('auc_train: ', auc_train[-1])
print('eer_train: ', eer_train[-1])
print('auc_eval: ', auc_eval[-1])
print('eer_eval: ', eer_eval[-1])



# threshold = -3.4
# preds = [int(score > threshold) for score in scores]
# total = sum([int(pred == pred) for (pred, truth) in zip(preds, truths)])
# correct = sum([int(pred == truth) for (pred, truth) in zip(preds, truths)])
# total, correct
