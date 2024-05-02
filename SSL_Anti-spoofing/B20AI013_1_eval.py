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
from matplotlib import pyplot as plt

from model import Model

# Configuration
experiment_save_name = 'custom_subset_eval_1'
root_directory_path_rishabh_subset = '../Dataset_Speech_Assignment' 
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = None
random_seed = 42
torch.manual_seed(random_seed)

# Load the model
model = Model(args, 'cpu')

# Load the best state dict .pth file
state_dict_path = '../models/Best_LA_model_for_DF.pth'
state_dict = torch.load(state_dict_path, map_location='cpu')
for key in list(state_dict.keys()):
    state_dict[key.replace('module.', '')] = state_dict.pop(key)
model.load_state_dict(state_dict)


# Load the dataset
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def loader_rishabhsubset(samplepath):
    cut = 64600
    X, fs = librosa.load(samplepath, sr=16000)
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    return x_inp


dataset = DatasetFolder(root_directory_path_rishabh_subset, loader=loader_rishabhsubset, extensions=('wav', 'mp3', ))

# Load the dataloader
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define evaluation utilities
def compute_det_curve(target_scores, nontarget_scores):
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_eer_auc(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    auc = np.trapz(1 - frr, far)
    return eer, auc, thresholds[min_index]

# Evaluation script
def eval_script(model, loader, device, savepath='B20AI013_evaldir', printlogs=True, savelogs=True, savefigs=True):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        scores = []
        truths = []
        for xs, labels in tqdm(loader):
            xs, labels = xs.to(device), labels.to(device)
            outputs = model(xs)
            scores.extend((outputs[:, 1]).data.cpu().numpy().ravel().tolist())
            truths.extend(labels.tolist())

        fpr, tpr, thresholds = metrics.roc_curve(np.array(truths), np.array(scores), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)

        if printlogs:
            print('FINAL SCORES BELOW!!!!!!!!!!')
            print('eer:', eer, '        auc:', auc,  "    thresh:", thresh)
            print('FINAL SCORES ABOVE!!!!!!!!!!')
            print()
        
        if savelogs:
            with open(os.path.join(savepath, experiment_save_name + '.json'), 'w') as f:
                json.dump(
                    {
                        'eer': float(f'{eer}'),
                        'auc': float(f'{auc}'),
                        'thresh': float(f'{thresh}'),
                    }, f, indent=4
                )
            
        if savefigs:    
            # Plot ROC curve and save
            image_save_path = os.path.join(savepath, experiment_save_name + '.png')
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(image_save_path)

    return eer, auc

eval_script(model, loader, device, printlogs=False, savelogs=False, savefigs=True)

# after this, start doing shit inside train.py. figure out the training and then train for 1 epoch and then run eval straight after the training and recompute everything.




# threshold = -3.4
# preds = [int(score > threshold) for score in scores]
# total = sum([int(pred == pred) for (pred, truth) in zip(preds, truths)])
# correct = sum([int(pred == truth) for (pred, truth) in zip(preds, truths)])
# total, correct