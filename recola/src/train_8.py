# library
import pandas as pd
import numpy as np
import pathlib
import plotly.express as px
from sklearn.cluster import KMeans
import scipy
from scipy.io import arff
from scipy import signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import librosa
import IPython.display
import json
from tqdm import tqdm
import statistics
import random
import os
from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

le = LabelEncoder()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

RANDOM_SEED = 42
seed_everything(RANDOM_SEED)

skf = StratifiedKFold(n_splits=3, random_state=RANDOM_SEED, shuffle=True)

# svr = SVR(kernel = "rbf",gamma = 1,C=100,epsilon = 0)
# cross_validate(svr,X,y_arousal,scoring="r2",cv=gkf, groups=speakers)

def train(X, y_arousal, y_valence, speakers):

    train_scores = []
    valid_scores = []
    oof_preds = np.zeros(len(X))

    v_train_scores = []
    v_valid_scores = []
    v_oof_preds = np.zeros(len(X))

    # arousal
    print("arousal")
    for kfoldidx, (train_idx, val_idx) in enumerate(skf.split(X, speakers)):

        print(kfoldidx, "-fold")
        # svr = SVR(kernel = "rbf")
        # svr = SVR(kernel = "rbf")
        svr = SVR(kernel = "rbf",gamma = 1,C=100,epsilon = 0.1)
        X_train = X[train_idx]
        y_train = y_arousal[train_idx]
        X_valid = X[val_idx]
        y_valid = y_arousal[val_idx]   

        # fit
        print("fit arousal")
        svr.fit(X_train, y_train)
        
        print("pred arousal")

        # train
        train_pred = svr.predict(X_train)
        train_score = mean_absolute_error(y_train, train_pred)
        
        # valid
        oof_pred = svr.predict(X_valid)
        valid_score = mean_absolute_error(y_valid, oof_pred)
        
        # return
        train_scores.append(train_score)
        valid_scores.append(valid_score)
        oof_preds[val_idx] = oof_pred

        # break

        
    # valence
    print("valence")
    # for train_idx, val_idx in skf.split(X, speakers):
    for kfoldidx, (train_idx, val_idx) in enumerate(skf.split(X, speakers)):
        # print("valence")
        # svr = SVR(kernel = "rbf")
        print(kfoldidx, "-fold")
        # svr = SVR(kernel = "rbf")
        svr = SVR(kernel = "rbf",gamma = 1,C=100,epsilon = 0.1)
        X_train = X[train_idx]
        y_train = y_valence[train_idx]
        X_valid = X[val_idx]
        y_valid = y_valence[val_idx]   

        # fit
        print("fit valence")
        svr.fit(X_train, y_train)
        print("pred valence")
        
        # train
        train_pred = svr.predict(X_train)
        train_score = mean_absolute_error(y_train, train_pred)
        
        # valid
        oof_pred = svr.predict(X_valid)
        valid_score = mean_absolute_error(y_valid, oof_pred)
        
        # return
        v_train_scores.append(train_score)
        v_valid_scores.append(valid_score)
        v_oof_preds[val_idx] = oof_pred

        # break


    return train_scores, valid_scores, oof_preds, v_train_scores, v_valid_scores, v_oof_preds


DUP = 8

print("duplicate:")
print(DUP)

_df_train = pd.read_csv("./output/train_unit" + str(DUP) + ".csv")
_X = np.load("./output/train_feature_unit" + str(DUP) + ".npy")

print(len(_X))


speakers = le.fit_transform(_df_train.speaker.values)
train_scores, valid_scores, oof_preds, v_train_scores, v_valid_scores, v_oof_preds = train(_X, _df_train.arousal.values, _df_train.valence.values, speakers)

print("train_scores", np.mean(train_scores))
print("valid_scores", np.mean(valid_scores))
print("v_train_scores", np.mean(v_train_scores))
print("v_valid_scores", np.mean(v_valid_scores))

np.save('./output/a_train_score_' + str(DUP), np.array(train_scores))
np.save('./output/a_valid_score_' + str(DUP), np.array(valid_scores))
np.save('./output/v_train_score_' + str(DUP), np.array(v_train_scores))
np.save('./output/v_valid_score_' + str(DUP), np.array(v_valid_scores))