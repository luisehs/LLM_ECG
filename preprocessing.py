import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy import signal

import wfdb
import ast
import os


# variables
path = os.getcwd()
pathData = '../../../Workspace/ECGLLM/physionet.org/files/ptb-xl/1.0.3/'
ptbxl_DB = 'ptbxl_database.csv'
scp_statements = 'scp_statements.csv'

s_rate = 100 # Frequency Sampling or Sampling rate
f0 = 50 # Noise Frequency Hz
Q = 30 # Quality Factor
ecg_method = 'emrich2023' # process and clean ECG by emrich2023

# Load raw data wdfb
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# Load and convert annotation data
Y = pd.read_csv(pathData+ptbxl_DB, index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, s_rate, pathData)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(pathData+scp_statements, index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Aggregate diagnostic in dictionary
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Preprocessing FFT
_X = np.transpose(X, (0,2,1))
# X_fft = np.zeros_like(_X)
X_fftabs = np.zeros_like(_X)

for i in range(len(_X)):
    for j in range(len(_X[i])):
        # X_fft[i][j] = np.fft.fft(_X[i][j])
        X_fftabs[i][j] = np.abs(np.fft.fft(_X[i][j]))

# Quality Notch
b_notch, a_notch = signal.iirnotch(f0, Q, s_rate)
X_notch = np.zeros_like(X_fftabs)

for i in range(len(X_fftabs)):
    for j in range(len(X_fftabs[i])):
        X_notch[i][j] = signal.filtfilt(b_notch, a_notch, X_fftabs[i][j])

# wfdb.plot_items(X[0,:,0])
# wfdb.plot_items(_X[0,0,:])
# wfdb.plot_items(X_fftabs[0,0,:])
# wfdb.plot_items(X_notch[0,0,:])

# Windows 10-seconds
def clean_emrich(signal):
    cleaned = nk.ecg_clean(signal, s_rate, ecg_method)
    return cleaned

def peak_emrich(signal):
    _, info = nk.ecg_peaks(signal, s_rate, ecg_method)
    peaks = info['ECG_R_Peaks']
    return peaks

def windows(signal ,peaks):
    for i in range(len(peaks)):


