import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import *


import tensorflow as tf
import pywt

# Project Info
JobDir = '/tensorvolar/joey/pywavelet_movingaverage'
TrainingDataDir = os.path.join(JobDir, '0831traindata')
# Dataset Info
LengthOfSeq = 7500
NumOfVars = 4
NumOfData = 40

class Dataset:
    def __init__(self, fullFilename):
        self.dataList = np.arange(LengthOfSeq*NumOfVars).reshape(NumOfVars, LengthOfSeq)
        self.dataFile = pd.read_excel(io=fullFilename, header=None)
        self.quality = np.float32(self.dataFile.iloc[-1, 0].split(':')[-1])
        self.dataList = self.dataFile.iloc[0:-1]

def isXlsFile(filename, file_dir):
    filename = os.path.join(file_dir, filename)
    [name, ext] = os.path.splitext(filename)
    if os.path.isfile(filename):
        if ext == '.xls':
            return 1
        else:
            return 0

def getXlsFile(data_dir):
    fullFilename = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if isXlsFile(f, data_dir) == 1]
    return fullFilename

def sortByQuality(dataset):
    return dataset.quality

def getSortedDataset(data_dir):
    xlsFileList = getXlsFile(data_dir)
    datasets = []
    for filename in xlsFileList:
        datasets.append(Dataset(filename))
    datasets.sort(key=sortByQuality, reverse=False)
    return datasets

pd.options.display.max_rows = 10
datasets = getSortedDataset(TrainingDataDir)

def data_preprocessing(custom_datasets):
    size = len(custom_datasets)
    df=pd.DataFrame(columns=np.arange(353))
    for i in range(size):
        row = np.array([],np.float32)
        for j in range(4):
            col = np.array(datasets[i].dataList[j],dtype=np.float32)
            afterwt = pywt.wavedec(col,'db8')
            row = np.hstack((row,afterwt[0],afterwt[1]))
        row = np.hstack((row,datasets[i].quality))
        df.loc[i] = row
    return df

        
            
after_processed = data_preprocessing(datasets)
print(after_processed) 
after_processed.to_csv('0903traindata.csv', index = False)
