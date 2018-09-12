import numpy as np
import pandas as pd

def load_data(filename,numofdata):
    #there are 48 variables in csv, the last column is quality
    datalist = np.arange(numofdata*112).reshape(112,numofdata)
    df = pd.read_csv(filename)
    quality = np.array(df.iloc[:,-1],dtype=np.float32)
    datalist = df.iloc[:,0:-1]
    print("Data loaded")
    return df, quality, datalist

def load_test_data(filename,numofdata):
    datalist = np.arange(numofdata*112).reshape(112,numofdata)
    df = pd.read_csv(filename)
    datalist = df.iloc[:,:]
    print("Data loaded")
    return df, datalist

def load_0903_test_data(filename,numofdata):
    datalist = np.arange(numofdata*176).reshape(176,numofdata)
    df = pd.read_csv(filename)
    datalist = df.iloc[:,:]
    print("Data loaded")
    return df, datalist

def load_0903_data(filename,numofdata):
    #there are 48 variables in csv, the last column is quality
    datalist = np.arange(numofdata*176).reshape(176,numofdata)
    df = pd.read_csv(filename)
    quality = np.array(df.iloc[:,-1],dtype=np.float32)
    datalist = df.iloc[:,0:-1]
    print("Data loaded")
    return df, quality, datalist
