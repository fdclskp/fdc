import numpy as np
import pandas as pd
import pathlib
import os


def getData():
    
	DataSet = np.loadtxt("9997.csv",delimiter=",")

    N_data = 200
    N_Loop = 4
    t = np.linspace(0, 2*np.pi*N_Loop, num=N_data)
    
    X = 0.8*np.sin(2.0*t)
    Y = 0.8*np.cos(1.0*t)
    
    N_train = int(N_data*0.75)
#    N_test = int(N_data*0.25)
    
    DataSet = np.c_[X, Y].astype(np.float32)
    
    train, test = np.array(DataSet[:N_train]), np.array(DataSet[N_train:])
    return train, test

print(getData())
