#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 09:55:19 2021

@author: paul
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc


# Number of binders against RMSE

results = np.loadtxt("./results.txt", dtype = str)
mhc_names = results[:,0]
n_binders = results[:,1].astype(int)
PSSM_errors = results[:,2].astype(float)
SMM_errors = results[:,3].astype(float)
ANN_errors = results[:,4].astype(float)

plt.figure()
data=[(PSSM_errors, 'PSSM_errors'), (SMM_errors, 'SMM_errors'), (ANN_errors, 'ANN_errors')]
x = n_binders
for partition, label in data:
    y = partition
    plt.scatter(x, y, label=label, marker='.')
plt.legend(frameon=False)
plt.xlabel('Number of binders in the data set')
plt.ylabel('Root Mean Squared Error')
plt.show()

plt.figure()
data=[(SMM_errors, 'SMM_errors'), (ANN_errors, 'ANN_errors')]
x = n_binders
for partition, label in data:
    y = partition
    plt.scatter(x, y, label=label, marker='.')
plt.legend(frameon=False)
plt.xlabel('Number of binders in the data set')
plt.ylabel('Root Mean Square Error')
plt.show()

a_s_p, a_p_s, s_a_p, s_p_a, p_a_s, p_s_a = 0,0,0,0,0,0

for i in range(len(n_binders)) :
    if ANN_errors[i] < SMM_errors[i] and SMM_errors[i] < PSSM_errors[i] :
        a_s_p += 1
    elif ANN_errors[i] < PSSM_errors[i] and PSSM_errors[i] < SMM_errors[i] :
        a_p_s += 1
    elif SMM_errors[i] < ANN_errors[i] and ANN_errors[i] < PSSM_errors[i] :
        s_a_p += 1
    elif SMM_errors[i] < PSSM_errors[i] and PSSM_errors[i] < ANN_errors[i] :
        s_p_a += 1
    elif PSSM_errors[i] < ANN_errors[i] and ANN_errors[i] < SMM_errors[i] :
        p_a_s += 1
    else :
        p_s_a += 1

print("Ranking of techniques according to RMSE:")
print("ANN > SMM > PSSM {} times".format(a_s_p))
print("ANN > PSSM > SMM {} times".format(a_p_s))
print("SMM > ANN > PSSM {} times".format(s_a_p))
print("SMM > PSSM > ANN {} times".format(s_p_a))
print("PSSM > ANN > SMM {} times".format(p_a_s))
print("PSSM > SMM > ANN {} times".format(p_s_a))
print("\n")
    



# Number of binders against AUC

finished_dir = "./data/finished/"
mhc_list = os.listdir(finished_dir)

PSSM_AUC = []
SMM_AUC = []
ANN_AUC = []
n_binders = []

for mhc in mhc_list :
    predictions = np.loadtxt(finished_dir+mhc+"/predictions", dtype = str)
    reference = [1 if predictions[i,2] == "True" else 0 for i in range(len(predictions))]
    PSSM_predictions = predictions[:,3].astype(float)
    SMM_predictions = predictions[:,4].astype(float)
    ANN_predictions = predictions[:,5].astype(float)
    index = int(np.where(mhc_names == mhc)[0])
    n_binders.append(int(results[index,1]))

    fpr, tpr, threshold = roc_curve(reference, PSSM_predictions)    
    PSSM_AUC.append(auc(fpr,tpr))
    fpr, tpr, threshold = roc_curve(reference, SMM_predictions)    
    SMM_AUC.append(auc(fpr,tpr))
    fpr, tpr, threshold = roc_curve(reference, ANN_predictions)    
    ANN_AUC.append(auc(fpr,tpr))
    
plt.figure()
data=[(PSSM_AUC, 'PSSM_AUC'), (SMM_AUC, 'SMM_AUC'), (ANN_AUC, 'ANN_AUC')]
x = n_binders
for partition, label in data:
    y = partition
    plt.scatter(x, y, label=label, marker='.')
plt.legend(frameon=False)
plt.xlabel('Number of binders in the data set')
plt.ylabel('AUC')
plt.show()


a_s_p, a_p_s, s_a_p, s_p_a, p_a_s, p_s_a = 0,0,0,0,0,0

for i in range(len(n_binders)) :
    if ANN_AUC[i] > SMM_AUC[i] and SMM_AUC[i] > PSSM_AUC[i] :
        a_s_p += 1
    elif ANN_AUC[i] > PSSM_AUC[i] and PSSM_AUC[i] > SMM_AUC[i] :
        a_p_s += 1
    elif SMM_AUC[i] > ANN_AUC[i] and ANN_AUC[i] > PSSM_AUC[i] :
        s_a_p += 1
    elif SMM_AUC[i] > PSSM_AUC[i] and PSSM_AUC[i] > ANN_AUC[i] :
        s_p_a += 1
    elif PSSM_AUC[i] > ANN_AUC[i] and ANN_AUC[i] > SMM_AUC[i] :
        p_a_s += 1
    else :
        p_s_a += 1
        
print("Ranking of techniques according to AUC:")
print("ANN > SMM > PSSM {} times".format(a_s_p))
print("ANN > PSSM > SMM {} times".format(a_p_s))
print("SMM > ANN > PSSM {} times".format(s_a_p))
print("SMM > PSSM > ANN {} times".format(s_p_a))
print("PSSM > ANN > SMM {} times".format(p_a_s))
print("PSSM > SMM > ANN {} times".format(p_s_a))
