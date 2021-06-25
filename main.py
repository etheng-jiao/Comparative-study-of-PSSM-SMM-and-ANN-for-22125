import numpy as np
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import time
import math

import ANN
import PSSM
import SMM

start = time.time()

def mse(y_target_array, y_pred_array):
    return np.sqrt(((y_target_array - y_pred_array)**2).mean())

# with open("./results.txt","a") as f :
#     f.write("MHC\tN_binders\tPSSM_error\tSMM_error\tANN_error\n")

data_dir = "./data/"
mhc_dir = data_dir + "mhc/"

mhc_list = os.listdir(mhc_dir)
binder_threshold = 1-math.log(500)/math.log(50000)

number_of_binders = []
PSSM_errors = []
SMM_errors = []
ANN_errors = []

for mhc in mhc_list:

    mhc_start = time.time()

    print("Started ", mhc)
    dataset = []

    np.random.seed(11)

    for i in range(5):
        filename = mhc_dir+mhc+"/c00" + str(i)
        dataset.append(np.loadtxt(filename, dtype = str))
        np.random.shuffle(dataset[i])
        dataset[i][:,2] = dataset[i][:,1].astype(float) > binder_threshold
  
    whole_dataset = np.concatenate(dataset, axis = 0)

    prediction_PSSM = [None, None, None, None, None]
    prediction_SMM = [None, None, None, None, None]
    prediction_ANN = [None, None, None, None, None]

    for outer_index in range(5) :

        print("\tOuter index: {}/5".format(outer_index+1))

        evaluation_data = dataset[outer_index]

        SMM_matrices = []
        inner_indexes = [i for i in range(5)]
        inner_indexes.remove(outer_index)

        for inner_index in inner_indexes :

            print("\t\tTraining/testing: {}/4".format(inner_indexes.index(inner_index)+1))

            test_data = dataset[inner_index]

            train_indexes = inner_indexes.copy()
            train_indexes.remove(inner_index)

            train_data = [dataset[i] for i in train_indexes]
            train_data = np.concatenate(train_data, axis = 0)

            print("\t\t\tTraining SMM ...")
            SMM_matrices.append(SMM.train(train_data, test_data))

            print("\t\t\tTraining ANN ...")
            synfile_name = mhc_dir+mhc+"/synfile"+"c00"+str(inner_index)
            ANN.train(train_data, test_data, synfile_name)

        print("\t\tTraining PSSM ...")
        PSSM_train = np.concatenate([dataset[i][:,0] for i in inner_indexes], axis = 0)
        PSSM_matrix = PSSM.train(PSSM_train)

        # Evaluating error
        evaluation_SMM = [None, None, None, None]
        evaluation_ANN = [None, None, None, None]

        print("\t\tEvaluating ...")
        for i in range(4):
            evaluation_SMM[i] = np.array(SMM.evaluate(evaluation_data, SMM_matrices[i])).reshape(-1,1)
            evaluation_ANN[i] = np.array(ANN.evaluate(evaluation_data, mhc_dir+mhc+"/synfile"+"c00"+str(inner_indexes[i]))).reshape(-1,1)
    
        prediction_PSSM[outer_index] = np.array(PSSM.evaluate(evaluation_data, PSSM_matrix))
        prediction_SMM[outer_index] = np.mean(np.concatenate(evaluation_SMM, axis = 1), axis = 1)
        prediction_ANN[outer_index] = np.mean(np.concatenate(evaluation_ANN, axis = 1), axis = 1)
    
    predictions_PSSM = np.concatenate(prediction_PSSM, axis = 0).reshape(-1,1)
    predictions_SMM = np.concatenate(prediction_SMM, axis = 0).reshape(-1,1)
    predictions_ANN = np.concatenate(prediction_ANN, axis = 0).reshape(-1,1)
        
    np.savetxt(mhc_dir+mhc+"/predictions", np.concatenate((whole_dataset,predictions_PSSM,predictions_SMM,predictions_ANN), axis = 1), fmt = "%s")
        
    PSSM_errors.append(mse(whole_dataset[:,1].astype(float), predictions_PSSM[:,0]))
    SMM_errors.append(mse(whole_dataset[:,1].astype(float), predictions_SMM[:,0]))
    ANN_errors.append(mse(whole_dataset[:,1].astype(float), predictions_ANN[:,0]))
    number_of_binders.append(np.count_nonzero(whole_dataset[:,2] == "True"))
    
    with open("./results.txt","a") as f :
        f.write(mhc+"\t"+str(number_of_binders[-1])+"\t"+str(PSSM_errors[-1])+"\t"+str(SMM_errors[-1])+"\t"+str(ANN_errors[-1])+"\n")

    print("\t{} completed in {}s".format(mhc, time.time()-mhc_start))

    print("\n--------------------------\n")

# plt.figure(figsize=(15,4))
data=[(PSSM_errors, 'PSSM_errors'), (SMM_errors, 'SMM_errors'), (ANN_errors, 'ANN_errors')]
x = number_of_binders
for partition, label in data:
    y = partition
    plt.scatter(x, y, label=label, marker='.')
plt.legend(frameon=False)
plt.xlabel('Number of binders in the data set')
plt.ylabel('Root Mean Squared Error')
plt.show()

print("Overall time spent: {} min".format((time.time()-start)/60))
