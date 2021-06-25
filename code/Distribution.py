#!/usr/bin/env python
# coding: utf-8

# In[85]:



import numpy as np
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning']= 0
import os
import time
import math

import seaborn as sns


# In[72]:


data_dir = " "
mhc_dir = data_dir + "project/mhc/"
mhc_list = os.listdir(mhc_dir)
print(mhc_list)
del mhc_list[4]
print(mhc_list)
dataset=[]
distribution=[]
mhc_B=[]
i=0
for mhc in mhc_list:
    #print (mhc)
    if mhc[0] =='B':
        mhc_B.append(mhc)
        file = mhc_dir + mhc + '/'+ mhc +'.dat'
        dataset.append(np.loadtxt(file,dtype=str))
        distribution.append(dataset[i][:,1].astype(float))
        i+=1


print(len(distribution))
    





# In[81]:
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
cm = plt.get_cmap('gist_rainbow')
#plt.figure()
print(cm)
fig,ax = plt.subplots()
ax.set_prop_cycle(color=['red', 'green', 'blue','orange','cyan','gray','dodgerblue','yellow','peru','fuchsia','lime','purple','gold','tan','lightseagreen','navy','springgreen','olive','maroon'])
#ax = fig.add_subplot(111)
for i, mhc in enumerate(mhc_B): #can swith to A file
    plt.figure()
    sns.kdeplot(distribution[i],shade=True,label= mhc)
    #ax.set_color(cm(i//4*float(4)/16))
    #ax.set_linestyle(LINE_STYLES[i%6])
    plt.legend(ncol=3) 
plt.show()  
    #plt.hist(distribution[i])
    
    #plt.legend(str(mhc),loc='best')
    
