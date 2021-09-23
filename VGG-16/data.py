#%%
import pandas as pd
import os
import shutil
import glob

#%%
train_df = pd.read_csv('./data/train_metadata.txt', header=None, sep = ' ')
test_df = pd.read_csv('./data/test_metadata.txt', header=None, sep = ' ')


#%%
train_list = train_df[1].tolist()
train_label = train_df[2].tolist()
test_list = test_df[1].tolist()
test_label = test_df[2].tolist()

#%%
for i in range(len(train_list)):
    sample = train_list[i]
    label = train_label[i]
    if label == 'COVID-19':    
        shutil.copy('./data/train/'+sample, './data/all/train/covid/'+sample)
    elif label == 'normal':    
        shutil.copy('./data/train/'+sample, './data/all/train/normal/'+sample)
    else:
        shutil.copy('./data/train/'+sample, './data/all/train/pneumonia/'+sample)
for i in range(len(test_list)):
    sample = test_list[i]
    label = test_label[i]
    if label == 'COVID-19':    
        shutil.copy('./data/test/'+sample, './data/all/test/covid/'+sample)
    elif label == 'normal':    
        shutil.copy('./data/test/'+sample, './data/all/test/normal/'+sample)
    else:
        shutil.copy('./data/test/'+sample, './data/all/test/pneumonia/'+sample)