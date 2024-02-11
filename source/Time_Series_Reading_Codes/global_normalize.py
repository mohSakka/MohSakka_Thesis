# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:23:41 2024

@author: Asus
"""

def global_normalize(train_data,val_data,test_data):

  # Normalize the entire dataset
  features = train_data.iloc[:, 0:-2].values  # Exclude 'user' and 'labels' columns
  mn = features.min(axis=0)
  mx = features.max(axis=0)
  X_std = (features - mn) / (mx - mn)
  train_data.iloc[:, 0:-2] = X_std
  val_features = val_data.iloc[:, 0:-2].values
  X_std = (val_features - mn) / (mx - mn)
  val_data_norm = val_data.copy()
  val_data_norm.iloc[:, 0:-2] = X_std
  test_features = test_data.iloc[:, 0:-2].values
  X_std = (test_features - mn) / (mx - mn)
  test_data_norm = test_data.copy()
  test_data_norm.iloc[:, 0:-2] = X_std
  return train_data,val_data_norm,test_data