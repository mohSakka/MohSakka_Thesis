# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:25:53 2024

@author: Asus
"""
import numpy as np
from pyts.image import GramianAngularField


def generate_images(data,window_size,local_norm):
    images = []
    labels = []
    local_norm_data = np.zeros((data.shape[0],4))
    
    unique_users = data['user'].unique()
    num_missed = 0
    for user in unique_users:
        user_data = data[data['user'] == user]
        start_index = 0
        # print(user_data.head())
        normalized_features = user_data.iloc[:,0:-2]
        
            # Check if all labels in the window are the same

        while start_index + window_size <= normalized_features.shape[0]:
            # print(start_index)
            label = user_data.iloc[start_index]['labels']  # Use the label of the first entry in the window
            if user_data.iloc[start_index:start_index + window_size]['labels'].nunique() <= 1:

            # Extract and reshape window data for each column (x, y, z)
              curr_img = np.zeros((window_size,window_size,4))
              for col_index in range(normalized_features.shape[1]):
                  col_window = normalized_features.iloc[start_index:start_index + window_size, col_index].to_numpy()
                  # print("col_window:   ",col_window)
                  reshaped_col_window = col_window.reshape(1, window_size)
                  if local_norm:
                    mn = reshaped_col_window.min()
                    mx = reshaped_col_window.max()
                    reshaped_col_window = (reshaped_col_window - mn) / (mx - mn)
                    local_norm_data[start_index:start_index + window_size,col_index]=reshaped_col_window.copy()
                  # Create GAF images using GASF for each column
                  gasf = GramianAngularField()
                  image = gasf.fit_transform(reshaped_col_window)
                  curr_img[:,:,col_index] = image
                  # Add each channel image to the list
              images.append(curr_img)
              labels.append(label)
            else:
              num_missed+=1
              



            start_index += window_size

    print('num_missed:',num_missed)
   

    return local_norm_data,images, labels