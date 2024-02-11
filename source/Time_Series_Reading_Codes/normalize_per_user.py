# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:15:34 2024

@author: Asus
"""

# -*- coding: utf-8 -*-



from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd

def SQSumSq(df):
    # compute Signal Vector Magnitude per input
    return np.linalg.norm(df,axis=1)

# Function to create GAF images for activity recognition
def generate_user_normalised_data(sensor_data,train_user,val_users,test_users):
   
    num = 1
    # unique_users = sensor_data['user'].unique()
    unique_users = train_user+val_users+test_users
    # val_users = unique_users[-2]
    # test_users = unique_users[-1]

    # Combine all data for normalization
    combined_data_normalised = pd.DataFrame()
    val_data_norm = pd.DataFrame()
    test_data_norm = pd.DataFrame()
    for user in unique_users:
        print("current user:",user)
        num+=1
        # if num>3:
        #     break
        user_data = sensor_data[sensor_data['user'] == user]
        user_features = user_data.iloc[:, 0:-2].values
        signal_magnitude = SQSumSq(user_features)
        # print(signal_magnitude.shape)
        
        user_features = np.c_[user_features,signal_magnitude]

        user_data_local_norm = user_data.copy()
        user_data_local_norm.insert(3, "SignalMag", signal_magnitude, True)

        transformer = RobustScaler().fit(user_features)
        transformer.transform(user_features)  
        user_data_local_norm.iloc[:, 0:-2] =  user_features
        ###########

        if user in val_users:
          val_data_norm =  pd.concat([val_data_norm,user_data_local_norm])
        elif user in test_users:
          test_data_norm = pd.concat([test_data_norm,user_data_local_norm])
        else:
          combined_data_normalised =  pd.concat([combined_data_normalised, user_data_local_norm])
  
    return combined_data_normalised,val_data_norm,test_data_norm


