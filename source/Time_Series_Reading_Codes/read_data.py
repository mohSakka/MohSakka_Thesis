# -*- coding: utf-8 -*-
import pandas as pd

def read_data(d_type):
    ankle_data = pd.read_csv("../../data/" + d_type + "_data.csv")
    ankle_data.dropna(inplace=True)
    return ankle_data
# chest_data = pd.read_csv("../../data/ankle_data.csv")
