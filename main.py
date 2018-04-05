# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:07:14 2018

@author: Trace
"""

import pandas as pd
import ID3

data = pd.read_csv('tennis.csv')

new_id3 = ID3.ID3(data)
new_id3.train()
new_id3.tree.name

test = pd.read_csv('test.csv')

print(new_id3.predict(test))