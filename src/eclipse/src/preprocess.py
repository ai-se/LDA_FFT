from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pandas as pd
import os
filename="eclipse-metrics-files-2.0.csv"
ROOT=os.getcwd()
path=ROOT+"/../data/"+filename

## drop_columns=['plugin', 'filename', 'post']
df=pd.read_csv(path,sep=";")
print(df.columns.values)