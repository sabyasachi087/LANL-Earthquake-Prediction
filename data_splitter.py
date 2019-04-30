"""The program splits the original training data set into segments where each segment belongs to one single experiment. 
The way to identify the change in experiment / segment is change in time_to_failure where instead of decreasing it will increase for once and decrease from there.
This program takes a lot of time (3 days) to split the data and hence requires tuning."""

import pandas as pd
import numpy as np

train_data_filename = 'D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/EarthquakePrediction/data/train.csv'
chunksize = 150000

segment_file = 1
for chunk in pd.read_csv(train_data_filename, chunksize=chunksize, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}):
    chunk.to_csv('data/train_data/seg_' + str(segment_file).zfill(4) + '.csv', index=False)
    print('Training file generated for segment', segment_file)
    segment_file += 1

print('Training data has been split into segments successfully')
