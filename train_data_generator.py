"""The program uses the segmented data created by `data_splitter.py` and further splits into a fixed size files which can be used for training purposes."""

import pandas as pd
from pathlib import Path
import numpy as np
from train_sampler import TrainAcousticSampler
# Directory where continuous segment data is kept. This data is created via data splitter. 
train_data_directory = 'D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data/'
max_itr_prc = 0.015;
sample_size = 150000


def getFilenames(directory=train_data_directory):
    ''' Function returns all csv files resides within the directory. However,
        the search is not recursive
    '''
    import glob;
    path = directory + '*.csv';
    files = glob.glob(path)
    return files;


def write(filename, data_df):
    seg_file = Path(filename)
    if seg_file.is_file():
            print('Skipping file', filename)
    else:
        data_df.to_csv(filename, columns=headers, index=False)
    

# Fixing the chunk size to 150000 which is equivalent to the test_random_forest files record size. This will help in keeping
# a nice synchronization between training and test_random_forest.
headers = ['acoustic_data', 'time_to_failure']
segment_data = []; record_count = 0;segment_id = 0;
print('Splitting training files .... ')
tas = TrainAcousticSampler();
for file in getFilenames():
    chunk_count = 0;
    df = pd.read_csv(file, low_memory=False);
    total_records = df.shape[0];
    max_indx = total_records - sample_size - 1;
    print('Total records in file ', file, 'is', total_records)
    max_itr = int((max_itr_prc / 100) * total_records)
    if max_indx > max_itr:
            for _ in range(0, max_itr):
                start_indx = np.random.randint(0, max_indx)
                sample_data = pd.DataFrame(df.iloc[start_indx : start_indx + sample_size])
                tas.add(sample_data)
                # file_to_generate = 'data/train_data_new/seg_' + str(segment_id).zfill(2) + str(chunk_count) + '.csv'
                # write(file_to_generate, sample_data);
                record_count += 1
                chunk_count += 1
    else:
        tas.add(pd.DataFrame(df))
    segment_id += 1
tas.saveData()
print('Training data has been split into', str(record_count), ' segments successfully')

# Segment_1638 row number 129586 is the partition. Files after this belongs to different 
