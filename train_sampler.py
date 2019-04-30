import pandas as pd
from pathlib import Path
import random, string
import numpy as np
import scipy
from sklearn.linear_model.base import LinearRegression


class TrainAcousticSampler:
    
    def __init__(self):
        self.seg_id = 0;
        self.train_data_record_list = []
        self.train_modeled_data_filename = 'train_modeled_data_lot_of_features.csv'
        self.train_data_columns = ['segment_id', 'acc_mean', 'acc_sd', 'acc_min', 'acc_max', 'chg_acc_min', 'chg_acc_max', 'time_to_failure']
        self.test_data_columns = ['segment_id', 'acc_mean', 'acc_sd', 'acc_min', 'acc_max', 'chg_acc_min', 'chg_acc_max']
    
    def add(self, data_df):
        self.__load_data(data_df)
    
    def __load_data(self, data_df):
        df = pd.DataFrame(data_df)
        df['acoustic_data'] = pd.to_numeric(df['acoustic_data'], errors='coerce')
        df['change_in_data'] = df['acoustic_data'].diff()
        df['time_to_failure'] = pd.to_numeric(df['time_to_failure'], errors='coerce')
        df = df.dropna(axis=0);            
        # df['time_to_failure'] = df['time_to_failure'].apply(lambda x: x * 1000000)
        self.__record_keeper_train(df, self.__getSegmentName())
    
    def __integrate(self, y):
        from scipy.integrate import simps
        # The value of interval '0.000002667' is calculated based on the assumption that each file has 150000 records with time duration of 0.04 seconds
        return simps(y, dx=0.00000025)
    
    def __add_trend_feature(self, arr, abs_values=False):
        idx = np.array(range(len(arr)))
        if abs_values:
            arr = np.abs(arr)
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)
        return lr.coef_[0]
    
    def __classic_sta_lta(self, x, length_sta, length_lta):
    
        sta = np.cumsum(x ** 2)
    
        # Convert to float
        sta = np.require(sta, dtype=np.float)
    
        # Copy for LTA
        lta = sta.copy()
    
        # Compute the STA and the LTA
        sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
        sta /= length_sta
        lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
        lta /= length_lta
    
        # Pad zeros
        sta[:length_lta - 1] = 0
    
        # Avoid division by zero by setting zero values to tiny float
        dtiny = np.finfo(0.0).tiny
        idx = lta < dtiny
        lta[idx] = dtiny
    
        return sta / lta
    
    def __calc_change_rate(self, x):
        change = (np.diff(x) / x[:-1]).values
        change = change[np.nonzero(change)[0]]
        change = change[~np.isnan(change)]
        change = change[change != -np.inf]
        change = change[change != np.inf]
        return np.mean(change)
    
    def __getSegmentName(self):
        self.seg_id += 1
        return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits + str(self.seg_id)) for _ in range(8))
    
    def __createAdditionStatsAttributes(self, record, sample_df):
        x = sample_df['acoustic_data'].values
        record['q95'] = np.quantile(x, 0.95)
        record['q99'] = np.quantile(x, 0.99)
        record['q05'] = np.quantile(x, 0.05)
        record['q01'] = np.quantile(x, 0.01)
        record['trend'] = self.__add_trend_feature(x)
        record['abs_trend'] = self.__add_trend_feature(x, abs_values=True)
        record['classic_sta_lta_100_5000'] = self.__classic_sta_lta(x, 100, 5000).mean()
        record['iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        record['mean_change_rate'] = self.__calc_change_rate(sample_df['acoustic_data'])
        for windows in [10, 100, 1000]:
            x_roll_std = sample_df['acoustic_data'].rolling(windows).std().dropna().values
            x_roll_mean = sample_df['acoustic_data'].rolling(windows).mean().dropna().values
            
            record['ave_roll_std_' + str(windows)] = x_roll_std.mean()
            record[ 'std_roll_std_' + str(windows)] = x_roll_std.std()
            record[ 'max_roll_std_' + str(windows)] = x_roll_std.max()
            record[ 'min_roll_std_' + str(windows)] = x_roll_std.min()
            record[ 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
            record[ 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
            record[ 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
            record[ 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
            record[ 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
            record[ 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            record[ 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
            
            record[ 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
            record[ 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
            record[ 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
            record[ 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
            record[ 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
            record[ 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
            record[ 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
            record[ 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
            record[ 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
            record[ 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            record[ 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
        
    def __record_keeper_train(self, sample_df, seg_name):
        sample_stats = sample_df.describe()
        record = {'segment_id':seg_name, 'acc_mean' : sample_stats['acoustic_data']['mean'], 'acc_sd' : sample_stats['acoustic_data']['std'], 'acc_min' : sample_stats['acoustic_data']['min'], \
                  'acc_max' : sample_stats['acoustic_data']['max'], 'chg_acc_min' : sample_stats['change_in_data']['min'], 'chg_acc_max' : sample_stats['change_in_data']['max'], \
                   'time_to_failure' : sample_stats['time_to_failure']['50%'], 'auc':self.__integrate(sample_df['acoustic_data'].values)\
                   , 'median_25': sample_stats['acoustic_data']['25%'], 'median_50': sample_stats['acoustic_data']['50%'], \
                   'median_70': sample_stats['acoustic_data']['75%']}
        self.__createAdditionStatsAttributes(record, sample_df);
        self.train_data_record_list.append(record) 
        if len(self.train_data_record_list) >= 5000:
            self.saveData()
        
    def saveData(self):
        record_df = pd.DataFrame(self.train_data_record_list)
        seg_file = Path(self.train_modeled_data_filename)
        if seg_file.is_file():
            record_df.to_csv(self.train_modeled_data_filename, mode='a', index=False, header=False)
        else:
            record_df.to_csv(self.train_modeled_data_filename, mode='a', index=False)
        self.train_data_record_list.clear()
        print('Flushed intermediate records into file')
        
    def fit(self):
        seg_file = Path(self.train_modeled_data_filename)
        if seg_file.is_file():
                print('Data file already exists ..Fitting completed')
        else:
            self.__readFiles();
            print('Data fitting completed')
    
    def get(self):
        seg_file = Path(self.train_modeled_data_filename)
        if seg_file.is_file():
            return pd.read_csv(self.train_modeled_data_filename)
        else:
            raise ValueError('Execute fit before get .. ') 
