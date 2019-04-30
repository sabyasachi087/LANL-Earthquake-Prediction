from sklearn.preprocessing.data import StandardScaler

import numpy as np
import pandas as pd


class LANLDataFormatter:
    
    def __init__(self, data_df, data_type='train', doTransform=True, doScale=False, columns_to_drop=['acc_mean', 'median_50', 'auc'], \
                  non_numeric_cols=['segment_id'], cols_to_keep=-1, most_dependent_columns=None):
        self.data_df = data_df
        self.doScale = doScale
        self.data_type = data_type
        self.doTransform = doTransform
        self.columns_to_drop = columns_to_drop
        self.non_numeric_cols = non_numeric_cols
        self.cols_to_log_trainsform = ['q99', 'q01', 'q01_roll_mean_10', 'q99_roll_mean_10', 'q99_roll_std_10', 'std_roll_std_10', \
                          'std_roll_mean_10', 'std_roll_std_100', 'std_roll_std_1000', 'q99_roll_std_100', 'abs_max_roll_std_10', \
                          'max_roll_std_10', 'abs_max_roll_std_100', 'max_roll_std_100', 'min_roll_mean_10', 'acc_sd', 'chg_acc_max'\
                          , 'chg_acc_min', 'acc_min', 'acc_max']
        self.cols_to_keep = cols_to_keep
        self.most_dependent_columns = most_dependent_columns
        if data_type == 'test':
            if not most_dependent_columns:
                raise ValueError('Most Dependent Column List is mandatory for test')
    
    def __to_numeric(self):
        all_cols = list(self.data_df.columns.values)
        for col in all_cols:
            if col not in self.non_numeric_cols:
                self.data_df[col] = pd.to_numeric(self.data_df[col], errors='coerce')
    
    def __clean(self):
        # Adjust all null or empty fields
        self.__to_numeric()
        self.data_df['time_per_max_range'] = np.divide(self.data_df['auc'].values, np.subtract(self.data_df['acc_max'].values, self.data_df['acc_min'].values))
        dist = np.subtract(self.data_df['acc_max'].values, self.data_df['acc_min'].values)
        chg_dist = np.subtract(self.data_df['chg_acc_max'].values, self.data_df['chg_acc_min'].values)
        self.data_df['distance'] = np.log(np.sqrt(np.add(np.square(dist), np.square(chg_dist))))
        self.data_df['time_per_max_chg_range'] = np.divide(self.data_df['auc'].values, np.subtract(self.data_df['chg_acc_max'].values, self.data_df['chg_acc_min'].values))
        self.data_df['chg_med25'] = np.divide(np.subtract(self.data_df['median_25'].values, self.data_df['acc_mean'].values), self.data_df['acc_mean'].values)
        self.data_df['chg_med70'] = np.divide(np.subtract(self.data_df['median_70'].values, self.data_df['acc_mean'].values), self.data_df['acc_mean'].values)
        self.data_df['max_dist_from_mean'] = np.log(np.abs(np.subtract(self.data_df['acc_max'].values, self.data_df['acc_mean'].values)))
        self.data_df['min_dist_from_mean'] = np.log(np.abs(np.subtract(self.data_df['acc_min'].values, self.data_df['acc_mean'].values)))
        if self.data_type == 'train':
            self.data_df = self.data_df.dropna()
            
    def __transform(self):
        # Using log transformation as did previously
        for col in self.cols_to_log_trainsform:
            self.data_df[col + '_log'] = self.data_df[col].apply(lambda x: np.log(x) if x > 0  else -1 * np.log(-1 * x))
    
    def __minMaxScaler(self):
        from sklearn.preprocessing import MinMaxScaler
        all_cols = list(self.data_df.columns.values)
        for col in all_cols:
            if col not in self.non_numeric_cols and col != 'time_to_failure':
                min_max_scaler = MinMaxScaler()
                min_max_scaler.fit(self.data_df[[col]])
                self.data_df[col] = min_max_scaler.transform(self.data_df[[col]])
        print('Min Max Scaler applied ... ')
            
    def __stdScaler(self):
        all_cols = list(self.data_df.columns.values)
        for col in all_cols:
            if col not in self.non_numeric_cols and col != 'time_to_failure':
                stdScaler = StandardScaler()
                stdScaler.fit(self.data_df[[col]])
                self.data_df[col] = stdScaler.transform(self.data_df[[col]])
        print('Standard Scaler applied ... ')
        
    def __dropLowCorrCols(self):
        # Removing Attributes which are not correlated
        self.data_df = self.data_df.drop(self.columns_to_drop, axis=1)
        print('Dropping columns ', self.columns_to_drop, '...')
        
    def __keepImportantCols(self):
        if self.data_type == 'test':
            self.data_df = self.data_df[self.most_dependent_columns]
        else:
            corr = self.data_df.corr()['time_to_failure']
            self.most_dependent_columns = corr.abs().nlargest(self.cols_to_keep + 1, keep='all').index[0:]
            self.most_dependent_columns = self.most_dependent_columns.tolist()
            self.most_dependent_columns.append('segment_id')
            self.data_df = self.data_df[self.most_dependent_columns]
    
    def getMostImpCols(self):
        cols = list(self.most_dependent_columns)
        cols.remove('time_to_failure')
        return cols
            
    def transform(self):
        self.__clean();
        if self.doTransform:
            self.__transform()
            print('Transformation completed ... ')
        self.data_df = self.data_df.replace([np.inf, -np.inf], np.nan)
        self.data_df = self.data_df.fillna(method='ffill')
        self.__dropLowCorrCols()
        if self.doScale:
            self.__minMaxScaler()
        if self.cols_to_keep > 0 or self.most_dependent_columns:
            self.__keepImportantCols()
        return self.data_df
