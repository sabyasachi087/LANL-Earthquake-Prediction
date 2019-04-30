import accoustic_sampler as acs
import data_formatter as dtFrm
import pandas as pd


class TrendLine:
    
    def __init__(self, data_type='train', models=['lgbm_regressor.model', 'linear_regressor.model', 'random_forest_regressor.model', 'support_vector_regression.model'\
               , 'logistic_regression.model', 'mlp_regression.model']):
        self.type = data_type
        self.models = models
        self.__eval_data()
        self.__trends()
    
    def __eval_data(self):
        if self.type == 'train':
            sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
            # sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/test_data', data_type='test')
            sampler.fit()
            self.data_df = sampler.get()
            self.y_train = self.data_df['time_to_failure']
        elif self.type == 'test':
            sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/test_data', data_type='test')
            sampler.fit()
            self.data_df = sampler.get()
        self.y_seg = self.data_df['segment_id']
    
    def __trends(self):
        from model_holder import  load_model
        self.result_map = {}
        for model_file in self.models:
            print('Running regression ', model_file, '....')
            mh = load_model(model_file)
            model, most_dependent_columns = mh.get()
            formatter = dtFrm.LANLDataFormatter(data_df=self.data_df, data_type='test', doTransform=True, doScale=True, \
                                            most_dependent_columns=most_dependent_columns)
            train_df = formatter.transform()
            train_df = train_df.drop(['segment_id'], axis=1)
            y_train_pred = model.predict(train_df)
            self.result_map[model_file] = y_train_pred
            
    def get(self):
        self.result_map['segment_id'] = self.y_seg.values
        if self.type == 'train':
            self.result_map['time_to_failure'] = self.y_train.values
            return pd.DataFrame(self.result_map)
        elif self.type == 'test':
            return pd.DataFrame(self.result_map)
