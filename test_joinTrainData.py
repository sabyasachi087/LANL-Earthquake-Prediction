import pandas as pd

main_df = pd.read_csv('train_modeled_data_main.csv')
small_df = pd.read_csv('train_modeled_data.csv')

new_df = pd.concat([main_df, small_df])

new_df.to_csv('train_data_model_concat.csv', index=False)
