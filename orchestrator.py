import accoustic_sampler as acs

sampler = acs.AccousticSampler('D:/PYTHON_WORKSPACES/Kaggles/EarthquakePrediction/LANL_Earthquake/data/train_data_new')
sampler.fit()

data_df = sampler.get()

print(data_df.corr())