# data_dummies = original data
# samples_data = synthetic data

# Modelling Score
ms = (data_dummies.corr()-samples_dummies.corr()).abs().sum().sum()

# Permutation Score
ps = (data_dummies.T.reset_index(drop=True).T.corr()-samples_dummies.T.sample(frac=1,random_state=1).reset_index(drop=True).T.corr()).abs().sum().sum()

# Benchmark Improvement Percentage 

bip = (ps-ms)/ps
