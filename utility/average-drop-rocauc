from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score

def model(df_final, target="Target"):
  model = LGBMClassifier()
  test =  df_final.head(int(len(df_final)*0.4))
  train = df_final[~df_final.isin(test)].dropna()
  model = model.fit(train.drop([target],axis=1),train[target])
  preds = model.predict(test.drop([target],axis=1))
  test =  df_final.head(int(len(df_final)*0.4))
  train = df_final[~df_final.isin(test)].dropna()
  model = model.fit(train.drop([target],axis=1),train[target])
  val = roc_auc_score(test[target],preds); 
  return val
  
dict_org = {}
for col in data_dummies.columns.to_list()[1:]:
  dict_org[col] = model(data_dummies, target=col)
  
dict_syn = {}
for col in samples_dummies.columns.to_list()[1:]:
  dict_syn[col] = model(samples_dummies, target=col)
  
org_auc = pd.DataFrame.from_dict(dict_org, orient="index")
syn_auc = pd.DataFrame.from_dict(dict_syn, orient="index")

combine = pd.concat((org_auc,syn_auc),axis=1)
combine.columns = ["AUC Original", "AUC Synthetic"]

vals = combine.sort_values("AUC Synthetic"); vals.head(8)

#Average Drop in ROCAUC

((vals["AUC Original"]-vals["AUC Synthetic"])/vals["AUC Original"]).mean()
