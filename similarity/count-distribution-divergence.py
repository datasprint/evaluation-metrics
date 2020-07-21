#!pip install heatmapz 

from heatmap import heatmap, corrplot

def plot_dist(data):
  data['cnt'] = np.ones(len(data))
  # bin_labels = ['Low (0-100)', 'Medium (100-150)', 'High (150+)']
  # data['horsepower-group'] = pd.cut(data['horsepower'], [0, 100, 150, data['horsepower'].max()], labels=bin_labels)
  # g = data.groupby(['horsepower-group', 'drive-wheels']).count()[['cnt']].replace(np.nan, 0).reset_index()
  # display(g)

  data = data[['geography_London', "employment_status_Unemployed", "have_a_mortgage_Yes", "have_other_borrowing_Yes","cnt"]]
  cols = ["In London", "Unemployed", "Have Mortgage","Have Other Borrowing"]
  data.columns = cols + ["cnt"]

  b = data.copy()

  g = data.groupby(["In London", "Unemployed", "Have Mortgage","Have Other Borrowing"]).count()[['cnt']].replace(np.nan, 0).reset_index()

  alt_cols = ["Not London","Employed","No Mortgage","No Other Borrowing"]
  for col, altcol in zip(cols, alt_cols):
    g[col] = g[col].apply(lambda x: col if x==1 else altcol)

  g["attributes"] = g[cols[1:]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

  g = g[[cols[0]]+["attributes","cnt"]]

  plt.figure(figsize=(5, 5))

  heatmap(
      x=g['In London'], # Column to use as horizontal dimension 
      y=g['attributes'], # Column to use as vertical dimension
      
      size_scale=700, # Change this to see how it affects the plot
      size=g['cnt'], # Values to map to size, here we use number of items in each bucket

      color=g['cnt'], # Values to map to color, here we use number of items in each bucket
      palette=sns.cubehelix_palette(128)[::-1] # We'll use black->red palette
  )
  return b, g
  
  
org_g, org_at =plot_dist(data_dummies)
syn_g, syn_at = plot_dist(samples_dummies)


org_gb = org_g.groupby("In London").sum().iloc[:,:-1]
syn_gb = syn_g.groupby("In London").sum().iloc[:,:-1]


# Weighted Percentage Divergence Separated

(((syn_gb-org_gb).abs()/org_gb)*(org_gb/org_gb.max().max())).sum().sum()
  
  
# Weighted Percentage Divergence Combined
  
(((syn_at["cnt"]-org_at["cnt"]).abs()/org_at["cnt"])*(org_at["cnt"]/org_at["cnt"].sum())).sum()
  

