import pandas as pd
import seaborn as sns

for file_idx in range(1, 6):
  accuracy_df = pd.read_csv(f'./preprocess/accuracy_all_{file_idx}.csv')
  log_df = pd.read_csv(f'./preprocess/log_all_{file_idx}.csv')

  data1 = log_df.loc[:, ["iter", "loss"]]
  data1.dropna(inplace=True)
  data1.loc[:, "loss"] = data1.loc[:, "loss"].apply(lambda loss:
    float(loss.split("(")[1].split(")")[0])
  )

  plot1 = sns.lineplot(x=data1["iter"], y=data1["loss"])
  plot1.get_figure().savefig(f"./figures/loss_{file_idx}.png")
  plot1.get_figure().clf()

  data2 = accuracy_df.loc[:, ["epoch", "Average Precision", "kind"]]
  data2.dropna(how="any", inplace=True)
  data2.drop('kind', axis=1, inplace=True)
  groups = data2.groupby('epoch')

  dic = []
  for key, group in groups:
    print(len(group))
    dic.append({
      "epoch": key,
      "Mean Average Precision": group["Average Precision"].mean()
    })

  df = pd.DataFrame(dic)
  # print(df)

  plot2 = sns.lineplot(x=df["epoch"], y=df["Mean Average Precision"])
  plot2.get_figure().savefig(f"./figures/ap_{file_idx}.png")
  plot2.get_figure().clf()