import pandas as pd

def js_to_python(str):
    str = str.replace("true", "True")
    str = str.replace("false", "False")
    str = str.replace("null", "None")
    return str

# bbox or segm
is_bbox = True

for file_idx in range(1, 6):
  log_file = open(f"./logs/221129085051332867771_{file_idx}.log", 'r')

  series = []
  accuracy_series = []

  epoch_idx = 0
  lines = log_file.readlines()

  for line_idx in range(len(lines)):
    line = js_to_python(lines[line_idx])

    if "Accumulating evaluation results..." in line:
      if is_bbox:
        kind = "bbox"
      else:
        kind = "segm"

      if not is_bbox:
        is_bbox = True
      else:
        epoch_idx += 1
        is_bbox = not is_bbox

    if " Average " in line:
      if "Average Precision" in line:
        criteria = "Average Precision"
      else:
        criteria = "Average Recall"

      accuracy_series.append({
        "epoch": epoch_idx,
        "IoU": line.split("IoU=")[1].split("|")[0].strip(),
        "area": line.split("area=")[1].split("|")[0].strip(),
        "maxDets": line.split("maxDets=")[1].split("]")[0].strip(),
        "kind": kind,
        criteria: line.split("=")[-1].strip(),
      })

    if line.startswith(":::MLLOG "):
      mllog_obj = eval(line.split(":::MLLOG ")[1])

      next_line = lines[line_idx + 1]

      if "maskrcnn_benchmark.trainer INFO: eta:" in next_line:
        datetime, info_list = next_line.split("maskrcnn_benchmark.trainer INFO:")

        for info in info_list.lstrip().split('  '):
          k, v = info.split(": ")
          mllog_obj[k] = v

      for key in mllog_obj['metadata'].keys():
        mllog_obj[f'metadata_{key}'] = mllog_obj['metadata'][key]

      del mllog_obj['metadata']

      if isinstance(mllog_obj['value'], dict):
        for key in mllog_obj['value'].keys():
          mllog_obj[f'value_{key}'] = mllog_obj['value'][key]
        del mllog_obj['value']

      series.append(mllog_obj)

  df = pd.DataFrame(series)
  df.to_csv(f"./preprocess/log_all_{file_idx}.csv")

  accuracy_df = pd.DataFrame(accuracy_series)
  accuracy_df.to_csv(f"./preprocess/accuracy_all_{file_idx}.csv")
