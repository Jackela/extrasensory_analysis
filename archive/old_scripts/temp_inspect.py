import pandas as pd

df = pd.read_csv("mvp_dataset.csv")
print([col for col in df.columns if "session" in col.lower()])
