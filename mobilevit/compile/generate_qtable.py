import pandas as pd

df = pd.read_csv("tmp/mobilevit_cali_table", sep=" ", on_bad_lines = 'warn', header=None)
df = df.drop(columns=[4])
df = df[5:]
df = df.rename(columns={0:'op_name', 1:'threshold', 2:'min', 3:'max'})
df = df[df['threshold'].astype(float)>10]
df = df.drop(columns=['threshold', 'min', 'max'])
df['quantize_mode'] = "F16"
df.to_csv("tmp/mobilevit_qtable_brute.csv", index=False, sep=' ', header=False)


