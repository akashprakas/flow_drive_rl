import pandas as pd


print("FlowDrive  navmini score ")
df = pd.read_csv('/home/akash/learn/navsim/exp/flow_agent_pretrained_eval/2026.05.09.19.56.37/2026.05.09.20.03.49.csv')
print(df[df['valid']==True]['score'].describe())

print("Diffusion drive navmini score ")
df1 = pd.read_csv('/home/akash/learn/navsim/exp/diffusiondrive_pretrained_navmini/2026.05.09.20.40.07/2026.05.09.20.47.47.csv')
print(df1[df1['valid']==True]['score'].describe())



print("Velcity agent  drive navmini score ")
df2 = pd.read_csv('/home/akash/learn/navsim/exp/constant_velocity_navmini/2026.05.09.20.52.23/2026.05.09.20.58.44.csv')
print(df2[df2['valid']==True]['score'].describe())

