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

print("\n" + "="*60)
print("NAVTEST RESULTS")
print("="*60)

print("\nDiffusionDrive navtest score (corrected two_frame_extended_comfort)")
df3 = pd.read_csv('/home/akash/learn/navsim/exp/diffusiondrive_navtest_eval/2026.05.12.09.45.49/2026.05.12.12.25.10.csv')
print(df3[df3['valid']==True]['score'].describe())

print("\nFlowDrive navtest score (corrected two_frame_extended_comfort)")
df4 = pd.read_csv('/home/akash/learn/navsim/exp/flow_navtest_eval/2026.05.12.09.45.46/2026.05.12.12.21.03.csv')
print(df4[df4['valid']==True]['score'].describe())

print("\nConstant Velocity navtest score (corrected two_frame_extended_comfort)")
df5 = pd.read_csv('/home/akash/learn/navsim/exp/constvel_navtest_eval/2026.05.12.09.45.44/2026.05.12.11.43.23.csv')
print(df5[df5['valid']==True]['score'].describe())

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
summary = {
    "Agent":        ["DiffusionDrive", "FlowDrive",     "ConstVelocity"],
    "navmini PDMS": [df1[df1['valid']==True]['score'].mean() * 100,
                     df [df ['valid']==True]['score'].mean() * 100,
                     df2[df2['valid']==True]['score'].mean() * 100],
    "navtest PDMS": [df3[df3['valid']==True]['score'].mean() * 100,
                     df4[df4['valid']==True]['score'].mean() * 100,
                     df5[df5['valid']==True]['score'].mean() * 100],
}
print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.2f}"))