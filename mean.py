import pandas as pd


print("FlowDrive  navmini score ")
df = pd.read_csv('/home/akash/learn/navsim/exp/flow_agent_pretrained_eval/2026.05.19.08.53.30/2026.05.19.09.00.58.csv')
print(df[df['valid']==True]['score'].describe())

print("Diffusion drive navmini score ")
df1 = pd.read_csv('/home/akash/learn/navsim/exp/diffusiondrive_pretrained_navmini/2026.05.19.09.20.11/2026.05.19.09.28.21.csv')
print(df1[df1['valid']==True]['score'].describe())

print("Velcity agent  drive navmini score ")
df2 = pd.read_csv('/home/akash/learn/navsim/exp/constant_velocity_navmini/2026.05.19.09.20.22/2026.05.19.09.27.05.csv')
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





############################ nav mini trained rl 2 epochs ##############################

print("\n" + "="*60)
print("RL TRAINED MODEL (navmini, 2 epochs navtrain)")
print("="*60)

print("\nFlowRL navmini score (2 epochs RL training)")
df_rl = pd.read_csv('/home/akash/learn/navsim/exp/flow_rl_navtrain_eval_mini/2026.05.18.20.15.12/2026.05.18.20.36.43.csv')
print(f"Total rows: {len(df_rl)}, valid: {df_rl['valid'].sum()}, invalid: {(~df_rl['valid']).sum()}")
print(df_rl[df_rl['valid']==True]['score'].describe())
print(f"\nMean PDMS (valid only): {df_rl[df_rl['valid']==True]['score'].mean() * 100:.2f}")

print("\n" + "="*60)
print("UPDATED SUMMARY TABLE (navmini)")
print("="*60)
summary2 = {
    "Agent":        ["DiffusionDrive", "FlowDrive (pretrained)", "FlowDrive RL (2ep)", "ConstVelocity"],
    "navmini PDMS": [df1[df1['valid']==True]['score'].mean() * 100,
                     df [df ['valid']==True]['score'].mean() * 100,
                     df_rl[df_rl['valid']==True]['score'].mean() * 100,
                     df2[df2['valid']==True]['score'].mean() * 100],
}
print(pd.DataFrame(summary2).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


print("\n" + "="*60)
print("RL TRAINED MODEL (navtest, 2 epochs navtrain)")
print("="*60)

print("\nFlowRL navtest score (2 epochs RL training)")
df_rl_navtest = pd.read_csv('/home/akash/learn/navsim/exp/flow_rl_navtest_eval/2026.05.19.07.23.35/2026.05.19.15.34.58.csv')
print(f"Total rows: {len(df_rl_navtest)}, valid: {df_rl_navtest['valid'].sum()}, invalid: {(~df_rl_navtest['valid']).sum()}")
print(df_rl_navtest[df_rl_navtest['valid']==True]['score'].describe())
print(f"\nMean PDMS (valid only): {df_rl_navtest[df_rl_navtest['valid']==True]['score'].mean() * 100:.2f}")

print("\n" + "="*60)
print("FULL SUMMARY TABLE (navmini + navtest)")
print("="*60)
summary3 = {
    "Agent":        ["DiffusionDrive", "FlowDrive (pretrained)", "FlowDrive RL (2ep)", "ConstVelocity"],
    "navmini PDMS": [df1[df1['valid']==True]['score'].mean() * 100,
                     df [df ['valid']==True]['score'].mean() * 100,
                     df_rl[df_rl['valid']==True]['score'].mean() * 100,
                     df2[df2['valid']==True]['score'].mean() * 100],
    "navtest PDMS": [df3[df3['valid']==True]['score'].mean() * 100,
                     df4[df4['valid']==True]['score'].mean() * 100,
                     df_rl_navtest[df_rl_navtest['valid']==True]['score'].mean() * 100,
                     df5[df5['valid']==True]['score'].mean() * 100],
}
print(pd.DataFrame(summary3).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

#### rl with multiplicative noise wta
print("\n" + "="*60)
print("RL v2 (navtest, multiplicative noise / rl likelihood / wta)")
print("="*60)

print("\nFlowRL v2 navtest score")
df_rl2_navtest = pd.read_csv('/home/akash/learn/navsim/exp/flow_rl_navtest_eval/2026.05.20.07.39.05/2026.05.20.16.16.41.csv')
print(f"Total rows: {len(df_rl2_navtest)}, valid: {df_rl2_navtest['valid'].sum()}, invalid: {(~df_rl2_navtest['valid']).sum()}")
print(df_rl2_navtest[df_rl2_navtest['valid']==True]['score'].describe())
print(f"\nMean PDMS (valid only): {df_rl2_navtest[df_rl2_navtest['valid']==True]['score'].mean() * 100:.2f}")

print("\nFlowRL v2 navmini score")
df_rl2_navmini = pd.read_csv('/home/akash/learn/navsim/exp/flow_rl_navtest_eval/2026.05.20.19.14.16/2026.05.20.19.39.06.csv')
print(f"Total rows: {len(df_rl2_navmini)}, valid: {df_rl2_navmini['valid'].sum()}, invalid: {(~df_rl2_navmini['valid']).sum()}")
print(df_rl2_navmini[df_rl2_navmini['valid']==True]['score'].describe())
print(f"\nMean PDMS (valid only): {df_rl2_navmini[df_rl2_navmini['valid']==True]['score'].mean() * 100:.2f}")

print("\n" + "="*60)
print("FULL SUMMARY TABLE (navmini + navtest, all variants)")
print("="*60)
summary4 = {
    "Agent":        ["DiffusionDrive", "FlowDrive (pretrained)", "FlowRL 2ep", "FlowRL v2", "ConstVelocity"],
    "navmini PDMS": [df1[df1['valid']==True]['score'].mean() * 100,
                     df [df ['valid']==True]['score'].mean() * 100,
                     df_rl[df_rl['valid']==True]['score'].mean() * 100,
                     df_rl2_navmini[df_rl2_navmini['valid']==True]['score'].mean() * 100,
                     df2[df2['valid']==True]['score'].mean() * 100],
    "navtest PDMS": [df3[df3['valid']==True]['score'].mean() * 100,
                     df4[df4['valid']==True]['score'].mean() * 100,
                     df_rl_navtest[df_rl_navtest['valid']==True]['score'].mean() * 100,
                     df_rl2_navtest[df_rl2_navtest['valid']==True]['score'].mean() * 100,
                     df5[df5['valid']==True]['score'].mean() * 100],
}
print(pd.DataFrame(summary4).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

