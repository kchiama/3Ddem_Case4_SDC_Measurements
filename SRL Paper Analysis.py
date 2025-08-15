import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import statistics as st
from statistics import median
from math import isnan
from itertools import filterfalse

# Load your main dataframe
df_2D = pd.read_csv('/Users/kchiama/Downloads/DEM_dataset.csv')
df = pd.read_csv('/Users/kchiama/Documents/PFC DEM/SRL 2025/csv/combinedCase123.csv')
    #'/Users/kchiama/Documents/PFC DEM/SRL 2025/3D DEM Dataset.csv')
df_Case4 = pd.read_csv('/Users/kchiama/Documents/PFC DEM/SRL 2025/csv/oblique/combinedCase4.csv')

df_1a = df_2D.loc[df_2D['Scarp_Class'].eq('Monoclinal')]               #009ffa blue
df_1b = df_2D.loc[df_2D['Scarp_Class'].eq('Monoclinal Collapse')]      #3f67b1 dark blue
df_2a = df_2D.loc[df_2D['Scarp_Class'].eq('Pressure Ridge')]           #f47820 orange
df_2b = df_2D.loc[df_2D['Scarp_Class'].eq('Pressure Ridge Collapse')]  #af773e brown
df_3a = df_2D.loc[df_2D['Scarp_Class'].eq('Simple')]                   #ed2024 red
df_3b = df_2D.loc[df_2D['Scarp_Class'].eq('Simple Collapse')]          #9f1d20 dark red
df_2DA = pd.concat([df_1a,df_2a,df_3a])

df_SURE = pd.read_csv('/Users/kchiama/Documents/SURE Dataset v2.0/SUE2.0_Slip_Obs/SURE.csv')
df_SURE_Wenchuan = df_SURE.loc[df_SURE['eq_name'].eq('Wenchuan')]
df_SURE_ChiChi = df_SURE.loc[df_SURE['eq_name'].eq('Chi-Chi')]

df_FDHI = pd.read_csv('/Users/kchiama/Files/PFC_DEM/2024_DEM_Paper/FDHI/02_FDHI_FLATFILE_MEASUREMENTS_20220719.csv')
df_FDHI_full = pd.read_csv('/Users/kchiama/Files/PFC_DEM/2024_DEM_Paper/FDHI/02_FDHI_FLATFILE_MEASUREMENTS_20220719.csv')
output_csv = '/Users/kchiama/Documents/PFC DEM/SRL 2025/FDHI_Cleaned_Measurements.csv'

df_FDHI_R = df_FDHI.loc[df_FDHI['style'].eq('Reverse')]
df_FDHI_RO = df_FDHI.loc[df_FDHI['style'].eq('Reverse-Oblique')]
df_FDHI = pd.concat([df_FDHI_R,df_FDHI_RO])
#display(df_FDHI)

#df_FDHI = df_FDHI.loc[df_FDHI['rupture_rank'].eq('Principal')]


df_FDHI_fns = df_FDHI[df_FDHI['fns_central_meters'] > 0]
df_FDHI_fzw = df_FDHI[df_FDHI['fzw_central_meters'] > 0]
df_FDHI_fzw = df_FDHI_fzw[df_FDHI_fzw['fzw_central_meters'] < 50]

# Find Positive Scarp Heights
df_FDHI_vs = df_FDHI[df_FDHI['vs_central_meters'] > 0]
df_FDHI_vsL = df_FDHI[df_FDHI['vs_low_meters'] > 0]
df_FDHI_vsH = df_FDHI[df_FDHI['vs_high_meters'] > 0]
df_FDHI_shC = df_FDHI[df_FDHI['sh_central_meters'] > 0]
df_FDHI_shL = df_FDHI[df_FDHI['sh_low_meters'] > 0]
df_FDHI_shH = df_FDHI[df_FDHI['sh_high_meters'] > 0]
df_FDHI_sh = pd.concat([df_FDHI_vs,df_FDHI_vsL,df_FDHI_vsH,df_FDHI_shC,df_FDHI_shL,df_FDHI_shH])
df_FDHI_sh = df_FDHI_sh[df_FDHI_sh['fzw_central_meters'] > 0]
df_FDHI_sh = df_FDHI_sh[df_FDHI_sh['fzw_central_meters'] < 50]

df_FDHI_sh_flag1 = df_FDHI_sh.loc[df_FDHI_sh['recommended_net_preferred_usage_flag'].eq('Check')]
df_FDHI_sh_flag2 = df_FDHI_sh.loc[df_FDHI_sh['recommended_net_preferred_usage_flag'].eq('Keep')]
df_FDHI_sh = pd.concat([df_FDHI_sh_flag1,df_FDHI_sh_flag2])

df_FDHI_Wenchuan = df_FDHI_sh.loc[df_FDHI_sh['eq_name'].eq('Wenchuan')]
df_FDHI_Kashmir = df_FDHI_sh.loc[df_FDHI_sh['eq_name'].eq('Kashmir')]
df_FDHI_Kashmir = df_FDHI_Kashmir[df_FDHI_Kashmir['vs_central_meters'] > 0]
df_FDHI_Kern = df_FDHI_sh.loc[df_FDHI_sh['eq_name'].eq('Kern')]

df_FDHI_sh.to_csv(output_csv, index=False)
print(f"Data has been saved to {output_csv}")

df_KernNew = pd.read_csv('/Users/kchiama/Documents/PFC DEM/2025 Earthquake Spectra DEM paper/Combine_BuwaldaFDHI_KernSDC.csv')

# 1. Make your mapping (adjust as needed for your dataset):
model_display_names = {
    "Case1": "Cylindrical Model",
    "Case2": "Variable Fault Dip",
    "Case3": "Variable Fault Seed",
}
df['model_display'] = df['model'].map(model_display_names)

oblique_display_names = {
    30: "Oblique 30º",
    45: "Oblique 45º",
    60: "Oblique 60º",
}
df_Case4['oblique_display_names'] = df_Case4['oblique'].map(oblique_display_names)

print(80*'=')

plt.figure(figsize=(8, 8))
#ax = sns.scatterplot(data=df_2D, x="DZW", y="Scarp_Height", hue="Scarp_Class", palette= ['#009ffa','#3f67b1','#f47820','#af773e','#ed2024','#9f1d20'],s=10, alpha = 0.1, linewidth=0.25, label = '2D DEM Data - Scarp Class')
ax = sns.scatterplot(data=df_2DA, x="DZW", y="Scarp_Height", hue="Scarp_Class", palette= ['#8ddcf7','#f7aa77','#f28f94'],s=10, alpha = 0.3, linewidth=0.25, label = '2D DEM Models')
#ax = sns.scatterplot(data=df_1a, x="DZW", y="Scarp_Height", hue="Scarp_Class", palette= ['#8ddcf7'],s=10, alpha = 0.3, linewidth=0.25, label = '2D DEM - Monoclinal')
#ax = sns.scatterplot(data=df_2a, x="DZW", y="Scarp_Height", hue="Scarp_Class", palette= ['#f7aa77'],s=10, alpha = 0.3, linewidth=0.25, label = '2D DEM - Pressure Ridge')
#ax = sns.scatterplot(data=df_3a, x="DZW", y="Scarp_Height", hue="Scarp_Class", palette= ['#f28f94'],s=10, alpha = 0.3, linewidth=0.25, label = '2D DEM - Simple')
ax = sns.scatterplot(data=df, x="dzw", y="adjusted_scarp_height", hue="scarp_type", palette= ['#ed2024','#009ffa','#f47820'],s=20, marker='^', alpha = 0.9, linewidth=0.1,label = '3D DEM Data - Scarp Class')
#ax = sns.scatterplot(data=df_Case4, x="dzw_perpendicular", y="adjusted_scarp_height", hue="scarp_type", palette= ['#009ffa','#ed2024','#f47820'],s=20,  marker='^', alpha = 0.9, edgecolor='black', linewidth=0.25, label = '3D Oblique DEM Data')
ax = sns.scatterplot(data=df_SURE_Wenchuan, x="FNC", y="SH", color='grey', s=100, marker='*', edgecolor='black',linewidth=1, label = 'Wenchuan (SURE)')
ax = sns.scatterplot(data=df_SURE_ChiChi, x="FNC", y="SH", color='teal',s=100, marker='*', edgecolor='black',linewidth=1, label = 'Chi-Chi (SURE)')
ax = sns.scatterplot(data=df_FDHI_Wenchuan, x="fzw_central_meters", y="vs_central_meters", color='grey', marker='*', s=150, edgecolor='white', linewidth=1,label='Wenchuan (FDHI)')
ax = sns.scatterplot(data=df_FDHI_Kashmir, x="fzw_central_meters", y="vs_central_meters", color='purple', marker='*', s=150, edgecolor='white', linewidth=1,label='Kashmir (FDHI)')
ax = sns.scatterplot(data=df_KernNew, x="DZW", y="Vertical", color='white', marker='*', s=100, edgecolor='blue', linewidth=1, label='Kern County \n (Buwalda & St. Amand, 1955)')
#ax = sns.scatterplot(data=df_2D.sort_values("Sediment_Strength", ascending=False), x="DZW", y="Scarp_Height", hue="Sediment_Strength", palette='Set2',s=10, alpha = 0.3, label = '2D DEM Data')
#ax.set_title('2D & 3D DEM Models, FDHI & SURE Datasets', loc='left', fontsize = 12)
legend = ax.legend(fontsize=1)
ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
ax.set_xlabel('Deformation Zone Width (m)', fontsize = 10)  # Set x-axis label 
ax.set_ylabel('Scarp Height (m)', fontsize = 10)  # Set y-axis label
ax.set_ylim(0, 5.5) #Set y axis label because FDHI has NaNs
ax.set_xlim(0, 55)
plt.show()

plt.savefig('/Users/kchiama/Documents/PFC DEM/SRL 2025/Fig-9-FDHI-SURE-DEM-2D-3D-Scatter.pdf', format='pdf', dpi=300) # or pdf