import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load your data
df = pd.read_csv('/Users/kchiama/Documents/PFC DEM/SRL 2025/csv/combinedCase123.csv')
df_2D = pd.read_csv('/Users/kchiama/Downloads/DEM_dataset.csv')

# Harmonize scarp_type naming
df['scarp_type'] = df['scarp_type'].replace({'Pressure_Ridge': 'Pressure Ridge'})

# Model display names
model_display_names = {
    "Case1": "Cylindrical Model",
    "Case2": "Variable Fault Dip",
    "Case3": "Variable Fault Seed",
}
df['model_display'] = df['model'].map(model_display_names)

# Create dip bins for the last panel (3D)
dip_bins = np.arange(df['dip'].min()//10*10, df['dip'].max()//10*10 + 11, 10)
dip_labels = [f"{int(low)}–{int(high-1)}º" for low, high in zip(dip_bins[:-1], dip_bins[1:])]
df['dip_bin'] = pd.cut(df['dip'], bins=dip_bins, labels=dip_labels, right=False)

# Only show these 3 for field & 3D panels, in this order
scarp_order = ['Simple', 'Monoclinal', 'Pressure Ridge']
df['scarp_type'] = pd.Categorical(df['scarp_type'], categories=scarp_order, ordered=True)

df_2D = df_2D[df_2D['Scarp_Class'].isin(scarp_order)].copy()
df_2D['Scarp_Class'] = pd.Categorical(df_2D['Scarp_Class'], categories=scarp_order, ordered=True)

# Sediment Strength order
strength_order = ['Weak', 'ModerateWeak', 'Moderate', 'Strong', 'CTU', 'Random']
df['strength_label'] = pd.Categorical(df['strength_label'], categories=strength_order, ordered=True)

# ---- ADD: df_2D filtered by Depth==3 for boxplot 1 ----
df_2D_depth3 = df_2D[df_2D["Depth"] == 3].copy()
# (no need to touch categories, already set above as needed)

# Plot setup
fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=False, sharey=False)

boxplot_info = [
    # (dataframe, ycolumn, y label, title, palette, xcolumn, xlim)
    (df_2D,            "Scarp_Class",   'Scarp Class',   "2D DEM Models by Scarp Class",           ['#ed2024','#009ffa', '#f47820'], "DZW", 50),
    (df_2D_depth3,     "Scarp_Class",   'Scarp Class',   "2D DEM with Depth=3 by Scarp Class",     ['#ed2024','#009ffa', '#f47820'], "DZW", 25),
    (df,               "scarp_type",    'Scarp Class',   "3D DEM Models by Scarp Class",           ['#ed2024','#009ffa', '#f47820'], "dzw", 25),
    (df,      "model_display", 'Model Case',    "3D DEM Models by Case",         "Set2",    "dzw", 25),
    (df,      "strength_label",'Sediment Strength', "3D DEM Models by Sediment Strength", "Set2", "dzw", 25),
    (df,      "dip_bin",       'Fault Dip (º bin)', "3D DEM Models by Fault Dip (10º Bins)", "flare", "dzw", 25)
]
panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
offset = 0.5  # horizontal n-value offset past box

for i, (this_df, ycol, ylabel, title, palette, xcol, xlim_val) in enumerate(boxplot_info):
    ax = axes[i]
    # Set plotting order
    if ycol == "strength_label":
        order = strength_order
    elif ycol in ["scarp_type", "Scarp_Class"]:
        order = scarp_order
    else:
        order = sorted(this_df[ycol].dropna().unique())
    # Create boxplot
    sns.boxplot(
        data=this_df,
        x=xcol,
        y=ycol,
        palette=palette,
        order=order,
        ax=ax,
        boxprops=dict(edgecolor="k", linewidth=1),
        whiskerprops=dict(color="k", linewidth=1),
        capprops=dict(color="k", linewidth=1),
        medianprops=dict(color="k", linewidth=1),
        flierprops=dict(
            marker='.',
            markerfacecolor='k',
            markeredgecolor='k',
            markersize=1,
            alpha=0.25
        )
    )
    # Far left panel label in figure margin
    ax.text(-0.20, 1.1, panel_letters[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')
    ax.set_title(title, fontsize=12, loc='left', x=0.0)
    ax.set_xlabel('Deformation Zone Width (m)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(0, xlim_val)
    # Rotate y-ticks for dip bins
    if ycol == "dip_bin":
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right")
    # -- Add sample size n to the right of each box --
    yticklabels = order
    for j, label in enumerate(yticklabels):
        if pd.isna(label):
            continue
        n = this_df[this_df[ycol] == label].shape[0]
        ax.text(xlim_val + offset, j, f"n={n}", va='center', ha='left', fontsize=9, color='k')

plt.tight_layout()
plt.show()
plt.savefig('/Users/kchiama/Documents/PFC DEM/SRL 2025/FDHI-DEM-DZW-ScarpHeights_boxplots.pdf', format='pdf', dpi=300)