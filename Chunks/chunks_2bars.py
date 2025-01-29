import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {
    'nodetypes': ["Identifier","SameAs","Creator","Primary\nE-Edition","Title","Person","DOI"],#,"Volume"],#,"Pagination","Toc-Page"],
    'KG-TOSA':[18513,12780,9467,9312,9291,7962,5936],#,3094],#,1580,138],
    'KG-WISE': [1518,754,1420,866,1440,4140,368],#,461],#,545,45],
    'chunks_percentage':[8.2,5.9,15,9.3,15.5,52,6.2]#,,14.9],#34.5,32.8],

}
chunks_diff=[elem- data['KG-WISE'][idx] for idx,elem in enumerate(data['KG-TOSA'])]
data['chunks_diff']=chunks_diff

# Colors for bars
colors = ['#4c8bf5', '#1aa260']
# Create DataFrame
df = pd.DataFrame(data)
# Set the width of the bars
bar_width = 0.48
# Set the x-axis positions for the groups
r1 = np.arange(len(df['nodetypes']))  # The label locations
r2 = [x + bar_width for x in r1]    # Position for second bar

# Create the subplots, 1 row, 3 columns
fig, ax1 = plt.subplots(1, 1, figsize=(7, 3))  # Adjusted figure size to fit 3 charts in a row

# Add the overall title for the dataset
# fig.suptitle(Dataset_Name, fontsize=14, y=0.95)

# 2. Second chart: Time Taken (Inference Time)
bars1 = ax1.bar(r1, df['KG-TOSA'], color=colors[0], width=bar_width, label='KG-TOSA')
bars2 = ax1.bar(r2, df['KG-WISE'], color=colors[1], width=bar_width, label='KG-WISE')

# Add labels and title for the time chart
# ax1.set_xlabel('# Workers', fontsize=12)
ax1.set_ylabel('Number of Chunks', fontsize=10)
# ax1.set_title('Inference Time', fontsize=14, y=0.98)
ax1.set_ylim(0, 20000)

# Add xticks on the middle of the group bars
ax1.set_xticks([r + bar_width / 2 for r in range(len(df['nodetypes']))])
ax1.set_xticklabels(df['nodetypes'])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticklabels(data['nodetypes'], rotation=20, fontsize=10)
# Add values on top of the bars for time chart
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{round(yval/1000, 1):.1f}K', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{round(yval/1000, 1):.1f}K', ha='center', va='bottom', fontsize=10)


if True:#_targets == 1600:
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10, ncol=2, bbox_to_anchor=(0.5, 1.1))

# Set layout adjustments to fit everything neatly
plt.tight_layout()
plt.savefig(f'dblp_chunks_2bars.pdf', dpi=1200, bbox_inches='tight',format='pdf')
plt.show()
