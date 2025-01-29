import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'Workers': [1, 2, 4,],# 8],
    'Graph-SAINT': [1240,685.43,678.32],  # Time for Graph-SAINT
    'KG-WISE': [462,273.98,196],           # Time for KG-WISE
    'Graph-SAINT RAM': [106,170,217],  # Memory for Graph-SAINT
    'KG-WISE RAM': [13.24,26.56,50.72],         # Memory for KG-WISE
    # 'Graph-SAINT Accuracy': [0.88,0.83,0.83,0.84], # Accuracy for Graph-SAINT
    # 'KG-WISE Accuracy': [0.9,0.9,0.9,0.9]       # Accuracy for KG-WISE
}


Dataset_Name = "Strong Scalability"#"DBLP {n_targets} target throughput"
# Colors for bars
colors = ['#4c8bf5', '#1aa260']

# Create DataFrame
df = pd.DataFrame(data)

# Set the width of the bars
bar_width = 0.35

# Set the x-axis positions for the groups
r1 = np.arange(len(df['Workers']))  # The label locations
r2 = [x + bar_width for x in r1]    # Position for second bar

# Create the subplots, 1 row, 3 columns
fig, ( ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))  # Adjusted figure size to fit 3 charts in a row

# Add the overall title for the dataset
fig.suptitle(Dataset_Name, fontsize=14, y=0.95)


# 2. Second chart: Time Taken (Inference Time)
bars1 = ax1.bar(r1, df['Graph-SAINT'], color=colors[0], width=bar_width, label='Graph-SAINT')
bars2 = ax1.bar(r2, df['KG-WISE'], color=colors[1], width=bar_width, label='KG-WISE')

# Add labels and title for the time chart
ax1.set_xlabel('# Workers', fontsize=12)
ax1.set_ylabel('Time (sec)', fontsize=12)
ax1.set_title('Inference Time', fontsize=14, y=0.98)
ax1.set_ylim(0, 1450)

# Add xticks on the middle of the group bars
ax1.set_xticks([r + bar_width / 2 for r in range(len(df['Workers']))])
ax1.set_xticklabels(df['Workers'])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# Add values on top of the bars for time chart
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{round(yval, 2):.0f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{round(yval, 2):.0f}', ha='center', va='bottom', fontsize=10)

# 3. Third chart: RAM Usage
bars3 = ax2.bar(r1, df['Graph-SAINT RAM'], color=colors[0], width=bar_width, label='Graph-SAINT RAM')
bars4 = ax2.bar(r2, df['KG-WISE RAM'], color=colors[1], width=bar_width, label='KG-WISE RAM')

# Add labels and title for the memory usage chart
ax2.set_xlabel('# Workers', fontsize=12)
ax2.set_ylabel('Memory (GBs)', fontsize=12)
ax2.set_title('Max Memory Usage', fontsize=14, y=0.98)
ax2.set_ylim(0, 250)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add xticks on the middle of the group bars for the memory chart
ax2.set_xticks([r + bar_width / 2 for r in range(len(df['Workers']))])
ax2.set_xticklabels(df['Workers'])

# Add values on top of the bars for memory usage chart
for bar in bars3:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{round(yval, 2):.0f}', ha='center', va='bottom', fontsize=10)
for bar in bars4:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1,f'{round(yval, 2):.0f}', ha='center', va='bottom', fontsize=10)

# Create a single legend for the entire figure
if False:#_targets == 1600:
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10, ncol=2, bbox_to_anchor=(0.5, 1.1))

# Set layout adjustments to fit everything neatly
plt.tight_layout()

# Show the plot
# plt.show()

plt.savefig(f'/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/TP/Strong_sca.pdf', dpi=1200, bbox_inches='tight',format='pdf')
