import matplotlib.pyplot as plt
import numpy as np

# Data
plt.rcParams.update({'font.size': 10})
models = [r'$M$', r'$\widetilde{\mathbb{M}}$']#['KG-TOSA', 'KG-WISE']
test_accuracy = [0.9, 0.91]  # Dummy values for Test Accuracy
kg_tosa_pre_model_inf = [12.17,17.98,20.2]  # Teacher, Student
wise_pre_model_inf = [6.15,	1.8,	19.54]# Train, Prune, Re-train
train_memory_gkd = [16.59, ]  # Dummy values for Train Memory (Non-stacked)
train_memory_gcnp = [4.74]  # Dummy values for Train Memory (Non-stacked)

# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(6, 3))
width = 0.6
fig.suptitle('DBLP Paper-Venue', fontsize=14,y =0.95)

# fig.suptitle('KG-TOSA and KG-WISE Comparison ',y=1.05,fontsize=16)
# Test Accuracy Plot
bars = axs[0].bar(models, test_accuracy, width, color=['#4c8bf5', '#1aa260'])
axs[0].set_ylabel('Accuracy in %', fontsize=10)
axs[0].set_title('A. Inference Accuracy', fontsize=10)
axs[0].set_ylim(0, 1)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
# Add text values on top of bars
for bar in bars:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval , f'{yval:.2f}', ha='center', va='bottom', fontsize=10,)

#F1BB71  #ede658  #4c8bf5   #1aa260
# Training Time Plot (Stacked)
# Calculate total training times
kg_tosa_total_time = np.sum(kg_tosa_pre_model_inf)
wise_total_time = np.sum(wise_pre_model_inf)

# Training Time Plot (Single Bar)
bars_train_time = axs[1].bar(models, [kg_tosa_total_time, wise_total_time], width, color=['#4c8bf5', '#1aa260'])

axs[1].set_ylabel('Time (sec)', fontsize=10)
axs[1].set_title('B. Total Inference Time', fontsize=10)
axs[1].set_ylim((0, 60))
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
# Add text values on top of bars
for bar in bars_train_time:
    yval = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
# Manually add the legend to avoid duplication
# fig.legend([bar1, bar2, bar3],  # Add only the unique bars to the legend
#               ['Pre-Process', 'Model Load', 'Forward-Pass'],  # Provide corresponding unique labels
#               loc='upper left', bbox_to_anchor=(0.28, 1.08), ncol=3)

# axs[1].legend(loc='upper left', bbox_to_anchor=(0.8, 1), ncol=1)

# Add text values on top of stacked bars
# For GKD
# total = 0
# for i,b in enumerate([bar1, bar2,bar3]):
#     for bar in b:
#         total +=bar.get_height()
#         if i == 2: 
#             axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y() + 0, 
#                         f'{total:.2f}', ha='center', va='bottom', fontsize=10,)

# # For GCNP
# total = 0
# total_gcnp_height = np.array(wise_pre_model_inf).sum()
# for i,b in enumerate([bar4, bar5, bar6]):
#     for bar in b:
#         total += bar.get_height()
#         if i==2:
#             axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y() + 0, 
#                         f'{total:.2f}', ha='center', va='bottom', fontsize=10,)

# Train Memory Plot (Non-Stacked)
bars_memory_gkd = axs[2].bar(models[0], train_memory_gkd[0], width, label='Train Memory', color='#4c8bf5')
bars_memory_gcnp = axs[2].bar(models[1], train_memory_gcnp[0], width, label='Train Memory', color='#1aa260')

axs[2].set_ylabel('Memory (GBs)', fontsize=10)
axs[2].set_title('C. Max Memory Usage', fontsize=10)
axs[2].set_ylim((0,20))
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
# axs[2].legend(loc='upper left', bbox_to_anchor=(0.8, 1), ncol=1)

# Add text values for non-stacked bars
for b in [bars_memory_gkd, bars_memory_gcnp]:
    for bar in b:
        yval = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width()/2, yval , f'{yval:.1f}', ha='center', va='bottom', fontsize=10,)

# Formatting
for ax in axs:
    ax.set_xticks(models)
    ax.set_xticklabels(models, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

# Layout adjustment
plt.tight_layout()

# Show the plot
# plt.show()
plt.savefig('DBLP.pdf', dpi=1200, bbox_inches='tight',format='pdf')
plt.show()

