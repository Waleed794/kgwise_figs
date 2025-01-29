import matplotlib.pyplot as plt
import numpy as np

# Data
models = [r'$M$', r'$\widetilde{\mathbb{M}}$']
test_accuracy = [0.2,0.35]  # Dummy values for Test Accuracy
kg_tosa_pre_model_inf = [17.8]  # Teacher, Student
wise_pre_model_inf = [15.2]# Train, Prune, Re-train
train_memory_gkd = [5.05, ]  # Dummy values for Train Memory (Non-stacked)
train_memory_gcnp = [1.02]  # Dummy values for Train Memory (Non-stacked)

# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(8, 3))
width = 0.6
fig.suptitle('DBLP Author-Affiliation', fontsize=16,y =0.95)

# fig.suptitle('KG-TOSA and KG-WISE Comparison ',y=1.05,fontsize=16)
# Test Accuracy Plot
bars = axs[0].bar(models, test_accuracy, width, color=['#1aa260', '#1aa260'])
axs[0].set_ylabel('Hits', fontsize=12)
axs[0].set_title('A. Inference Accuracy', fontsize=14)
axs[0].set_ylim(0, 1)

# Add text values on top of bars
for bar in bars:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval , f'{yval:.2f}', ha='center', va='bottom', fontsize=10,)

#F1BB71  #ede658  #4c8bf5   #1aa260
# Training Time Plot (Stacked)
bar1 = axs[1].bar(models[0], kg_tosa_pre_model_inf[0], width, label='Train/Teacher', color='#1aa260')
# bar2 = axs[1].bar(models[0], kg_tosa_pre_model_inf[1], width, bottom=kg_tosa_pre_model_inf[0], label='Re-train/Student', color='#ede658')
# bar3 = axs[1].bar(models[0], kg_tosa_pre_model_inf[2], width, bottom=np.array(kg_tosa_pre_model_inf[:2]).sum(), label='Re-train/Student', color='#4c8bf5')

bar4 = axs[1].bar(models[1], wise_pre_model_inf[0], width, label='Train/Teacher', color='#1aa260')
# bar5 = axs[1].bar(models[1], wise_pre_model_inf[1], width, bottom=wise_pre_model_inf[0], label='Prune', color='#ede658')
# bar6 = axs[1].bar(models[1], wise_pre_model_inf[2], width, bottom=np.array(wise_pre_model_inf[:2]).sum(), label='Re-train/Student', color='#4c8bf5')

axs[1].set_ylabel('Time in seconds', fontsize=12)
axs[1].set_title('B. Total Inference Time', fontsize=14)
axs[1].set_ylim((0,60))
# Manually add the legend to avoid duplication
# fig.legend([bar1, ],  # Add only the unique bars to the legend
#               ['Pre-Process', 'Model Load', 'Inference'],  # Provide corresponding unique labels
#               loc='upper left', bbox_to_anchor=(0.28, 1.08), ncol=3)

# axs[1].legend(loc='upper left', bbox_to_anchor=(0.8, 1), ncol=1)

# Add text values on top of stacked bars
# For GKD
total = 0
for i,b in enumerate([bar1, ]):
    for bar in b:
        total +=bar.get_height()
        if i == 0: 
            axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y() + 0, 
                        f'{total:.2f}', ha='center', va='bottom', fontsize=10,)

# For GCNP
total = 0
total_gcnp_height = np.array(wise_pre_model_inf).sum()
for i,b in enumerate([bar4, ]):
    for bar in b:
        total += bar.get_height()
        if i==0:
            axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y() + 0, 
                        f'{total:.2f}', ha='center', va='bottom', fontsize=10,)

# Train Memory Plot (Non-Stacked)
bars_memory_gkd = axs[2].bar(models[0], train_memory_gkd[0], width, label='Train Memory', color='#1aa260')
bars_memory_gcnp = axs[2].bar(models[1], train_memory_gcnp[0], width, label='Train Memory', color='#1aa260')

axs[2].set_ylabel('Gigabytes (GBs)', fontsize=12)
axs[2].set_title('C. Max Memory Usage', fontsize=14)
axs[2].set_ylim((0,20))
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
plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/tosa_v_wise/LP_DBLP.pdf', dpi=1200, bbox_inches='tight',format='pdf')

