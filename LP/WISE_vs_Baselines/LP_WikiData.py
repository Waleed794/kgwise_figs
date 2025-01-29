
import matplotlib.pyplot as plt

save = True

# Data with updated category names and colors
categories = ['DQ','MorsE', 'KG\nWISE']
total_time = [0,453.7, 2.62]
hits_at_10 = [0,0.53, 0.38]
max_ram_usage = [0,3.5, 0.67]
colors = ['#4c8bf5','#4c8bf4', '#1aa260'] 
bar_width = 0.8
Dataset_Name = 'WikiKG Person-Occupation'

# fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig, axs = plt.subplots(1, 3, figsize=(6, 3))

fig.suptitle(Dataset_Name, fontsize=14,y=0.9)
# Plot for Total TIME
bars = axs[1].bar(categories, total_time, color=colors, width=bar_width)
axs[1].set_title('B. Inference Time')
axs[1].set_ylabel('Time (sec)')
axs[1].margins(x=0.08)
axs[1].set_ylim(0,500)
axs[1].text(bars[0].get_x() + bars[0].get_width()/2, 0.0, 'T-OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
# axs[1].text(bars[1].get_x() + bars[1].get_width()/2, 0.0, 'OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
for bar in bars[1:]:
    yval = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
# axs[1].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Plot for Hits@10
bars = axs[0].bar(categories, hits_at_10, color=colors, width=bar_width)
axs[0].set_ylim(0, 1)
axs[0].set_title('A. Inference\nHits@10')
axs[0].set_ylabel('Hits')
axs[0].margins(x=0.08)
axs[0].text(bars[0].get_x() + bars[0].get_width()/2, 0.0, 'T-OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
# axs[0].text(bars[1].get_x() + bars[1].get_width()/2, 0.0, 'OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
for bar in bars[1:]:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')
# axs[0].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Plot for Max RAM Usage
bars = axs[2].bar(categories, max_ram_usage, color=colors, width=bar_width)
axs[2].set_title('C. Max Memory\nUsage')
axs[2].set_ylabel('Memory (GBs)')
axs[2].margins(x=0.08)
axs[2].set_ylim(0,5)
axs[2].text(bars[0].get_x() + bars[0].get_width()/2, 0.0, 'T-OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
# axs[2].text(bars[1].get_x() + bars[1].get_width()/2, 0.0, 'OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
for bar in bars[1:]:
    yval = bar.get_height()
    axs[2].text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
# axs[2].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Adjust layout
plt.tight_layout()
# plt.show()
if save:
    plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/LP_WikiKG_INF.pdf', dpi=1200, bbox_inches='tight',format='pdf')
    # plt.savefig('LP_WikiKG_INF.pdf', dpi=1200, bbox_inches='tight',format='pdf')

""" FOR LINE GRAPH """

if True:
    morse_time = [430,435.37,475.87,565]
    kg_wise_time = [1.36,2.51,3.92,9.2,]
    
    morse_RAM= [3.5,3.5,3.5,5.7] 
    kg_wise_RAM = [0.63,0.67,0.67,0.7]
    num_targets = ["100", "200","400","800",]  # Example target counts, replace with actual values if needed
    
    # Plotting
    # fig, ax = plt.subplots(figsize=(5, 3))
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2))
    
    # Plot lines for each category
    ax[0].plot(num_targets, morse_time, marker='o', color=colors[0], label=categories[0])
    ax[0].plot(num_targets, kg_wise_time, marker='o', color=colors[2], label=categories[2])
    
    # Adding titles and labels
    ax[0].set_xlabel('# Target Nodes Queried', fontsize=10)
    ax[0].set_ylabel('Time (sec)', fontsize=10)
    ax[0].set_xticks(num_targets)
    ax[0].set_xticklabels(num_targets)
    ax[0].set_ylim((0,600))
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    for i, v in enumerate(kg_wise_time):
        ax[0].text(num_targets[i], v + 0.0, str(round(v,1)), ha='center', va='bottom') # 0.3

    for i, v in enumerate(morse_time):
        ax[0].text(num_targets[i], v + 0.0, str(round(v,1)), ha='center', va='bottom') # 0.3    

    # Plot lines for each category
    ax[1].plot(num_targets, morse_RAM, marker='o', color=colors[0], label=categories[0])
    ax[1].plot(num_targets, kg_wise_RAM, marker='o', color=colors[2], label=categories[2])
    
    # Adding titles and labels
    # ax[1].set_title('RAM consumed by Each Method Against Number of Targets', fontsize=14)
    ax[1].set_xlabel('# Target Nodes Queried', fontsize=10)
    ax[1].set_ylabel('Memory (GBs)', fontsize=10)
    ax[1].set_xticks(num_targets)
    ax[1].set_xticklabels(num_targets)
    ax[1].set_ylim((0,6))
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    for i, v in enumerate(kg_wise_RAM):
        ax[1].text(num_targets[i], v + 0.00, str(v), ha='center', va='bottom')

    for i, v in enumerate(morse_RAM):
        ax[1].text(num_targets[i], v + 0.00, str(v), ha='center', va='bottom')
        
    fig.suptitle('WikiKG2 Occupation', fontsize=14,y=0.9)
    plt.tight_layout()
    if save:
        plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/LP_WikiKG_INF_TARGETS.pdf', dpi=1200, bbox_inches='tight',format='pdf')
        # plt.savefig('LP_WikiKG_INF_TARGETS.pdf', dpi=1200,bbox_inches='tight', format='pdf')
plt.show()

