
import matplotlib.pyplot as plt

save = True

# Data with updated category names and colors
categories = ['DQ','MorsE', 'KG\nWISE']#['DQ', 'KG\nWISE']
categories_legend = ['DQ','MorsE', 'KG-WISE']
total_time = [0,452,41.27]
hits_at_10 = [0,0.13,0.35,]
max_ram_usage = [0,33.35,1.82]


colors = ['#4c8bf5','#4c8bf4', '#1aa260'] 
bar_width = 0.8
Dataset_Name = 'DBLP Author-AffiliatedWith'



# fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig, axs = plt.subplots(1, 3, figsize=(6., 3.5))

fig.suptitle(Dataset_Name, fontsize=14,y=0.95)
# Plot for Total TIME
bars = axs[1].bar(categories, total_time, color=colors, width=bar_width)
axs[1].set_title('B. Inference Time')
axs[1].set_ylabel('Time (sec)')
axs[1].margins(x=0.08)
axs[1].set_ylim(0,500)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
""" FOR OUT OF MEMORY"""
axs[1].text(bars[0].get_x() + bars[0].get_width()/2, 0.0, 'T-OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
# axs[1].text(bars[1].get_x() + bars[1].get_width()/2, 0.0, 'OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
for bar in bars[1:]:
    yval = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
fig.legend(bars, categories_legend, loc='lower center', bbox_to_anchor=(0.5, .95), ncol=3)

# axs[1].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='lower center', bbox_to_anchor=(0.5, 1.2), ncol=2)
# plt.legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='lower center', bbox_to_anchor=(0.5, 1.2), ncol=2)
# Plot for Hits@10
bars = axs[0].bar(categories, hits_at_10, color=colors, width=bar_width)
axs[0].set_ylim(0, 1)
axs[0].set_title('A. Inference\nHits@10')
axs[0].set_ylabel('Hits')
axs[0].margins(x=0.08)
axs[0].text(bars[0].get_x() + bars[0].get_width()/2, 0.01, 'T-OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
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
axs[2].set_ylim(0,40)
axs[2].text(bars[0].get_x() + bars[0].get_width()/2, 0.05, 'T-OOM', ha='center',color='red', fontsize=12,fontweight='bold',rotation=90, va='bottom')
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
    plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/LP_dblp_INF.pdf', dpi=1200, bbox_inches='tight',format='pdf')
    # plt.savefig('LP_dblp_INF.pdf', dpi=1200, bbox_inches='tight',format='pdf')
plt.show()

""" FOR LINE GRAPH """
if False:
    plt.rcParams.update({'font.size': 10})
    kg_tosa_time = [ 74.71,147.87,300,581,1203] #DQ
    kg_wise_time = [8.09, 15.22,27.51,51.32,101.9]
    kg_tosa_RAM= [3.92,3.97,4.01,4.05,4.2] #DQ
    kg_wise_RAM = [1.01,1.02,1.04,1.06,1.09]
    num_targets = ["100", "200","400","800","1600"]  # Example target counts, replace with actual values if needed
    
    # Plotting
    # fig, ax = plt.subplots(figsize=(5, 3))
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2)) # 10,3
    
    # Plot lines for each category
    # ax[0].plot(num_targets, kg_tosa_time, marker='o', color=colors[0], label=categories[0])
    ax[0].plot(num_targets, kg_wise_time, marker='o', color=colors[2], label=categories[2])
    
    # Adding titles and labels
    # ax[0].set_title('', fontsize=14)
    ax[0].set_xlabel('# Target Nodes Queried', fontsize=10)
    ax[0].set_ylabel('Time (sec)', fontsize=10)
    ax[0].set_xticks(num_targets)
    ax[0].set_xticklabels(num_targets)
    ax[0].set_ylim((0.5,120))
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    for i, v in enumerate(kg_wise_time):
        ax[0].text(num_targets[i], v + 0.3, str(round(v,1)), ha='center', va='bottom')
    
    
    
    # Plot lines for each category
    # ax[1].plot(num_targets, kg_tosa_RAM, marker='o', color=colors[0], label=categories[0])
    ax[1].plot(num_targets, kg_wise_RAM, marker='o', color=colors[2], label=categories[2])
    
    # Adding titles and labels
    # ax[1].set_title('RAM consumed by Each Method Against Number of Targets', fontsize=14)
    ax[1].set_xlabel('# Target Nodes Queried', fontsize=10)
    ax[1].set_ylabel('Memory (GBs)', fontsize=10)
    ax[1].set_xticks(num_targets)
    ax[1].set_xticklabels(num_targets)
    ax[1].set_ylim((0,2))
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    for i, v in enumerate(kg_wise_RAM):
        ax[1].text(num_targets[i], v + 0., str(v), ha='center', va='bottom')
 
    handles, labels = [], []

    h, l = ax[0].get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)    
# Create a single legend outside the subplots
    fig.legend(handles, labels, loc='upper center',ncols=1,bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    fig.suptitle('DBLP Paper-Venue', fontsize=14,y=0.9)
    
    if save:
        plt.tight_layout()
        plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/LP_dblp_INF_TARGETS.pdf', dpi=1200, bbox_inches='tight',format='pdf')
        # plt.savefig('LP_dblp_INF_TARGETS.pdf', dpi=1200,bbox_inches='tight', format='pdf')
plt.show()
    

