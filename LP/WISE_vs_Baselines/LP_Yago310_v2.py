
import matplotlib.pyplot as plt

save = True

# Data with updated category names and colors
categories = ['DQ', 'MorsE',' KG\nWISE']
total_time = [203.5,10,1.15 ]
hits_at_10 = [0.01,0.02,0.25,]
max_ram_usage = [7.27,0.59,0.53 ]
colors = ['#4c8bf5','#4c8bf4', '#1aa260'] 
bar_width = 0.8
Dataset_Name = 'YAGO3-10 Airport-ConnectsTo'

fig, axs = plt.subplots(1, 3, figsize=(6, 3.0))
fig.suptitle(Dataset_Name, fontsize=14,y=0.9)
# Plot for Total TIME
bars = axs[1].bar(categories, total_time, color=colors, width=bar_width)
axs[1].set_title('B. Inference Time')
axs[1].set_ylabel('Time (sec)')
axs[1].margins(x=0.08)
axs[1].set_ylim(0,250)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
for bar in bars:
    yval = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, yval + 0.0, round(yval, 2), ha='center', va='bottom')
# axs[1].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Plot for Hits@10
bars = axs[0].bar(categories, hits_at_10, color=colors, width=bar_width)
axs[0].set_ylim(0, 1)
axs[0].set_title('A. Inference\nHits@10')
axs[0].set_ylabel('Hits')
axs[0].margins(x=0.08)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
for bar in bars:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')
# axs[0].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Plot for Max RAM Usage
bars = axs[2].bar(categories, max_ram_usage, color=colors, width=bar_width)
axs[2].set_title('C. Max Memory\nUsage')
axs[2].set_ylabel('Memory (GBs)')
axs[2].margins(x=0.08)
axs[2].set_ylim(0,50)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
for bar in bars:
    yval = bar.get_height()
    axs[2].text(bar.get_x() + bar.get_width()/2, yval + 0.00, round(yval, 2), ha='center', va='bottom')
# axs[2].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)], labels=categories, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Adjust layout
plt.tight_layout()
# plt.show()
if save:
    plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/LP_YAGO310_INF.pdf', dpi=1200, bbox_inches='tight',format='pdf')
    # plt.savefig('LP_YAGO310_INF.pdf', dpi=1200,bbox_inches='tight', format='pdf')

""" FOR LINE GRAPH """

if True:
    DQ_time = [49,68.79,106.6,203.5,341.8]
    morse_time = [9.98,10,13.28,18,27.51]#<--FG #[1.2,1.56,2.7,4.49,8.38]
    kg_wise_time = [0.85,0.86,1.3,1.66,1.96]
    
    DQ_RAM = [7.27,7.27,7.2,7.27,7.35]
    morse_RAM= [0.58,0.59,0.65,0.72,0.9]#[0.76,0.76,0.79,0.81,0.81] #DQ
    kg_wise_RAM = [0.52,0.53,0.56,0.57,0.58]
    num_targets = ["100", "200","400","800","1600"]  
    
    # Plotting
    # fig, ax = plt.subplots(figsize=(5, 3))
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2))
    
    # Plot lines for each category
    ax[0].plot(num_targets, morse_time, marker='o', color=colors[0], label=categories[0])
    ax[0].plot(num_targets, DQ_time, marker='o', color=colors[0], label=categories[0])
    ax[0].plot(num_targets, kg_wise_time, marker='o', color=colors[2], label=categories[2])
    
    # Adding titles and labels
    ax[0].set_xlabel('# Target Nodes Queried', fontsize=10)
    ax[0].set_ylabel('Time (sec)', fontsize=10)
    ax[0].set_xticks(num_targets)
    ax[0].set_xticklabels(num_targets)
    ax[0].set_ylim((0,350))
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    # Adding legend
    # ax[0].legend()
    
    # Adding value labels on the points
    # for i, v in enumerate(morse_time):
    #     ax.text(num_targets[i], v + 0.3, str(v), ha='center', va='bottom')
    # for i, v in enumerate(morse_time,):
    #     ax[0].text(num_targets[i], v + 0.0, str(v), ha='center', va='bottom') # 0.3
    # for i, v in enumerate(kg_wise_time,):
    #     ax[0].text(num_targets[i], v + 0.0, str(round(v,1)), ha='center', va='bottom') # 0.3
    

    # num_targets = [100, 200,400,800,1600]  # Example target counts, replace with actual values if needed
    
    # Plotting
    # fig, ax = plt.subplots(figsize=(5, 3))
    
    # Plot lines for each category
    ax[1].plot(num_targets, morse_RAM, marker='o', color=colors[0], label=categories[0])
    ax[1].plot(num_targets, DQ_RAM, marker='o', color=colors[0], label=categories[0])
    ax[1].plot(num_targets, kg_wise_RAM, marker='o', color=colors[2], label=categories[2])
    
    # Adding titles and labels
    # ax[1].set_title('RAM consumed by Each Method Against Number of Targets', fontsize=14)
    ax[1].set_xlabel('# Target Nodes Queried', fontsize=10)
    ax[1].set_ylabel('Memory (GBs)', fontsize=10)
    ax[1].set_xticks(num_targets)
    ax[1].set_xticklabels(num_targets)
    ax[1].set_ylim((0,8))
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    # for i, v in enumerate(morse_RAM,):
    #     ax[1].text(num_targets[i], v + 0.0, str(v), ha='center', va='bottom')
    # for i, v in enumerate(kg_wise_RAM,):
        # ax[1].text(num_targets[i], v - 0.25, str(v), ha='center', va='bottom')
    fig.suptitle('YAGO ConnectedTo', fontsize=14,y=0.9)
    if save:
        plt.tight_layout()
        plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/LP_YAGO310_INF_TARGETS.pdf', dpi=1200, bbox_inches='tight',format='pdf')
        # plt.savefig('LP_YAGO310_INF_TARGETS.pdf', dpi=1200,bbox_inches='tight', format='pdf')
plt.show()
    