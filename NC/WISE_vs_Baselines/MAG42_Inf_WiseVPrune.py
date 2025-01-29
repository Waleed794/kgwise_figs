import matplotlib.pyplot as plt
import numpy as np

save = True
# Data for the bars
labels = ['Graph\nSAINT', ' GCNP', 'DQ', 'GKD', 'KG\nWISE']  # 'Default\nTOSA' [3]

# Inference_TIME
inference_time_Load_Default = [27.21]
inference_time_Model_Default = [67.3]
inference_time_Inf_Default = [47.82]

inference_time_Load_KG_WISE = [13.35]
inference_time_Model_KG_WISE = [4.18]
inference_time_Inf_KG_WISE = [47.4]

inference_time_Load_Prune = [0]
inference_time_Model_Prune = [0]
inference_time_Inf_Prune = [0]

inference_time_Load_GKD = [0]

# inference_time_Load_Default_TOSA = [25.91 ]
# inference_time_Model_Default_TOSA = [19.02]
# inference_time_Inf_Default_TOSA = [37.42]

inference_time_Load_DQ = [25.77]
inference_time_Model_DQ = [72.33]
inference_time_Inf_DQ = [442.56]

# ACCURACY
accuracy_Default = [0.7]
accuracy_KG_WISE = [0.72]
accuracy_Prune_train = [0]
# accuracy_Default_TOSA = [0.79]
accuracy_DQ = [0.35]
accuracy_GKD = [0]

# MAX RAM USAGE
ram_Default = [106.19]
ram_KG_WISE = [13.38]
ram_Prune_train = [0]
# ram_Default_TOSA = [40.68]
ram_DQ = [161.54]
ram_GKD = [0]

x = np.arange(len(labels))

# Width of the bars
width = 0.65

# Create a new figure
# fig = plt.figure(figsize=(14, 7))
fig = plt.figure(figsize=(11, 6))
fig.suptitle('MAG Paper-Venue', fontsize=16, y=0.96)

# ACCURACY subplot
ax2 = fig.add_subplot(231)
bars2_1 = ax2.bar(x[0], accuracy_Default, width, color='#1aa260', label='Default')  # 4c8bf5
bars2_2 = ax2.bar(x[1], accuracy_Prune_train, width, color='#1aa260', label='GCNP')  # F1BB71
bars2_5 = ax2.bar(x[2], accuracy_DQ, width, color='#1aa260', label='DQ')
bars2_3 = ax2.bar(x[3], accuracy_GKD, width, color='#1aa260', label='GKD')  # CBF7DD
bars2_4 = ax2.bar(x[4], accuracy_KG_WISE, width, color='#1aa260', label='KG-WISE')

ax2.set_ylabel('Accuracy in %', fontsize=13)
ax2.set_title('A. Inference Accuracy', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
# ax2.tick_params(axis='y', labelsize=13)
ax2.set_ylim(0, 1)
ax2.margins(x=0.08)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for bars in [bars2_1, bars2_5, bars2_4]:  # bars2_3 (KG-TOSA)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0, yval, ha='center', va='bottom')
# Adding red "X" annotation for 'GCNP' bars
for bars in [bars2_2, bars2_3]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), 'T-OOM', ha='center', va='bottom', color='red',
                 fontsize=12, fontweight='bold', rotation=90)

# fig.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/MAG_INF-ACC.pdf', dpi=1200, bbox_inches='tight',format='pdf')

# inference_time subplot
ax1 = fig.add_subplot(232)

bars1_1a = ax1.bar(x[0], inference_time_Load_Default, width, color='#F1BB71', label='Load', edgecolor='black')
bars1_1b = ax1.bar(x[0], inference_time_Model_Default, width, bottom=inference_time_Load_Default, color='#ede658',
                   label='Model')
bars1_1c = ax1.bar(x[0], inference_time_Inf_Default, width,
                   bottom=np.add(inference_time_Load_Default, inference_time_Model_Default), color='#4c8bf5',
                   label='Inference')

bars1_2a = ax1.bar(x[1], inference_time_Load_Prune, width, color='#F1BB71', edgecolor='black')
bars1_2b = ax1.bar(x[1], inference_time_Model_Prune, width, bottom=inference_time_Load_Prune, color='#ede658')
bars1_2c = ax1.bar(x[1], inference_time_Inf_Prune, width,
                   bottom=np.add(inference_time_Load_Prune, inference_time_Model_Prune), color='#4c8bf5')

bars1_3a = ax1.bar(x[3], inference_time_Load_GKD, width, color='#F1BB71', edgecolor='black')
# bars1_3b = ax1.bar(x[4], inference_time_Model_Default_TOSA, width, bottom=inference_time_Load_Default_TOSA, color='#ede658')
# bars1_3c = ax1.bar(x[4], inference_time_Inf_Default_TOSA, width, bottom=np.add(inference_time_Load_Default_TOSA, inference_time_Model_Default_TOSA), color='#4c8bf5')

bars1_4a = ax1.bar(x[4], inference_time_Load_KG_WISE, width, color='#F1BB71', edgecolor='black')
bars1_4b = ax1.bar(x[4], inference_time_Model_KG_WISE, width, bottom=inference_time_Load_KG_WISE, color='#ede658')
bars1_4c = ax1.bar(x[4], inference_time_Inf_KG_WISE, width,
                   bottom=np.add(inference_time_Load_KG_WISE, inference_time_Model_KG_WISE), color='#4c8bf5')

bars1_5a = ax1.bar(x[2], inference_time_Load_DQ, width, color='#F1BB71', edgecolor='black')
bars1_5b = ax1.bar(x[2], inference_time_Model_DQ, width, bottom=inference_time_Load_DQ, color='#ede658')
bars1_5c = ax1.bar(x[2], inference_time_Inf_DQ, width, bottom=np.add(inference_time_Load_DQ, inference_time_Model_DQ),
                   color='#4c8bf5')

ax1.set_ylabel('Time (sec)', fontsize=13)
ax1.set_title('B. Inference Time', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
# ax1.tick_params(axis='y', labelsize=13)
ax1.margins(x=0.08)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
# ax1.legend([bars1_1a, bars1_1b], ['Load', 'Inference'], loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=2)
# ax1.legend([bars1_1a, bars1_1b], ['Load', 'Inference'], loc='upper left', bbox_to_anchor=(0.55, 1), ncol=1)
# ax1.legend([bars1_1a, bars1_1b, bars1_1c], ['Pre-Process', 'Model Load', 'Inference'], loc='upper left', bbox_to_anchor=(0.55, 1), ncol=3)

""" Graph_SAINT bars """
gSaint_bars = [bars1_1a, bars1_1b, bars1_1c]
max_val = 0
for bars in gSaint_bars:
    max_val += bars[0].get_height()
    if bars is gSaint_bars[-1]:
        for bar in bars:
            yval = bar.get_height() + bar.get_y()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, int(max_val), ha='center', va='bottom')

""" GCNP Bars """
# gcnp_bars = [bars1_2a,bars1_2b,bars1_2c]
# max_val = 0
# for bars in gcnp_bars:
#     max_val += bars[0].get_height()
#     if bars is gcnp_bars[-1]:
#         for bar in bars:
#             yval = bar.get_height() + bar.get_y()
#             ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(max_val), ha='center', va='bottom')
""" DQ Bars """
dq_bars = [bars1_5a, bars1_5b, bars1_5c]
max_val = 0
for bars in dq_bars:
    max_val += bars[0].get_height()
    if bars is dq_bars[-1]:
        for bar in bars:
            yval = bar.get_height() + bar.get_y()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, int(max_val), ha='center', va='bottom')

""" KG-WISE Bars """
kgWise_bars = [bars1_4a, bars1_4b, bars1_4c]
max_val = 0
for bars in kgWise_bars:
    max_val += bars[0].get_height()
    if bars is kgWise_bars[-1]:
        for bar in bars:
            yval = bar.get_height() + bar.get_y()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, int(max_val), ha='center', va='bottom')

# Adding text values on top of the bars
# for bars in [bars1_1a, bars1_1b,bars1_5a,bars1_5b,bars1_5c ,bars1_1c,  bars1_4a, bars1_4b, bars1_4c]: # bars1_3a, bars1_3b, bars1_3c KG-TOSA
#     for bar in bars:
#         yval = bar.get_height() + bar.get_y()
#         ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(bar.get_height()), ha='center', va='bottom')

# Adding red "X" annotation for 'GCNP' bars
for bars in [bars1_2a, bars1_3a]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, 'T-OOM', ha='center', va='bottom',
                 color='red', fontsize=12, fontweight='bold', rotation=90)

# ax1.set_ylim(0, max(inference_time_Inf_Default + inference_time_Inf_Default_TOSA+inference_time_Inf_Prune + inference_time_Inf_KG_WISE) * 2)
# Set y-axis limit to accommodate stacked bars
max_ylim = max((inference_time_Load_Default[0] + inference_time_Model_Default[0] + inference_time_Inf_Default[0]),  # ,+
               # (inference_time_Load_Default_TOSA[0] + inference_time_Model_Default_TOSA[0] + inference_time_Inf_Default_TOSA[0]),# +
               (inference_time_Load_Prune[0] + inference_time_Model_Prune[0] + inference_time_Inf_Prune[0]),  # +
               (inference_time_Load_KG_WISE[0] + inference_time_Model_KG_WISE[0] + inference_time_Inf_KG_WISE[0]),
               (inference_time_Load_DQ[0] + inference_time_Model_DQ[0] + inference_time_Inf_DQ[0]), ) * 1.1
ax1.set_ylim(0, max_ylim)
print(f'MAX Y_LIM is {max_ylim}')

# fig.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/MAG_INF-TIME.pdf', dpi=1200, bbox_inches='tight',format='pdf')

# MAX RAM USAGE subplot
ax3 = fig.add_subplot(233)
bars3_1 = ax3.bar(x[0], ram_Default, width, color='#1aa260', label='Default')
bars3_2 = ax3.bar(x[1], ram_Prune_train, width, color='#1aa260', label='GCNP')
bars3_3 = ax3.bar(x[3], ram_GKD, width, color='#1aa260', label='GKD')
bars3_4 = ax3.bar(x[4], ram_KG_WISE, width, color='#1aa260', label='KG-WISE')
bars3_5 = ax3.bar(x[2], ram_DQ, width, color='#1aa260', label='DQ')

ax3.set_ylabel('Memory (GBs)', fontsize=13)
ax3.set_title('C. Max Memory Usage', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=10)
# ax3.tick_params(axis='y', labelsize=13)
ax3.margins(x=0.08)
ax3.set_ylim(0, max(ram_Default + ram_Prune_train + ram_KG_WISE + ram_DQ) * 1.1)
# ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
for bars in [bars3_1, bars3_5, bars3_4]:  # bars3_3
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, round(yval, 2), ha='center', va='bottom')

# Adding red "X" annotation for 'GCNP' bars
for bars in [bars3_2, bars3_3]:
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, 'T-OOM', ha='center', va='bottom',
                 color='red', fontsize=12, fontweight='bold', rotation=90)

# fig.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/DBLP_INF-MEM.pdf', dpi=1200, bbox_inches='tight',format='pdf')
# Adjust layout
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
fig.subplots_adjust(wspace=0.25, hspace=0.6)  # Adjust horizontal spacing between subplots.
if save:
    plt.savefig('MAG_INF_combined_plots.pdf', dpi=1200,
                bbox_inches='tight', format='pdf')

if True:
    num_targets = ['100', '200', '400', '800', '1600']  # Example target counts, replace with actual values if needed

    # Total Inference Times
    default_time = [142.33 for x in range(len(num_targets))]
    gcnp_time = [10 for x in range(len(num_targets))]
    DQ_time = [539 for x in range(len(num_targets))]
    # kg_tosa_time = [83.6 for x in range(len(num_targets))]
    kg_wise_time = [77.97, 81.42, 83.27, 89.79, 92.91]

    # MAX RAM Usage
    # Data for Max RAM Usage
    default_RAM = [106.22 for _ in range(len(num_targets))]
    gcnp_RAM = [0 for _ in range(len(num_targets))]
    DQ_RAM = [161.54 for _ in range(len(num_targets))]
    # kg_tosa_RAM = [40.68 for _ in range(len(num_targets))]
    kg_wise_RAM = [30.54, 30.66, 30.67, 30.7, 30.8]

    fig, axs = plt.subplots(1, 2, figsize=(7, 2))
    fig.suptitle('MAG Paper-Venue', fontsize=16, y=0.92)  # y=1.04)

    colors = ['#4c8bf5', '#F1BB71', '#CBF7DD', '#1aa260', '#9f62f5']

    # Plot lines for each category
    axs[0].plot(num_targets, default_time, marker='o', color=colors[0], label=labels[0])
    axs[0].plot(num_targets, gcnp_time, marker='X', markerfacecolor=colors[1], linestyle='', markeredgecolor='black',
                markersize=10, color=colors[1], label=labels[1])
    # axs[0].plot(num_targets, kg_tosa_time, marker='o', color=colors[2], label=labels[3])
    axs[0].plot(num_targets, kg_wise_time, marker='o', color=colors[3], label=labels[3])
    axs[0].plot(num_targets, DQ_time, marker='o', color=colors[4], label=labels[2])

    # Adding titles and labels
    # axs[0].set_title('Total Inference Time Against Query Size', fontsize=14)
    axs[0].set_xlabel('# Target Nodes Queried', fontsize=12)
    axs[0].set_ylabel('Time (sec)', fontsize=12)
    axs[0].set_xticks(num_targets)
    axs[0].set_xticklabels(num_targets, fontsize=12)
    axs[0].tick_params(axis='y', labelsize=12)
    axs[0].set_ylim(0, max(default_time[0], DQ_time[0]) + 20)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    # Adding legend
    # axs[0].legend(loc='upper left', bbox_to_anchor=(0.5, 1.3),ncols=4)

    # Adding value labels on the points
    # for i, v in enumerate(kg_tosa_time):
    #     ax.text(num_targets[i], v + 0.3, str(v), ha='center', va='bottom')
    # for i, v in enumerate(kg_wise_time):
    #     ax.text(num_targets[i], v + 0.3, str(v), ha='center', va='bottom')
    for i, v in enumerate(kg_wise_RAM):
        axs[1].text(num_targets[i], v - 0.8, str(v), ha='center', va='top',fontsize=10)

    for i, v in enumerate(default_RAM):
        axs[1].text(num_targets[i], v - 4.2, str(v), ha='center', va='top', fontsize=10)

    axs[1].plot(num_targets, default_RAM, marker='o', color=colors[0], label=labels[0])
    axs[1].plot(num_targets, gcnp_RAM, marker='X', markerfacecolor=colors[1], linestyle='', markeredgecolor='black',
                markersize=10, color=colors[1], label=labels[1])
    # axs[1].plot(num_targets, kg_tosa_RAM, marker='o', color=colors[2], label=labels[3])
    axs[1].plot(num_targets, kg_wise_RAM, marker='o', color=colors[3], label=labels[3])
    axs[1].plot(num_targets, DQ_RAM, marker='o', color=colors[4], label=labels[2])
    # axs[1].set_title('Max RAM Usage Against Query Size', fontsize=14)
    axs[1].set_xlabel('# Target Nodes Queried', fontsize=12)
    axs[1].set_ylabel('Memory (GB)', fontsize=12)
    axs[1].set_xticks(num_targets)
    axs[1].set_xticklabels(num_targets, fontsize=12)
    axs[1].tick_params(axis='y', labelsize=12)
    axs[1].set_ylim(0, max(default_RAM[0], DQ_RAM[0]) + 5)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    handles, labels = [], []

    h, l = axs[0].get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    # Create a single legend outside the subplots
    # fig.legend(handles, labels, loc='upper center',ncols=4,bbox_to_anchor=(0.5, 1.25))
    plt.tight_layout()

    if save:
        # plt.savefig('/home/afandi/GitRepos/Bar_graphs/TRAINING_wise_v_prune/MAG_INF_vTargets.pdf', dpi=1200,bbox_inches='tight', format='pdf')
        plt.savefig('MAG_INF_vTargets.pdf', dpi=1200,bbox_inches='tight', format='pdf')
plt.show()