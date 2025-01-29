#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:24:28 2024

@author: afandi
"""
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
num_targets = 1600
categories=["Identifier","SameAs","Creator","Primary\nE-Edition","Title","Person","DOI","Volume","Pagination","Toc-Page"]
chunks=[1518,754,1420,866,1440,4140,368,461,545,45]
chunks_percentage=[8.2,5.9,15,9.3,15.5,52,6.2,14.9,34.5,32.8]
chunks_total=[18513,12780,9467,9312,9291,7962,5936,3094,1580,138]
chunks_diff=[elem-chunks[idx] for idx,elem in enumerate(chunks_total)]

# if percentage:
#     chunks = [52.0, 8.2, 15.5, 15.0, 9.3, 5.9, 34.5, 14.9, 6.2, 32.8]

# Create a bar graph
plt.figure(figsize=(6, 3))
# create data
# plot bars in stack manner
kgwise_bars=plt.bar(categories, chunks, color='#1aa260')
for idx,bar in enumerate(kgwise_bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(chunks_percentage[idx])}%', ha='center', va='bottom', fontsize=10)

plt.bar(categories, chunks_diff, bottom=chunks, color='#4c8bf5')
# plt.xlabel("Teams")
plt.ylabel("Number of Chunks")
plt.legend(["KG-WISE", "KG-TOSA"])
plt.xticks(categories, rotation=25, fontsize=10)
for pos in ['right', 'top']: #'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)
plt.tight_layout()
ax = plt.gca()
ax.set_ylim([0, 20000])
plt.savefig(f'dblp_chunks.pdf', dpi=1200, bbox_inches='tight',format='pdf')
plt.show()



# bars = plt.bar(categories, chunks, color='#4c8bf5', width=0.5,)
# # Add text on top of each bar
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval}', ha='center', va='bottom', fontsize=12)
#
# # Add labels and title
# # plt.xlabel('Category', fontsize=12)
# plt.ylabel('Number of Chunks', fontsize=12) if percentage == False else plt.ylabel('% of Total Chunks Loaded', fontsize=12)
# plt.title(f'DBLP #Chunks Accessed ({num_targets} Targets)', fontsize=12)
# plt.xticks(categories, rotation=25, fontsize=8)  # Rotate x-ticks for better visibility
# if percentage == False:
#     plt.ylim((0,max(chunks)+550))
# else:
#     plt.ylim((0,100))
#
# # Show the plot
# plt.tight_layout()
# plt.savefig(f'dblp_{num_targets}_{type_}.pdf', dpi=1200, bbox_inches='tight',format='pdf')
# plt.show()