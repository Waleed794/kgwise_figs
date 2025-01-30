import matplotlib.pyplot as plt

# Data
models = ['Graph-SAINT', 'KG-WISE']
plt.rcParams.update({'font.size': 10})
c=1
DBLP_Emission = [0.0003936666667, 0.0001613333333]  # Raw emission values

# Normalize emissions
normalized_emission = [1, DBLP_Emission[0] / DBLP_Emission[1]]  # Graph-SAINT = 1, KG-WISE = ratio

# Scale the raw values (e.g., by 10^6 to make them larger)
scale_factor = 10**4  # Adjust this factor as needed
scaled_emission = [val * scale_factor for val in DBLP_Emission]

# Create figure with two side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(5, 2.2))  # Two plots side by side

# First plot: Scaled raw emissions
axes[0].bar(models, scaled_emission, color=['#4c8bf5', '#1aa260'], width=0.6)
# axes[0].set_title('Carbon Emissions', fontsize=12)
axes[0].set_ylabel('Emissions(Kg CO$_2$ × $10^4$)', fontsize=10)  # Use scientific notation in the label
axes[0].set_ylim((0,5))
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
# Add text values on top of bars (scaled)
for bar, raw_val in zip(axes[0].patches, scaled_emission):
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

# Second plot: Normalized emissions
axes[1].bar(models, normalized_emission, color=['#4c8bf5', '#1aa260'], width=0.6)
# axes[1].set_title('Emission Savings', fontsize=12)
axes[1].set_ylabel('Relative Saving Times', fontsize=10)
axes[1].set_ylim((0,5))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
# Add text values on top of bars (normalized)
for bar, norm_val in zip(axes[1].patches, normalized_emission):
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}x', ha='center', va='bottom', fontsize=10)

# Layout adjustment
plt.tight_layout()
# fig.suptitle('Carbon Emission Comparison', fontsize=14, y=1.05)

# Save the plot
plt.savefig('CarbonEmission_combined.pdf', dpi=1200, bbox_inches='tight', format='pdf')
plt.show()