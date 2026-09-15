import json

nb_path = 'd:/Y3S1/WE-AI-30-ML-Assignment/members/member-4/notebooks/04_evaluation.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replacements
code_A_B = '''# Visualizing the comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Internal Clustering Quality and Stability Assessment", fontsize=18, fontweight='bold', y=1.02)

# Create a combined label for plotting
eval_table['model_name'] = eval_table['method'] + " (k=" + eval_table['k'].astype(str) + ")"
stability_plot_df = stability_df.copy()
stability_plot_df['model_name'] = stability_plot_df['method'] + " (k=" + stability_plot_df['k'].astype(str) + ")"

# 1. Silhouette Score
sns.barplot(data=eval_table, x='silhouette', y='model_name', ax=axes[0, 0], palette='Blues_r')
axes[0, 0].set_title('Silhouette Score (Higher is better)', fontsize=14)
axes[0, 0].set_xlabel('Score [-1 to 1]')
axes[0, 0].set_ylabel('')

# 2. Davies-Bouldin Index
sns.barplot(data=eval_table, x='davies_bouldin', y='model_name', ax=axes[0, 1], palette='Reds')
axes[0, 1].set_title('Davies-Bouldin Index (Lower is better)', fontsize=14)
axes[0, 1].set_xlabel('Index')
axes[0, 1].set_ylabel('')

# 3. Calinski-Harabasz Score
sns.barplot(data=eval_table, x='calinski_harabasz', y='model_name', ax=axes[1, 0], palette='Greens_r')
axes[1, 0].set_title('Calinski-Harabasz Score (Higher is better)', fontsize=14)
axes[1, 0].set_xlabel('Variance Ratio')
axes[1, 0].set_ylabel('')

# 4. Stability (Mean ARI)
sns.barplot(data=stability_plot_df.sort_values('ari_mean', ascending=False), 
            x='ari_mean', y='model_name', ax=axes[1, 1], palette='Purples_r')
axes[1, 1].set_title('Initialization Stability (Mean ARI, max 1.0)', fontsize=14)
axes[1, 1].set_xlabel('Mean Adjusted Rand Index')
axes[1, 1].set_ylabel('')
axes[1, 1].set_xlim(0, 1.05)

plt.tight_layout()
plt.show()'''

code_C = '''# Visualise cluster sizes as percentages
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Cluster Size Distributions Across Candidates (% of Customers)", fontsize=18, fontweight='bold', y=1.02)
axes = axes.flatten()

for i, m in enumerate(methods):
    counts = assignments_df[m].value_counts().sort_index()
    pcts = (counts / len(assignments_df)) * 100
    sns.barplot(x=pcts.index, y=pcts.values, ax=axes[i], palette='crest')
    axes[i].set_title(m.replace('_', ' '), fontsize=14)
    axes[i].set_xlabel("Cluster ID")
    axes[i].set_ylabel("Percentage (%)")
    for j, val in enumerate(pcts.values):
        axes[i].text(j, val + 1.5, f"{val:.1f}%", ha='center', fontsize=11, fontweight='bold')
    axes[i].set_ylim(0, 100)

plt.tight_layout()
plt.show()'''

code_D = '''# Visualize Profiles (Boxplots) for the main candidates (KMeans k=2 and KMeans k=3)
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle("Historical RFM Distributions: K-Means k=2 vs k=3", fontsize=18, fontweight='bold', y=1.02)

# k=2
sns.boxplot(data=assignments_df, x='KMeans_k2', y='Recency', ax=axes[0, 0], palette='Pastel1')
axes[0, 0].set_title('K-Means (k=2) - Recency', fontsize=14)
axes[0, 0].set_ylabel('Days Since Last Purchase')

sns.boxplot(data=assignments_df, x='KMeans_k2', y='Frequency', ax=axes[1, 0], palette='Pastel1')
axes[1, 0].set_title('K-Means (k=2) - Frequency', fontsize=14)
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylabel('Order Count (log scale)')

sns.boxplot(data=assignments_df, x='KMeans_k2', y='Monetary', ax=axes[2, 0], palette='Pastel1')
axes[2, 0].set_title('K-Means (k=2) - Monetary', fontsize=14)
axes[2, 0].set_yscale('log')
axes[2, 0].set_ylabel('Total Spend £ (log scale)')

# k=3
sns.boxplot(data=assignments_df, x='KMeans_k3', y='Recency', ax=axes[0, 1], palette='Pastel2')
axes[0, 1].set_title('K-Means (k=3) - Recency', fontsize=14)
axes[0, 1].set_ylabel('Days Since Last Purchase')

sns.boxplot(data=assignments_df, x='KMeans_k3', y='Frequency', ax=axes[1, 1], palette='Pastel2')
axes[1, 1].set_title('K-Means (k=3) - Frequency', fontsize=14)
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylabel('Order Count (log scale)')

sns.boxplot(data=assignments_df, x='KMeans_k3', y='Monetary', ax=axes[2, 1], palette='Pastel2')
axes[2, 1].set_title('K-Means (k=3) - Monetary', fontsize=14)
axes[2, 1].set_yscale('log')
axes[2, 1].set_ylabel('Total Spend £ (log scale)')

plt.tight_layout()
plt.show()'''

code_E = '''# STEP 10: CREATE VISUALISATIONS (Future Validation)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Three-Month Future Validation (Sept 9 - Dec 9, 2011)", fontsize=18, fontweight='bold', y=1.02)

# Create a combined label for plotting
comparison_table['Label'] = comparison_table['Method'].str.replace('_', ' ') + " (C" + comparison_table['Cluster'].astype(str) + ")"

sns.barplot(data=comparison_table, x='Repeat_Purchase_Rate', y='Label', ax=axes[0, 0], palette='rocket')
axes[0, 0].set_title('Future Repeat Purchase Rate', fontsize=14)
axes[0, 0].set_xlabel('Proportion of Customers Purchasing')
axes[0, 0].set_ylabel('')
axes[0, 0].set_xlim(0, 1)

sns.barplot(data=comparison_table, x='Median_Future_Spend', y='Label', ax=axes[0, 1], palette='rocket')
axes[0, 1].set_title('Median Future Spend', fontsize=14)
axes[0, 1].set_xlabel('Spend (£)')
axes[0, 1].set_ylabel('')

sns.barplot(data=comparison_table, x='Median_Future_Orders', y='Label', ax=axes[1, 0], palette='rocket')
axes[1, 0].set_title('Median Future Orders', fontsize=14)
axes[1, 0].set_xlabel('Number of Orders')
axes[1, 0].set_ylabel('')

sns.barplot(data=comparison_table, x='Total_Future_Spend', y='Label', ax=axes[1, 1], palette='rocket')
axes[1, 1].set_title('Total Future Spend Generated', fontsize=14)
axes[1, 1].set_xlabel('Total Spend (£)')
axes[1, 1].set_ylabel('')

plt.tight_layout()
plt.show()'''

md_takeaway = '''### Final Evaluation Takeaway

- **Selection:** K-Means with k=3 is the recommended segmentation model.
- **Quality & Stability:** It successfully balances strong geometric clustering quality (Silhouette = 0.37) with near-perfect initialization stability (Mean ARI > 0.99), ensuring reproducible and robust groupings.
- **Behavioral Distinctions:** The three clusters clearly divide the customer base into actionable tiers:
  - **Champions (Cluster 0):** Highly engaged, high-spending VIPs.
  - **At-Risk / Inactive (Cluster 1):** Low recent activity, negligible future spend.
  - **Regular / Developing (Cluster 2):** The middle tier of active, moderate spenders.
- **Business Utility:** Unlike the RFM baseline (which uses fixed thresholds) or K-Means k=2 (which lumps VIPs into a broader 'active' pool), K-Means k=3 is entirely data-driven and provides distinct, actionable marketing segments. It allows the marketing team to safely reward Champions, selectively attempt win-backs for the At-Risk group, and focus upsell strategies on the Developing core.'''

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0:
        src = "".join(cell['source'])
        if '# 1. Silhouette Score' in src and 'axes[0, 0]' in src and 'calinski_harabasz' in src:
            cell['source'] = [code_A_B]
        elif 'counts = assignments_df[m].value_counts().sort_index()' in src and 'axes[i].set_ylim(0, 100)' in src:
            cell['source'] = [code_C]
        elif '# k=2' in src and 'axes[2, 0].set_yscale(\'log\')' in src and 'axes[2, 1].set_yscale(\'log\')' in src:
            cell['source'] = [code_D]
        elif '# STEP 10: CREATE VISUALISATIONS' in src or ('sns.barplot' in src and 'Total_Future_Spend' in src):
            cell['source'] = [code_E]

# Append takeaway
nb['cells'].append({"cell_type": "markdown", "metadata": {}, "source": [md_takeaway]})

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook visual cleanup complete.")
