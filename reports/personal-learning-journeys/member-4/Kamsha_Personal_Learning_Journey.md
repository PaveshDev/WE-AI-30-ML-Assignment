# Personal Learning Journey

**Student:** Kamsha S | **Student ID:** IT24100697 
**Group:** WE-AI-30 | **Role:** Member 4 – Evaluation and Presentation  
**Project:** Customer Segmentation Based on Purchasing Behaviour — UCI Online Retail

## My role and contribution

My work began with Member 3's cluster assignments and evaluation metrics. I was responsible for turning those modelling outputs into a defensible, business-relevant recommendation. With AI-assisted implementation, I reviewed the handoff, evaluated internal clustering quality, analysed initialization stability, profiled customer clusters using historical RFM values, performed a three-month future behavioural validation, checked for data leakage, and synthesised the combined evidence into a final method selection. I named and interpreted the three final segments and translated them into business and marketing actions. I did not perform EDA, data cleaning, or model training; those responsibilities belonged to Members 1, 2, and 3 respectively.

## What I learned and why I made these decisions

I learned that evaluating unsupervised learning requires assembling multiple kinds of evidence rather than selecting the candidate with the single highest metric. Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index each capture a different geometric aspect of clustering quality, and no single score is sufficient on its own. I found that K-Means k=2 produced the strongest purely geometric performance, while K-Means k=3 provided a meaningful additional customer tier and stronger practical differentiation.

The stability analysis reinforced this. Using Mean Adjusted Rand Index across five random seeds showed that K-Means k=3 was near-perfectly stable (Mean ARI > 0.99), which gave me confidence that the segments were not an artefact of a particular initialisation.

The most novel aspect for me was the three-month future behavioural validation. I derived observed future behaviour from raw transaction data dated from 9 September to 9 December 2011, keeping that period strictly separate from the historical window used for training. I verified the leakage check explicitly: the cluster assignment columns in the validation set matched the originally loaded assignments exactly, confirming that no future information had influenced the segment labels. This taught me that temporal data leakage is subtle and must be tested, not assumed absent.

Synthesising all the evidence — internal quality, stability, cluster size balance, RFM profiles, and future validation — led me to recommend K-Means k=3 as the overall segmentation solution. I learned to frame that recommendation honestly: it is not that k=3 won every metric, but that it offered the best balance of geometric quality, stability, interpretability, and practical value for a retailer.

## Challenges, collaboration and handoff

One challenge was interpreting clusters carefully without overstating certainty. The three segments I named — Cluster 0 (Champions / Loyal High-Value), Cluster 1 (At-Risk / Inactive), and Cluster 2 (Regular / Developing) — are business interpretations of unsupervised outputs, not ground-truth labels. I learned to present these personas as useful approximations grounded in observed RFM patterns and future behaviour, rather than definitive classifications.

A second challenge was structuring the evaluation notebook so that each section followed logically from the previous one. Removing abandoned placeholder cells and running the full notebook from a fresh kernel with zero execution errors required careful tracing of dependencies and consistent variable naming across cells.

My work depended directly on Member 3's exported metrics, stability data, and cluster assignments. Consuming that handoff clearly — understanding what had already been computed and what remained for me — made the evaluation tractable and helped me avoid duplicating or undermining earlier work.

## Responsible AI use and future improvement

I used AI assistance for code generation, documentation drafting, and iterative review throughout the evaluation notebook and this report. I grounded all findings in executable notebook results rather than accepting generated explanations as evidence. Metric values, cluster sizes, and future validation figures came from the actual dataset. I verified the notebook by running it from a fresh kernel to confirm reproducibility before submission.

My main lesson is that a rigorous ML evaluation requires honest acknowledgement of what the evidence does and does not prove. Next time, I would extend the future validation window if data permitted, incorporate customer revenue weighting into the validation metrics, and explore statistical significance testing for cluster profile differences. I would also document segment interpretation decisions more formally, so that business stakeholders could trace a recommendation back to specific observed figures rather than narrative descriptions.
