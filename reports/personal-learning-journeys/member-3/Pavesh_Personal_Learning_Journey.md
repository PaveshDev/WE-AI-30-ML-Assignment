# Personal Learning Journey

**Student:** Pavesh T | **Student ID:** IT24100022  
**Group:** WE-AI-30 | **Role:** Member 3 – ML/Data-Mining Strategy & Modelling  
**Project:** Customer Segmentation Based on Purchasing Behaviour — UCI Online Retail

## My role and contribution

My work began with Member 2’s model-ready RFM table containing 3,362 customers. I was responsible for turning those prepared features into candidate customer groups. With AI-assisted implementation, I developed a four-group RFM baseline and compared K-Means, Ward agglomerative clustering and Gaussian Mixture Models. My contribution included checking the handoff, experimenting with group counts, analysing stability, generating figures and raw RFM profiles, exporting customer assignments, and preparing the technical handoff. I consumed the existing transformations and scaling; I did not repeat Member 1’s EDA or Member 2’s cleaning pipeline.

## What I learned and why I made these decisions

I learned that evaluating segmentation requires a different kind of judgement from supervised prediction. There were no known correct segment labels, so a clustering score could not be interpreted as prediction accuracy. I used the same scaled RFM matrix across methods and excluded CustomerID because differences between identifier numbers have no behavioural meaning. Reviewing the handoff also helped me understand why scaling matters: the units of spend should not dominate distances measured alongside days and order counts.

I tested two through eight groups and used Ward linkage because its variance-based merging suited scaled numerical data. Comparing silhouette, Davies–Bouldin, Calinski–Harabasz, sizes and profiles helped me justify a choice beyond an elbow plot. K-Means with two clusters led technically, with silhouette 0.4114 and substantial groups. I retained three clusters as a stable, finer alternative because the strongest internal result might still be too coarse for a retailer.

The GMM comparison was my clearest lesson in interpreting metrics. Eight components achieved the lowest primary AIC/BIC in the tested range, yet separation and seed stability were weaker. Component diagnostics showed very narrow distributions around repeated Frequency values. I learned to distinguish fitting a density closely from producing useful customer groups. I kept that solution as a diagnostic rather than presenting the lowest BIC as sufficient justification.

## Challenges, collaboration and handoff

One practical challenge was scoring customers with identical order counts. Average-rank scoring kept equal values together instead of arbitrarily splitting them to force equal-sized bins. This showed me that a simple baseline still needs defensible design decisions.

Stability required similar care. Cluster numbers can swap between runs without changing membership, so I used Adjusted Rand Index to compare partitions. Repeating the stochastic methods with five seeds showed consistent two-cluster K-Means assignments. I also recognised the limit: initialization stability does not establish future usefulness.

My work depended on Member 1’s data understanding and Member 2’s prepared features. Preserving their input and documenting checks made the boundary between stages explicit. I prepared assignments, profiles, comparisons and stability evidence for Member 4, keeping the period from 9 September 2011 reserved for future behavioural validation. This taught me that a useful handoff needs clear definitions and limitations, not just a CSV. I left final segment names, business recommendations and the final winner to Member 4.

## Responsible AI use and future improvement

I used Codex for substantial coding, review and documentation assistance. I grounded the technical decisions in executable results and documented comparisons rather than accepting generated explanations as evidence. Metrics came from the real dataset; they were not invented. Fresh-kernel reproduction and independent output verification checked the notebook results, assignments and input preservation. AI assistance made checking provenance and understanding the reasoning particularly important.

My main lesson is that defensible ML work combines evidence with an explanation of what that evidence cannot prove. Next time, I would extend stability checks to customer resampling and investigate GMM covariance and regularization sensitivity more systematically. Those improvements would address limitations of this study without confusing technical experimentation with Member 4’s business validation.

