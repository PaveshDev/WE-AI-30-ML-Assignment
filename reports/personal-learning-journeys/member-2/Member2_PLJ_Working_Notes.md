# Member 2 Personal Learning Journey — Working Notes & Complete Draft Answers

**Student:** Sathira S | **Student ID:** IT24101155  
**Group:** WE-AI-30 | **Role:** Member 2 – Data Quality, Cleaning, Preprocessing, Feature Engineering, Transformations, Scaling  
**Project:** Customer Segmentation Based on Purchasing Behaviour — UCI Online Retail  

> **Note:** This document contains the full-length 12 technical draft answers and the personal confirmation questions. The final submitted one-page report is maintained in [`Member2_Personal_Learning_Journey.md`](Member2_Personal_Learning_Journey.md).

---

## Part 1: Complete Set of 12 Revised Technical Draft Answers

### 1. Cleaning Order
I learned that data cleaning operations cannot be run in an arbitrary sequence because each step establishes the valid cohort for subsequent calculations. Removing 5,268 exact duplicates first was critical so repeated line items would not artificially inflate customer monetary totals or transaction counts. Filtering out 135,037 missing CustomerIDs and 8,872 cancellations before computing customer aggregates ensured we only summarized genuine purchases. Most importantly, enforcing the 2011-09-09 date cutoff before calculating RFM prevented future orders from falsely lowering customers' Recency scores. Every step logically depends on earlier filtering.

### 2. Missing CustomerID
Removing 135,037 records with missing CustomerIDs—representing approximately 24.9% of the original 541,909 rows—was a major project trade-off. Our objective was customer segmentation, which inherently requires tracking repeat purchasing behavior over time for individual buyers. Without CustomerIDs, transactions cannot be reliably linked to distinct accounts, and grouping anonymous guest or till sales under arbitrary customer keys would severely distort frequency and spending. While this exclusion means our segments represent only registered, identifiable customers, establishing this explicit cohort boundary was necessary for reliable RFM modeling.

### 3. Cancellation Criterion — OR versus AND
In `02_cleaning_and_rfm.py`, I implemented cancellation removal using union (OR) logic: `mask_cancel = mask_c_prefix | mask_neg_qty`. An intersection (AND) rule would only catch records possessing both a 'C' invoice prefix and a negative quantity, risking missing negative quantity adjustments that lack the prefix. Although nearly all cancellation rows in this dataset share both attributes, OR is the defensively correct implementation. An earlier informal draft of the preprocessing log described this step using AND; ensuring the documentation strictly matches the executed code was an important lesson in technical reproducibility.

### 4. Temporal Leakage
The raw dataset spans December 2010 to December 2011. To prevent temporal data leakage, I applied a strict date cutoff at 2011-09-09 before building the RFM table, retaining 231,806 transactions across 3,362 historical customers. If purchases after this date had been included in Recency or Monetary totals, future information would contaminate historical customer profiles. Reserving all transactions from 9 September 2011 onward ensures a clean, untouched validation period for Member 4, who will confirm the final evaluation window endpoint when evaluating segment stability and behavioral drift.

### 5. RFM Metric Definitions
In my pipeline, Recency measures whole elapsed days from a customer’s latest retained purchase before 2011-09-09 to that snapshot date, capturing purchase recency rather than customer tenure. Frequency counts distinct invoice numbers (`nunique`), measuring separate shopping visits rather than total unit quantities, so a large single basket does not inflate repeat purchase frequency. Monetary is the sum of positive line spend (`Quantity * UnitPrice`). Because cancellations were filtered out earlier rather than subtracted from sales, Monetary represents gross customer spend rather than net profit, which is vital when interpreting cluster value.

### 6. Log Transformations
Both Frequency and Monetary showed severe right skew; maximum spend was £177,729.62 compared to a median of £555.02. I applied `np.log1p()` to compress these wide positive tails so extreme accounts would not dominate Euclidean distance calculations. However, log transformation does not make data Gaussian. In our cohort, 1,366 customers (40.63%) made only a single purchase, creating a prominent spike at `log1p(1) ≈ 0.693`. Member 3’s GMM diagnostics confirmed that this discrete clump persisted, showing that mathematical transformation compresses scale but cannot erase discrete behavioral realities.

### 7. Feature Scaling
I used `StandardScaler` to scale Recency, `Log_Frequency`, and `Log_Monetary` to zero mean and unit variance. `MinMaxScaler` was rejected because residual extreme values—such as our top spender—would drastically expand the denominator `(max - min)`. That would compress over 95% of typical customers into a narrow band near zero and diminish meaningful variance. `StandardScaler` standardizes the spread across all three features based on variance, ensuring Euclidean distance in Member 3’s K-Means clustering weighs recency, order frequency, and monetary value equally without letting outliers suppress normal customer differences.

### 8. Genuine Challenges
A major technical challenge was handling extreme right skew in Monetary and Frequency without discarding high-spending wholesale accounts. Applying arbitrary outlier caps would have destroyed real business signals, so I addressed this by combining `log1p` transformation with `StandardScaler`, successfully compressing ranges while preserving true relative rankings. Conceptually, managing 135,037 missing CustomerID rows required deciding whether to impute or drop them. Recognizing that unverified imputation would distort customer-level metrics, I established a clear project boundary: our analysis specifically profiles identifiable, registered customer behavior.

### 9. Team Collaboration
My preprocessing was directly guided by Member 1’s exploratory data analysis (Q1–Q12), which flagged missing identifiers, negative quantities, and non-product stock codes. I addressed these findings systematically, removing codes like POST and DOT while retaining 171 manual 'M' transactions across 127 customers. My output—a clean 3,362-row RFM table—formed the direct input for Member 3’s clustering. By exporting both raw and scaled features, I enabled Member 3 to fit models on standardized variables while generating intuitive business profiles using original pounds, days, and visit counts.

### 10. Automated Input Audit
Before running clustering models, Member 3 executed an automated input audit on my `rfm_table.csv`. The audit verified the 9-column schema, confirmed exactly 3,362 unique customers with zero null or infinite values, and checked that all transactions occurred before 2011-09-09. Crucially, it verified that `Frequency_scaled` and `Monetary_scaled` matched `StandardScaler(log1p())` within a 1e-7 tolerance. Passing every contract check with zero required downstream adjustments demonstrated that my preprocessing was reliable, fully reproducible, and honored our agreed interface.

### 11. Responsible AI Usage
Using AI assistance effectively requires independent verification rather than blind acceptance. Throughout the preprocessing work, every AI-assisted code suggestion and markdown draft was validated against actual dataset metrics. I checked row counts at every funnel step (from 541,909 raw records down to 231,806 cleaned rows), verified column data types, inspected mathematical formulas against pandas implementations, and confirmed reproducibility using clean-environment script runs. Relying on AI without cross-referencing code against the actual data dictionary and raw records would risk silent data loss or distorted features.

### 12. What I Would Do Differently
If I were restarting this work, I would make two concrete improvements. First, I would embed automated `assert` statements directly into the cleaning script after each step (e.g., verifying `len(df) == 231806` and `df['CustomerID'].isnull().sum() == 0`), making data validation immediate and self-testing. Second, I would replace the hard-coded absolute directory path with dynamic repository discovery using `pathlib.Path(__file__)`, ensuring the pipeline runs out-of-the-box on any teammate’s machine without manual configuration. These enhancements would further strengthen code portability and pipeline maintainability.

---

## Part 2: Confirmed Personal Reflections (Sathira S, IT24101155)

The following 12 items record my actual personal experiences, decisions, and learning throughout the project:

1. **Cleaning Order (Section 2.1):**
   I did not fully understand all the dependencies at the beginning. While working through the preprocessing steps and checking the results, I realised that the order matters. For example, duplicate records needed to be handled before calculating customer spending because otherwise the same transaction could contribute more than once. I also learned that the temporal cutoff needed to be applied before building RFM features so that future transactions did not affect the historical customer features. This made me plan the later preprocessing steps more carefully.

2. **Missing CustomerID (Section 2.2):**
   When I saw that 135,037 rows had missing CustomerIDs, removing almost 25% of the original data initially seemed like a large loss. However, our task depends on identifying individual customers over time. Without a CustomerID, I could not reliably calculate Recency, Frequency or Monetary values for a customer. I considered the purpose of the analysis and decided that keeping anonymous transactions would make the customer-level RFM features unreliable. I also understood that this means our final customer cohort only represents identifiable customers.

3. **Cancellation Criterion — OR Logic (Section 2.3):**
   I used OR logic because either condition could indicate an invalid or cancelled transaction. A transaction could have an InvoiceNo beginning with C, while another negative adjustment might be identified through a negative Quantity even if its invoice format was different. Therefore, using `mask_c_prefix | mask_neg_qty` was a safer way to catch both cases. Later, when reviewing the preprocessing documentation, I noticed that an earlier informal explanation did not clearly describe the OR condition. I corrected the documentation so that it matched the actual implementation.

4. **Temporal Leakage (Section 2.4):**
   I had learned about train and test separation before, but temporal leakage became much clearer to me during this assignment. I understood that when working with time-based transaction data, randomly mixing future transactions with historical customer features could produce unrealistic results. Using the 2011-09-09 cutoff helped me understand why future information must remain unavailable when creating features for earlier periods. This changed the way I think about validation because it is not only about splitting data; it is also about respecting when information becomes available.

5. **RFM Definitions (Section 2.5):**
   The main area I needed to think about was the exact meaning of Frequency and Monetary. Frequency could have been based on product rows, quantities or invoices. I understood that distinct invoices better represent separate shopping occasions, so Frequency was calculated using unique InvoiceNo values. For Monetary, I used the positive transaction value represented by Quantity multiplied by UnitPrice after the cancellation and invalid-transaction cleaning steps. This experience showed me that feature definitions must be documented clearly because different definitions can change the resulting customer segments.

6. **Log Transformation (Section 2.6):**
   Before this project, I mostly thought of log transformation as a way to make skewed data look more normally distributed. Through this project, I learned that its more important purpose here was to reduce the effect of very large positive values. The difference between the median customer spend and the extreme maximum showed how strongly right-skewed the data was. After using `np.log1p()`, the extreme values were compressed, but the one-time buyer spike still remained. This helped me understand that log transformation does not automatically make real-world data Gaussian.

7. **Feature Scaling (Section 2.7):**
   I considered the difference between MinMaxScaler and StandardScaler before finalising the scaling method. Because the dataset still contained some very large customer values, MinMaxScaler could compress most ordinary customers into a small part of the 0–1 range. StandardScaler was more suitable for preparing Recency, Frequency and Monetary features for distance-based clustering after the log transformation. This also helped prevent one feature from dominating only because of its numerical scale. I learned that scaling should be selected based on the data distribution and modelling method.

8. **Genuine Challenges (Section 3.1):**
   One major challenge was handling the highly skewed customer behaviour without simply deleting important high-value customers. Large spend values could be genuine wholesale customers rather than errors, so I did not want to apply arbitrary outlier removal. Instead, I used log transformation and scaling while preserving those customers. Another challenge was deciding what to do with transactions that could not be connected to a CustomerID. Removing them reduced the dataset significantly, but I learned to prioritise the reliability of the customer-level analysis over simply keeping the maximum number of rows.

9. **Team Collaboration (Section 3.2):**
   I worked with the outputs and findings from Member 1's EDA when deciding how the data should be cleaned. The EDA helped identify issues such as special stock codes, unusual quantities and other transaction patterns that required preprocessing decisions. After preparing the cleaned transactions and RFM table, I handed the processed features to Member 3 for clustering. We shared the work through the project repository and group communication, and I made sure my preprocessing documentation explained the feature definitions and transformations clearly so that the next stage could use the data correctly.

10. **Automated Input Audit (Section 3.3):**
    I found Member 3's automated audit useful because it showed that preprocessing should not only produce a file that looks correct. The output also needs to satisfy the requirements of the next stage. Seeing checks for the schema, row count, missing values, numerical ranges and algebraic consistency gave me more confidence that the handoff was reliable. The fact that no downstream correction was required showed the value of validating the interface between team members. In future projects, I would introduce similar checks earlier in my own pipeline.

11. **Responsible AI Usage (Section 4.1):**
    During the project, I used AI tools such as ChatGPT and AI-assisted development tools to help explain concepts, review code, troubleshoot errors and improve documentation. I did not treat AI output as automatically correct. I compared suggestions with the actual dataset, checked row counts and feature values, reran the preprocessing pipeline, inspected the generated files and compared the results with our documented requirements. This taught me that AI is useful for support, but I still need to understand, test and take responsibility for the final work.

12. **What I Would Do Differently (Section 4.2):**
    If I restarted this part of the project, I would add validation checks immediately after each major cleaning stage rather than mainly checking the final outputs. Simple assert statements for expected columns, null values, row counts and valid numerical ranges would make problems easier to detect. I would also organise file paths more dynamically using pathlib instead of relying on fixed paths. Finally, I would document important preprocessing decisions while implementing them rather than updating some explanations later. This would reduce the chance of documentation becoming inconsistent with the final code.
