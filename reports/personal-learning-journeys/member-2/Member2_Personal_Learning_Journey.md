# Personal Learning Journey

**Student:** Sathira S | **Student ID:** IT24101155  
**Group:** WE-AI-30 | **Role:** Member 2 – Data Cleaning, Preprocessing and RFM Feature Engineering  
**Project:** Customer Segmentation Based on Purchasing Behaviour — UCI Online Retail  

## My Role and Contribution

My responsibility in WE-AI-30 was to turn the raw 541,909-row transaction dataset into a clean, model-ready RFM table for Member 3's clustering. Guided by Member 1's exploratory analysis, I designed a six-step cleaning pipeline, aggregated customer-level Recency, Frequency, and Monetary features, applied log transformations, and standardized them with StandardScaler. Deliverables include cleaned transactions (231,806 rows, 8 columns), an RFM table (3,362 customers, 9 columns), four diagnostic figures, and technical documentation. I observed the project boundary: I did not run clustering, interpret segments, or use transactions on or after the 2011-09-09 cutoff, handing the feature matrix directly to Member 3.

## What I Learned and Why I Made These Decisions

I did not fully understand all dependencies at the start. While testing steps and checking outputs, I realised that order matters. Removing 5,268 exact duplicates first prevented repeated line items from inflating spending. Removing 135,037 rows lacking CustomerIDs (~24.9% of raw data) seemed a large loss. However, segmentation requires tracking buyers over time; anonymous records would make customer features unreliable, so our cohort represents identifiable customers only. Applying the 2011-09-09 cutoff before building RFM features prevented future purchases from leaking into historical recency and spend, showing that validation requires respecting when information becomes available.

To detect cancellations, I used a union (OR) criterion (`mask_cancel = mask_c_prefix | mask_neg_qty`), catching C-prefix invoices and un-prefixed negative adjustments. When reviewing documentation later, I noticed an earlier informal explanation did not clearly state the OR condition and corrected it to match the code.

For RFM definitions, Recency measures whole days to the snapshot date. I calculated Frequency using distinct invoice counts (`nunique`) rather than item rows because invoices better represent separate shopping visits. Monetary represents gross positive spend (`Quantity * UnitPrice`) after removing cancellations and non-product codes (POST, DOT, C2, S; 1,248 rows removed), while retaining 171 manual 'M' lines across 127 customers. Because maximum spend reached £177,729.62 against a median of £555.02, `np.log1p()` compressed extreme values. This reduced skew, but the spike from 1,366 one-order buyers (40.63%) remained, showing that log transformation does not make real-world data Gaussian. After transformation, I selected StandardScaler over MinMaxScaler because residual extreme values would compress typical customers into a narrow 0–1 band.

## Challenges, Collaboration and Handoff

One challenge was handling highly skewed spend without deleting high-value accounts. Because large purchases could reflect genuine wholesale customers, I avoided arbitrary outlier capping and used log transformation and scaling. Deciding to exclude transactions without CustomerIDs was another challenge, where I prioritised feature reliability over row count.

My work connected Member 1's EDA with Member 3's clustering. The EDA helped identify special stock codes, unusual quantities, and spending patterns. After preparing the cleaned transactions and RFM table, I shared the files and documentation through our repository. Member 3's automated input audit checked the schema, 3,362 rows, zero null or infinite values, date bounds (< 2011-09-09), and scaling identities within 1e-7 tolerance. Passing without downstream fixes gave me confidence in my pipeline and showed the value of validating interfaces.

## Responsible AI Use and What I Would Do Differently

During the project, I used AI tools such as ChatGPT and AI-assisted development tools to help explain concepts, review code, troubleshoot errors and improve documentation. I did not treat AI output as automatically correct. I compared suggestions with the actual dataset, checked row counts and feature values, reran the preprocessing pipeline, inspected the generated files and compared the results with our documented requirements. This taught me that AI is useful for support, but I still need to understand, test and take responsibility for the final work.

If I restarted this phase, I would add validation checks immediately after each cleaning stage rather than mainly checking final outputs, using assert statements for row counts, columns, and nulls. I would also organise file paths dynamically using pathlib instead of fixed paths, and document important preprocessing decisions while implementing them to keep documentation consistent with the code.
