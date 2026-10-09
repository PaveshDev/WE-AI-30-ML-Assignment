# Member 3

Core Responsibility:
ML/Data-Mining Strategy, Baseline Method,
Alternative Models, Model Experiments, Parameter Analysis,
Model Comparison and Validation Design.

## Completed modelling contribution

- [Executed notebook](notebooks/03_modelling_and_comparison.ipynb)
- [Modelling strategy](docs/modelling_strategy.md)
- [Actual comparison evidence](docs/model_comparison.md)
- [Member 4 handoff](docs/member3_handoff.md)
- [Decision-log integration draft](docs/modelling_decisions.md)
- [Frozen assignments](../../data/processed/cluster_assignments.csv)

The leading technical candidate is K-Means k=2; k=3 is retained as a finer secondary
candidate. Ward k=2 and GMM k=2 provide cross-method comparators alongside the fixed
four-group RFM baseline. GMM k=8 is retained explicitly as an AIC/BIC diagnostic.
Member 4 owns future validation, business names and final method choice.

## Run from a clone

Use Python 3.11 and a virtual environment. From the repository root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -r members/member-3/requirements-notebook.txt
.venv/Scripts/python.exe members/member-3/notebooks/run_notebook.py --verify-reproducible
.venv/Scripts/python.exe members/member-3/notebooks/verify_outputs.py
```

On Linux/macOS use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.
The existing RFM and historical cleaned-transactions CSVs must be present. This workflow
does not download raw data or rerun other members' pipelines.

For matching numerical versions, the executed environment used Python 3.11.3,
numpy 1.26.3, pandas 2.3.3, scikit-learn 1.4.0, scipy 1.11.4 and matplotlib 3.9.4.
Root requirements remain unpinned following the existing project convention; if exact
reproduction is needed, install these recorded versions before running. The full set of
relevant tested package versions is in `results/run_manifest.json`. Different library
versions can change numerical results and cluster numbering; the comparison guards
fail visibly if the recorded leading-candidate claims cease to hold.

`run_notebook.py` generates the notebook from the readable percent-cell Python source,
then executes all cells with a fresh kernel using the invoking Python interpreter.
The verification option repeats the entire run in another fresh kernel and checks
byte-for-byte equality of assignments, result CSVs/JSON, figures, generated documents
and manifest (not notebook timing metadata). Use the source and report helper for edits;
regeneration replaces Member 3's derived outputs.

The local execution environment reuses already-installed scientific/Jupyter packages
through an ignored `.venv` with system-site-packages; the commands above describe an
isolated setup for other group members. No virtual environment is committed.

## Dependencies and files

The user's existing `scikit-learn` requirement is preserved. `scipy` is now an explicit
direct dependency because the notebook calls its dendrogram function on sklearn's
fitted Ward hierarchy. SciPy was already installed (and is also a sklearn dependency).
`requirements-notebook.txt` lists only the tooling used to generate and execute notebooks.

`notebooks/modelling_helpers.py` contains the input contract, scoring, model factories,
metrics, stability and profiles. `notebooks/modelling_report.py` renders evidence docs
from computed results. `notebooks/verify_outputs.py` independently checks exported
assignments, metrics, profile counts, stability coverage and artifact hashes.

All experiment and profile data are under `results/`; useful plots are under `figures/`.
No trained-model pickle is needed for Member 4: validation joins outcomes to the frozen
historical assignments. These files do not define an inference service for new customers.

## Limits and group decisions

The future period begins at **2011-09-09 00:00:00** and has not been used for fitting,
PCA or candidate selection. Agree the exact three-month endpoint and qualify Monetary
as retained positive spend, not net revenue/profit. See the strategy for the upstream
AND/OR documentation mismatch and manual-line/cancellation qualifications.
The original submitted documents are absent from this checkout; the implementation
follows the supplied commitments and rubric, pending the group's final wording check.
