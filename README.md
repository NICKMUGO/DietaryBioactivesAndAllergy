# Dietary bioactives and the allergy transcriptome

Computing Bioinformatics group project.

**Research question.** Are the genes that are consistently dysregulated across human allergic tissues
over-represented among the curated targets of common dietary bioactive compounds, beyond what gene
study bias explains? And do these compounds shift their targets' expression in the direction opposite to disease?

**Short answer.** There is a robust, reproducible cross-tissue allergy signature (1,177 genes, 7 contrasts,
291 samples). After correcting for study bias, the curated targets of 44 dietary bioactives are **not** specially
enriched in it: only sulforaphane remains significant. Bioactives are no more enriched than ordinary CTD
chemicals, and they do not preferentially oppose the disease direction. Bioactive-target genes classify
allergic vs healthy samples well, but no better than random genes. See Stage 8 of the notebook.

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NICKMUGO/DietaryBioactivesAndAllergy/blob/main/notebooks/analysis.ipynb)

The first cell clones this repository and downloads the raw data (about 110 MB). The whole notebook runs top to
bottom in roughly 10–15 minutes on a standard Colab CPU runtime. Outputs are saved in the committed notebook.

## See the results in a UI

```bash
./run_app.sh            # or: streamlit run app.py
```

Then open http://localhost:8501. The app has a sidebar with one page per stage: an overview of the hypotheses, data and QC,
an interactive volcano plot, the consensus signature with gene search, bioactive enrichment, direction of effect,
Reactome pathways, the compound–gene network, machine learning and conclusions. If no results exist yet, `run_app.sh`
first downloads the data and runs the analysis.

## Repository layout

| Path | What it is |
|---|---|
| `notebooks/analysis.ipynb` | **The deliverable.** Stages 1–8 with code, outputs, figures and interpretation |
| `notebooks/analysis.py` | The same notebook as a plain Python script (percent-cell format), easier to diff and review |
| `app.py`, `run_app.sh` | Streamlit results explorer and its one-command launcher |
| `scripts/download_data.py` | Downloads every raw file from GEO, CTD, HGNC and Reactome into `data/raw/` |
| `scripts/build_notebook.py` | Rebuilds `analysis.ipynb` from `analysis.py` (`--execute` runs it and saves outputs) |
| `scripts/export_dashboard.py` | Turns the notebook's results into the flat tables used by the results dashboard |
| `results/` | DE tables per study, consensus meta-analysis, enrichment, reversal, Reactome and ML tables (CSV) |
| `results/figures/` | All notebook figures as PNG (QC, EDA, volcanoes, heatmaps, enrichment, network, ML) |
| `results/dashboard/` | Dashboard tables (CSV, plus a JSON bundle) and the dashboard page source |

## Data

| Source | Data | Role |
|---|---|---|
| NCBI GEO GSE19187 | Nasal epithelium, children: allergic rhinitis ± asthma vs healthy (HuGene 1.0 ST) | contrast |
| NCBI GEO GSE41861 | Nasal and bronchial brushings, asthma vs control (HG-U133 Plus 2) | 2 contrasts |
| NCBI GEO GSE36830 | Chronic rhinosinusitis with nasal polyps vs control (HG-U133 Plus 2) | contrast |
| NCBI GEO GSE136825 | CRSwNP polyps vs control turbinate, RNA-seq counts | contrast |
| NCBI GEO GSE32924 | Atopic dermatitis lesional vs healthy skin (HG-U133 Plus 2) | contrast |
| NCBI GEO GSE58640 | Eosinophilic oesophagitis vs control, RNA-seq FPKM | contrast |
| CTD | Curated chemical–gene interactions, chemicals, curated gene–disease links | bioactive targets, directions, validation |
| HGNC | Ensembl → gene symbol | RNA-seq annotation |
| Reactome | Pathway gene sets | enrichment |

## Methods in one paragraph

Per-study QC (scale check, expression filter, outlier and duplicate detection, sex-chromosome and tissue-marker checks)
→ limma-style moderated t-tests with BH-FDR → weighted Stouffer meta-analysis into a consensus signature →
enrichment of CTD targets of 44 dietary bioactives with a **degree-matched null** (study-bias correction) and a
comparison against all other CTD chemicals → directional "reversal" test against compound-specific expectations →
Reactome over-representation, CTD curated-disease validation and a NetworkX compound–gene network → logistic regression and
random forest classifiers against majority-class and random-gene-set baselines, within studies and leave-one-study-out.

## Reproduce locally

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn networkx nbformat nbconvert ipykernel requests
python scripts/download_data.py
python scripts/build_notebook.py --execute     # about 5 minutes on 12 cores
python scripts/export_dashboard.py
```
