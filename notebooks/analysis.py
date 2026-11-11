# %% [markdown]
# # Dietary bioactives and the shared transcriptomic signature of allergic disease
#
# **Computing Bioinformatics – Group Project**
#
# This notebook runs top-to-bottom in Google Colab (or locally). Every number,
# table and figure below is produced by the code in this notebook from public data.
#
# ## Stage 1 – Research question, objectives and hypotheses
#
# **Biological problem.** Allergic diseases (allergic rhinitis, asthma, chronic
# rhinosinusitis with nasal polyps, atopic dermatitis, eosinophilic oesophagitis)
# affect different organs but share type-2 inflammation and epithelial-barrier
# dysfunction. Many food-derived compounds (flavonoids such as quercetin, curcumin,
# resveratrol, omega-3 fatty acids, vitamin D, …) are marketed or studied as
# "anti-allergic". Whether their known molecular targets actually overlap the genes
# that are dysregulated in allergic tissue – and whether they push those genes in the
# *opposite* direction to disease – is rarely tested systematically or against a
# fair baseline.
#
# **Research question.** *Are the genes that are consistently dysregulated across human
# allergic tissues over-represented among the curated gene targets of common dietary
# bioactive compounds, beyond what is expected from how well-studied those genes are,
# and do these compounds tend to shift their targets' expression in the direction
# opposite to disease?*
#
# **Objectives**
#
# 1. **O1 – Allergy signature.** Derive differential-expression signatures (allergic vs
#    healthy, FDR-controlled) in 7 public transcriptomic comparisons covering nose,
#    bronchus, sinus, skin and oesophagus, and combine them into a consensus
#    cross-tissue allergy signature.
# 2. **O2 – Bioactive target enrichment and direction.** Test whether curated human
#    targets (CTD) of a panel of dietary bioactives are enriched in that signature,
#    using a degree-matched null and a comparison against all other CTD chemicals, and
#    test whether their reported effects oppose the disease direction ("reversal").
# 3. **O3 – Biological interpretation.** Interpret the signature and the
#    bioactive-reversible genes with Reactome pathway enrichment, CTD curated
#    gene–disease evidence and a compound–gene network (NetworkX).
# 4. **O4 – Prediction.** Test whether expression of bioactive-target genes classifies
#    allergic vs healthy samples, within and across datasets, against proper
#    baselines (majority class, random gene sets of equal size, all genes).
#
# **Hypotheses**
#
# * **H1:** the consensus allergy signature is enriched for dietary-bioactive targets
#   after correcting for gene "study bias" (degree-matched null; FDR < 0.05).
# * **H2:** dietary bioactives are *more* enriched than typical non-dietary CTD
#   chemicals (one-sided Mann–Whitney).
# * **H3:** for signature genes, bioactive-induced expression changes are more often
#   opposite to the disease direction than same-direction (> 50 %, and more than
#   non-dietary chemicals).
# * **H4:** bioactive-target genes classify disease status better than a majority
#   baseline and at least as well as random gene sets of the same size.
#
# Null hypotheses are the negations; we report every result whether or not it
# supports the hypothesis.

# %% [markdown]
# ### Setup
# In Colab this cell clones the repository and downloads the raw data (≈110 MB,
# 2–4 minutes). Locally it uses the repository you are in.

# %%
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/NICKMUGO/DietaryBioactivesAndAllergy.git"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    if not Path("DietaryBioactivesAndAllergy").exists():
        subprocess.run(["git", "clone", REPO_URL], check=True)
    os.chdir("DietaryBioactivesAndAllergy")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "networkx", "statsmodels"], check=True)

ROOT = Path.cwd()
if not (ROOT / "scripts" / "download_data.py").exists():
    ROOT = ROOT.parent
subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True)

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures"
for d in (PROC, RES, FIG):
    d.mkdir(parents=True, exist_ok=True)
print("project root:", ROOT)

# %%
import gzip, io, json, re, zipfile, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats, special
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 42
rng = np.random.default_rng(SEED)

C_CTRL, C_DIS, C_AQUA, C_YEL, C_VIOLET = "#2a78d6", "#e34948", "#1baf7a", "#eda100", "#6250d6"
C_ORANGE, C_MUTED, C_GRID, C_INK = "#eb6834", "#898781", "#e1e0d9", "#0b0b0b"
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bluegrayred", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32a2a"])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
    "legend.frameon": False,
})

def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.show()

DASH = {}

# %% [markdown]
# ## Stage 2 – Data acquisition
#
# All data are public and downloaded by `scripts/download_data.py` (source URLs are
# listed there). Nothing is edited by hand.
#
# | File | Source | What it is | Used for |
# |---|---|---|---|
# | `GSE19187_series_matrix.txt.gz` | NCBI GEO | Affymetrix HuGene 1.0 ST (GPL6244), RMA log2; nasal epithelium of 38 children: healthy, allergic rhinitis, rhinitis + controlled / uncontrolled asthma | contrast **AR/asthma vs healthy (nasal, child)** |
# | `GSE41861_series_matrix.txt.gz` | NCBI GEO | Affymetrix HG-U133 Plus 2 (GPL570), log2; nasal (57) and bronchial (81) brushings, asthma vs control adults | contrasts **asthma vs control (nasal)** and **(bronchial)** |
# | `GSE36830_series_matrix.txt.gz` | NCBI GEO | GPL570, log2; uncinate tissue of controls and nasal polyps of CRSwNP patients | contrast **CRSwNP polyp vs control (array)** |
# | `GSE136825_genecounts.txt.gz` + series matrix | NCBI GEO | RNA-seq (HiSeq 4000) raw featureCounts, GENCODE v26 Ensembl IDs; 42 polyps, 33 polyp-patient turbinates, 28 control turbinates | contrast **CRSwNP polyp vs control (RNA-seq)** |
# | `GSE32924_series_matrix.txt.gz` | NCBI GEO | GPL570, log2; skin of atopic dermatitis (lesional, non-lesional) and healthy volunteers | contrast **AD lesional vs normal skin** |
# | `GSE58640_RNAseq_processed.txt.gz` + series matrix | NCBI GEO | RNA-seq FPKM (gene symbols); oesophageal biopsies, active EoE vs healthy | contrast **EoE vs control** |
# | `GPL570.annot.gz`, `GPL6244.annot.gz` | NCBI GEO | platform annotation: probe → gene symbol / Entrez ID | probe-to-gene mapping |
# | `hgnc_complete_set.txt` | HGNC | approved symbols with Ensembl / Entrez IDs, locus group | Ensembl → symbol for GSE136825 |
# | `CTD_chem_gene_ixns.tsv.gz` | Comparative Toxicogenomics Database | curated chemical–gene interactions with action types (e.g. `decreases^expression`) and PubMed IDs | bioactive target sets and directions |
# | `CTD_chemicals.tsv.gz` | CTD | chemical vocabulary (MeSH IDs, synonyms) | chemical annotation |
# | `CTD_curated_genes_diseases.tsv.gz` | CTD | curated gene–disease associations (marker/mechanism, therapeutic) | external validation of the signature |
# | `ReactomePathways.gmt.zip`, `ReactomePathwaysRelation.txt` | Reactome | pathway → gene sets; pathway hierarchy | pathway enrichment |

# %%
def read_series_matrix(path):
    meta_rows, table, in_table = defaultdict(list), [], False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True; continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table.append(line.rstrip("\n").split("\t"))
            elif line.startswith("!Sample_"):
                key, *vals = line.rstrip("\n").split("\t")
                meta_rows[key].append([v.strip('"') for v in vals])
    meta = {}
    for key, rows in meta_rows.items():
        if key == "!Sample_characteristics_ch1":
            for row in rows:
                for i, v in enumerate(row):
                    if ":" in v:
                        k, val = v.split(":", 1)
                        meta.setdefault(k.strip().lower(), [None] * len(row))[i] = val.strip()
        else:
            meta[key.replace("!Sample_", "")] = rows[0]
    meta = pd.DataFrame(meta)
    meta.index = meta["geo_accession"]
    expr = None
    if len(table) > 1:
        header = [h.strip('"') for h in table[0]]
        expr = pd.DataFrame([r[1:] for r in table[1:]], index=[r[0].strip('"') for r in table[1:]],
                            columns=header[1:])
        expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr, meta


def read_platform_annot(path):
    with gzip.open(path, "rt", errors="replace") as fh:
        lines = fh.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("!platform_table_begin")) + 1
    end = next(i for i, l in enumerate(lines) if l.startswith("!platform_table_end"))
    ann = pd.read_csv(io.StringIO("".join(lines[start:end])), sep="\t", dtype=str)
    return ann[["ID", "Gene symbol"]].rename(columns={"Gene symbol": "symbol"})

raw = {}
for gse in ["GSE19187", "GSE41861", "GSE36830", "GSE32924", "GSE136825", "GSE58640"]:
    raw[gse] = read_series_matrix(RAW / "geo" / f"{gse}_series_matrix.txt.gz")
    e, m = raw[gse]
    print(f"{gse}: samples={len(m):4d}  expression rows in matrix={0 if e is None else len(e):6d}")

annot = {g: read_platform_annot(RAW / "geo" / f"{g}.annot.gz") for g in ["GPL570", "GPL6244"]}
hgnc = pd.read_csv(RAW / "hgnc" / "hgnc_complete_set.txt", sep="\t", dtype=str,
                   usecols=["symbol", "locus_group", "entrez_id", "ensembl_gene_id"])
gse136825_counts = pd.read_csv(RAW / "geo" / "GSE136825_genecounts.txt.gz", sep="\t")
gse58640_fpkm = pd.read_csv(RAW / "geo" / "GSE58640_RNAseq_processed.txt.gz", sep="\t")
print("GSE136825 counts:", gse136825_counts.shape, " GSE58640 FPKM:", gse58640_fpkm.shape)
print("HGNC genes:", len(hgnc))

# %% [markdown]
# **Variables used from the metadata**
#
# * GSE19187 – `disease state` (Healthy / Rhinitis / Rhinitis with controlled asthma / Uncontrolled asthma), `gender`, `age`.
# * GSE41861 – `disease` (Asthma / Control), `severity`, `tissue` (Upper airway (Nasal) / Lower airway (bronchial)), `sex`, `age`.
# * GSE36830 – `title` (Control UT / CRSsNP UT / CRSwNP UT / CRSwNP polyp).
# * GSE136825 – `source_name_ch1` (Nasal Polyp Tissue / Nasal Polyp Inferior Turbinate / Control Inferior Turbinate); `description` links the sample to the count-matrix column (`RHPxxxx.bam`).
# * GSE32924 – `condition` (AL = lesional, ANL = non-lesional, Normal), `individual`, `eczema severity` (SCORAD).
# * GSE58640 – `patient cohort` (healthy control / active EoE); `title` matches the FPKM column names.
#
# Expression values: microarray series matrices hold log2 RMA intensities per probe; GSE136825 holds
# raw integer read counts per Ensembl gene; GSE58640 holds FPKM per gene symbol.
#
# **Design choices that matter.** For each study we keep only independent individuals in the two
# groups being compared (e.g. AD *lesional vs normal* rather than lesional vs non-lesional of the same
# patient; CRSwNP *polyp vs control turbinate* rather than polyp vs the same patient's turbinate).
# CRS without polyps (not type-2 dominated) is excluded. GSE41861 nasal and bronchial samples come
# largely from the same people, so those two contrasts are not fully independent – we note this
# where it matters.

# %%
def collapse_probes(expr, ann):
    a = ann.dropna(subset=["symbol"])
    a = a[~a.symbol.str.contains("///")].set_index("ID")["symbol"]
    e = expr.loc[expr.index.intersection(a.index)].copy()
    e["__sym"] = a.loc[e.index].values
    e["__mean"] = e.drop(columns="__sym").mean(axis=1)
    e = e.sort_values("__mean", ascending=False).drop_duplicates("__sym")
    return e.set_index("__sym").drop(columns="__mean")

datasets = {}

def add_dataset(name, expr, meta, label, tissue, disease, gse, kind):
    keep = label.dropna().index
    datasets[name] = dict(expr_raw=expr[keep], meta=meta.loc[keep], label=label.loc[keep].astype(int),
                          tissue=tissue, disease=disease, gse=gse, kind=kind)

e, m = raw["GSE19187"]
lab = m["disease state"].map(lambda s: 0 if s == "Healthy" else 1)
add_dataset("AR/asthma nasal (child)", collapse_probes(e, annot["GPL6244"]), m, lab,
            "nasal epithelium", "allergic rhinitis ± asthma", "GSE19187", "array")

e, m = raw["GSE41861"]
ge = collapse_probes(e, annot["GPL570"])
lab = m["disease"].map({"Asthma": 1, "Control": 0})
for tis, nm in [("Upper airway (Nasal)", "Asthma nasal"), ("Lower airway (bronchial)", "Asthma bronchial")]:
    idx = m.index[m["tissue"] == tis]
    add_dataset(nm, ge[idx], m.loc[idx], lab.loc[idx],
                "nasal epithelium" if "Nasal" in tis else "bronchial epithelium", "asthma", "GSE41861", "array")

e, m = raw["GSE36830"]
lab = m["title"].map(lambda t: 0 if t.startswith("Control") else (1 if "polyp" in t else np.nan))
add_dataset("CRSwNP polyp (array)", collapse_probes(e, annot["GPL570"]), m, lab,
            "sinonasal tissue", "chronic rhinosinusitis with nasal polyps", "GSE36830", "array")

e, m = raw["GSE32924"]
lab = m["condition"].map({"AL": 1, "Normal": 0})
add_dataset("Atopic dermatitis skin", collapse_probes(e, annot["GPL570"]), m, lab,
            "skin", "atopic dermatitis", "GSE32924", "array")

_, m = raw["GSE136825"]
cnt = gse136825_counts.copy()
cnt["ens"] = cnt["Geneid"].str.split(".").str[0]
ens2sym = (hgnc.dropna(subset=["ensembl_gene_id"]).drop_duplicates("ensembl_gene_id")
           .set_index("ensembl_gene_id")["symbol"])
cnt = cnt[cnt.ens.isin(ens2sym.index)]
cnt["sym"] = ens2sym.loc[cnt.ens].values
cnt = cnt.drop(columns=["Geneid", "Chr", "Start", "End", "Strand", "Length", "ens"]).groupby("sym").sum()
col2gsm = {f"{d}.bam": g for g, d in zip(m.index, m["description"])}
cnt = cnt.rename(columns=col2gsm)[[c for c in m.index if c in col2gsm.values()]]
lab = m["source_name_ch1"].map({"Nasal Polyp Tissue": 1, "Control Inferior Turbinate": 0})
add_dataset("CRSwNP polyp (RNA-seq)", cnt, m, lab, "sinonasal tissue",
            "chronic rhinosinusitis with nasal polyps", "GSE136825", "rnaseq_counts")

_, m = raw["GSE58640"]
fp = gse58640_fpkm.drop(columns="locus").groupby("gene_id").sum()
t2gsm = dict(zip(m["title"], m.index))
fp = fp.rename(columns=t2gsm)[list(m.index)]
lab = m["patient cohort"].map({"active EoE": 1, "healthy control": 0})
add_dataset("Eosinophilic oesophagitis", fp, m, lab, "oesophagus", "eosinophilic oesophagitis",
            "GSE58640", "rnaseq_fpkm")

summary = pd.DataFrame([{ "dataset": k, "GEO": v["gse"], "tissue": v["tissue"], "data type": v["kind"],
    "genes (raw)": v["expr_raw"].shape[0], "allergic": int(v["label"].sum()),
    "healthy": int((v["label"] == 0).sum())} for k, v in datasets.items()])
summary

# %% [markdown]
# **Interpretation.** We have 7 allergic-vs-healthy contrasts from 6 GEO series (3 platforms:
# two Affymetrix arrays and RNA-seq), 5 tissues and 5 allergic conditions; 295 samples enter QC and 291 are analysed
# (four outliers removed in Stage 3). Group sizes are unbalanced and small in two studies (CRSwNP array 6 vs 6; EoE 10 vs 6),
# which limits power there and is the reason we combine studies rather than rely on any single one.

# %% [markdown]
# ### CTD: dietary bioactive panel
#
# The panel is defined *a priori* (before looking at any expression result): food-derived
# compounds (polyphenols, carotenoids, vitamins, fatty acids, isothiocyanates, alkaloids/spice
# compounds, methylxanthines, a short-chain fatty acid) with ≥ 40 curated human gene interactions
# in CTD, so that each has enough targets to test. Drugs and non-food herbal toxins (e.g.
# tretinoin, triptolide, celastrol) are deliberately excluded.

# %%
CTD_COLS = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
            "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
ctd = pd.read_csv(RAW / "ctd" / "CTD_chem_gene_ixns.tsv.gz", sep="\t", comment="#", header=None,
                  names=CTD_COLS, usecols=["ChemicalName", "ChemicalID", "GeneSymbol", "GeneForms",
                                           "OrganismID", "InteractionActions", "PubMedIDs"], dtype=str)
n_all = len(ctd)
ctd = ctd[ctd.OrganismID == "9606"].dropna(subset=["GeneSymbol"]).drop_duplicates()
print(f"CTD interactions: {n_all:,} total -> {len(ctd):,} human (after removing duplicates)")
print(f"human chemicals: {ctd.ChemicalName.nunique():,}; human genes: {ctd.GeneSymbol.nunique():,}")

BIOACTIVES = {
    "Quercetin": ("Quercetin", "flavonoid"), "kaempferol": ("Kaempferol", "flavonoid"),
    "Luteolin": ("Luteolin", "flavonoid"), "Apigenin": ("Apigenin", "flavonoid"),
    "myricetin": ("Myricetin", "flavonoid"), "fisetin": ("Fisetin", "flavonoid"),
    "chrysin": ("Chrysin", "flavonoid"), "naringenin": ("Naringenin", "flavonoid"),
    "naringin": ("Naringin", "flavonoid"), "hesperetin": ("Hesperetin", "flavonoid"),
    "nobiletin": ("Nobiletin", "flavonoid"), "Rutin": ("Rutin", "flavonoid"),
    "baicalein": ("Baicalein", "flavonoid"), "Catechin": ("Catechin", "flavonoid"),
    "epigallocatechin gallate": ("EGCG (green tea)", "flavonoid"),
    "Genistein": ("Genistein (soy)", "isoflavone"), "daidzein": ("Daidzein (soy)", "isoflavone"),
    "Resveratrol": ("Resveratrol", "stilbene"), "pterostilbene": ("Pterostilbene", "stilbene"),
    "Curcumin": ("Curcumin (turmeric)", "curcuminoid"),
    "Ellagic Acid": ("Ellagic acid", "phenolic acid"), "Gallic Acid": ("Gallic acid", "phenolic acid"),
    "Chlorogenic Acid": ("Chlorogenic acid (coffee)", "phenolic acid"),
    "caffeic acid": ("Caffeic acid", "phenolic acid"),
    "sulforaphane": ("Sulforaphane (broccoli)", "isothiocyanate"),
    "indole-3-carbinol": ("Indole-3-carbinol", "indole"), "3,3'-diindolylmethane": ("DIM", "indole"),
    "Capsaicin": ("Capsaicin (chili)", "spice compound"), "gingerol": ("Gingerol (ginger)", "spice compound"),
    "piperine": ("Piperine (pepper)", "spice compound"), "cinnamaldehyde": ("Cinnamaldehyde", "spice compound"),
    "thymoquinone": ("Thymoquinone (black seed)", "spice compound"), "Eugenol": ("Eugenol (clove)", "spice compound"),
    "Caffeine": ("Caffeine", "methylxanthine"),
    "Lycopene": ("Lycopene (tomato)", "carotenoid"), "beta Carotene": ("β-Carotene", "carotenoid"),
    "Cholecalciferol": ("Vitamin D3", "vitamin"), "Ascorbic Acid": ("Vitamin C", "vitamin"),
    "alpha-Tocopherol": ("α-Tocopherol (vit. E)", "vitamin"),
    "Docosahexaenoic Acids": ("DHA (omega-3)", "fatty acid"),
    "Eicosapentaenoic Acid": ("EPA (omega-3)", "fatty acid"), "Linoleic Acid": ("Linoleic acid", "fatty acid"),
    "Oleic Acid": ("Oleic acid", "fatty acid"), "Butyric Acid": ("Butyrate", "short-chain fatty acid"),
}
present = [c for c in BIOACTIVES if c in set(ctd.ChemicalName)]
assert len(present) == len(BIOACTIVES), set(BIOACTIVES) - set(present)
bio_targets = ctd[ctd.ChemicalName.isin(BIOACTIVES)].groupby("ChemicalName").GeneSymbol.nunique()
print(f"{len(BIOACTIVES)} bioactives; human target genes per compound: "
      f"median {bio_targets.median():.0f}, range {bio_targets.min()}–{bio_targets.max()}")

# %% [markdown]
# ## Stage 3 – Pre-processing and quality control
#
# Steps, per dataset:
#
# 1. **Scale check** – microarray matrices must be log2 (max ≲ 16); RNA-seq is transformed by us.
# 2. **RNA-seq normalisation** – GSE136825: keep genes with ≥ 1 count-per-million in at least as many
#    samples as the smaller group, DESeq2-style median-of-ratios size factors, log2(CPM + 1).
#    GSE58640: log2(FPKM + 1), keep genes with FPKM ≥ 1 in ≥ 6 samples.
# 3. **Low-expression filter** for arrays – drop genes whose 75th-percentile intensity is in the
#    bottom 20 % of the dataset (mostly unexpressed probes).
# 4. **Duplicate and outlier samples** – Pearson correlation between samples; duplicates (r > 0.999)
#    and outliers (median correlation below Q1 − 3·IQR) are flagged and removed.
# 5. **Biology that must hold** – (a) reported sex must agree with *XIST* (female-only) and the
#    Y-chromosome genes *RPS4Y1/DDX3Y*; (b) tissue identity markers must be high in the expected
#    tissue (keratins *KRT1/KRT10* in skin, *KRT13/KRT4* in oesophagus, ciliary *FOXJ1* in airway);
#    (c) known positive controls for type-2 allergic inflammation (e.g. *POSTN*, *CLCA1*,
#    *SERPINB2*, *CCL26*, *CST1*) should go up in disease – checked in Stage 5.

# %%
def median_ratio_size_factors(counts):
    lg = np.log(counts.replace(0, np.nan))
    geo_mean = lg.mean(axis=1)
    ok = geo_mean.notna() & np.isfinite(geo_mean)
    return np.exp((lg[ok].sub(geo_mean[ok], axis=0)).median(axis=0))

qc_log = []
for name, d in datasets.items():
    x = d["expr_raw"].copy()
    n0 = x.shape[0]
    step = {"dataset": name, "genes before": n0, "samples before": x.shape[1]}
    if d["kind"] == "array":
        step["max value"] = round(float(np.nanmax(x.values)), 1)
        x = x.dropna()
        if step["max value"] > 20:
            x = np.log2(x.clip(lower=np.nanpercentile(x.values[x.values > 0], 0.1)))
            step["transform"] = "log2 applied (was linear)"
        else:
            step["transform"] = "already log2"
        q75 = x.quantile(0.75, axis=1)
        x = x[q75 > q75.quantile(0.20)]
    elif d["kind"] == "rnaseq_counts":
        libsize = x.sum(axis=0)
        step["max value"] = int(x.values.max())
        cpm = x / libsize * 1e6
        min_n = int(min(d["label"].sum(), (d["label"] == 0).sum()))
        x = x[(cpm >= 1).sum(axis=1) >= min_n]
        sf = median_ratio_size_factors(x)
        d["size_factors"] = sf
        d["libsize"] = libsize
        x = np.log2(x / (sf * libsize.median()) * 1e6 + 1)
        step["transform"] = "median-of-ratios, log2(CPM+1)"
    else:
        step["max value"] = round(float(x.values.max()), 1)
        x = x[(x >= 1).sum(axis=1) >= 6]
        x = np.log2(x + 1)
        step["transform"] = "log2(FPKM+1)"
    corr = np.corrcoef(x.values.T)
    np.fill_diagonal(corr, np.nan)
    dup = [x.columns[j] for i, j in zip(*np.where(np.triu(corr, 1) > 0.999))]
    medc = np.nanmedian(corr, axis=1)
    q1, q3 = np.percentile(medc, [25, 75])
    outl = list(x.columns[medc < q1 - 3 * (q3 - q1)])
    drop = sorted(set(dup + outl))
    x = x.drop(columns=drop)
    d["expr"] = x
    d["label"] = d["label"].drop(drop)
    d["meta"] = d["meta"].drop(drop)
    d["sample_median_corr"] = pd.Series(medc, index=d["expr_raw"].columns)
    step.update({"genes after": x.shape[0], "samples after": x.shape[1],
                 "duplicates": ", ".join(dup) or "–", "outliers removed": ", ".join(outl) or "–"})
    qc_log.append(step)

qc_table = pd.DataFrame(qc_log)
qc_table

# %% [markdown]
# ### QC check: reported sex vs sex-chromosome gene expression
# *XIST* is expressed only from the inactive X in females; *RPS4Y1* and *DDX3Y* are on the
# Y chromosome and expressed only in males. A sample whose expression contradicts its recorded sex is a
# likely sample swap or annotation error.

# %%
def infer_sex(x):
    y = x.loc[[g for g in ["RPS4Y1", "DDX3Y", "KDM5D", "EIF1AY"] if g in x.index]].mean()
    score = y - x.loc["XIST"] if "XIST" in x.index else y
    srt = np.sort(score.values)
    cut = srt[np.argmax(np.diff(srt))] + np.max(np.diff(srt)) / 2
    return pd.Series(np.where(score > cut, "M", "F"), index=x.columns), y

sex_rows = []
fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
for ax, name in zip(axes, ["AR/asthma nasal (child)", "Asthma nasal", "Asthma bronchial",
                           "CRSwNP polyp (RNA-seq)"]):
    d = datasets[name]; x = d["expr"]
    pred, ys = infer_sex(x)
    xs = x.loc["XIST"] if "XIST" in x.index else pd.Series(np.arange(x.shape[1]), index=x.columns)
    sexcol = next((c for c in ["gender", "sex"] if c in d["meta"].columns), None)
    rep = d["meta"][sexcol].str[0].str.upper() if sexcol else pd.Series("?", index=x.columns)
    has_rep = rep.isin(["M", "F"])
    gap = ys[rep == "M"].mean() - ys[rep == "F"].mean() if has_rep.any() else np.nan
    sex_signal = not (has_rep.any() and gap < 1)
    mismatch = [s for s in x.columns if sex_signal and rep[s] in ("M", "F") and rep[s] != pred[s]]
    for sx, colr, lab in [("F", C_VIOLET, "reported female"), ("M", C_AQUA, "reported male"),
                          ("?", C_MUTED, "not reported")]:
        sel = (rep == sx).values
        if sel.any():
            ax.scatter(xs[sel], ys[sel], s=22, c=colr, label=lab, edgecolor="white", lw=0.6)
    if mismatch:
        ax.scatter(xs[mismatch], ys[mismatch], s=80, facecolor="none", edgecolor=C_DIS, lw=1.5,
                   label="mismatch")
    ax.set(title=name + ("" if sex_signal else "\nNO sex signal: values appear sex-adjusted"),
           xlabel="XIST (log2)" if "XIST" in x.index else "sample (XIST not on array)",
           ylabel="Y genes RPS4Y1/DDX3Y/KDM5D (log2)")
    ax.legend(fontsize=7, loc="best")
    sex_rows.append({"dataset": name, "samples with reported sex": int((rep != "?").sum()),
                     "Y-gene gap M−F (log2)": None if np.isnan(gap) else round(float(gap), 2),
                     "inferred F": int((pred == "F").sum()) if sex_signal else None,
                     "inferred M": int((pred == "M").sum()) if sex_signal else None,
                     "mismatches": len(mismatch) if sex_signal else "not testable",
                     "mismatched samples": ", ".join(mismatch) or "–"})
fig.suptitle("Sex check: expected two clean clusters (female: high XIST, no Y-gene expression)", y=1.08)
savefig(fig, "qc_sex_check")
sex_table = pd.DataFrame(sex_rows)
sex_table

# %% [markdown]
# ### QC check: tissue identity markers

# %%
markers = ["KRT1", "KRT10", "LOR", "KRT13", "KRT4", "FOXJ1", "SCGB1A1", "MUC5AC", "BPIFA1", "PTPRC"]
tissue_mat = pd.DataFrame({name: d["expr"].median(axis=1).rank(pct=True).reindex(markers) * 100
                           for name, d in datasets.items()}).T
fig, ax = plt.subplots(figsize=(8, 3.2))
seq = matplotlib.colors.LinearSegmentedColormap.from_list("blueseq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
sns.heatmap(tissue_mat, cmap=seq, vmin=0, vmax=100, annot=tissue_mat.round(0).fillna(-1).astype(int).astype(str)
            .replace("-1", "filtered"), fmt="", annot_kws={"size": 7}, mask=tissue_mat.isna(),
            cbar_kws={"label": "expression percentile within dataset"}, ax=ax, linewidths=1, linecolor="white")
ax.set_facecolor("#f9f9f7")
ax.set(title="Tissue-identity markers (blank = below expression filter, i.e. not expressed)", xlabel="", ylabel="")
savefig(fig, "qc_tissue_markers")

# %% [markdown]
# **QC interpretation.**
#
# * **Suspicious finding:** four of the five microarray matrices are on a log2 scale (max ≈ 14–16), but
#   **GSE36830 is not** (median ≈ 1, max ≈ 328: linear, median-scaled MAS5-type values). Analysing it as
#   if it were log2 would have produced meaningless fold changes, so it was log2-transformed (values
#   floored at their 0.1th percentile). GSE41861 contains missing values for some probes; those genes
#   were dropped. RNA-seq data were normalised as described. The expression filter removes unexpressed
#   probes/genes (before/after counts in the QC table).
# * No technical duplicate samples were found; the outlier rule flags any sample with an unusually low
#   correlation to the rest of its study (listed in the table – removed).
# * **Sex check (two findings).** (i) In GSE19187 and GSE136825 samples split into two clean clusters
#   (high *XIST*/no Y-gene signal vs the reverse), as biology requires. One GSE19187 sample recorded as
#   female clearly expresses Y-chromosome genes (listed in the table) – a probable metadata error. We keep
#   it because sex is not our variable of interest, but it is a warning that GEO annotations are not
#   infallible. (ii) **In GSE41861 (both tissues) there is no sex signal at all**: *XIST* and
#   *RPS4Y1/DDX3Y/KDM5D* have the same mean in reported males and females (gap ≈ 0 log2 units, versus
#   ≈ 4 in GSE19187). That is biologically impossible for raw data, so the deposited values were almost
#   certainly adjusted for sex (and possibly other covariates) before submission. The consequence is that
#   sex cannot be checked there, and we must verify that the *disease* signal survives – which the
#   positive-control genes in Stage 5 do (*POSTN*, *CLCA1*, *SERPINB2* are up in asthmatic bronchi).
# * **Tissue markers** behave as biology requires: epidermal keratins (*KRT1/KRT10*, *LOR*) are high only in
#   skin; squamous oesophageal keratins (*KRT13/KRT4*) dominate the oesophagus; airway epithelial and
#   ciliated-cell genes (*SCGB1A1*, *BPIFA1*, *FOXJ1*, *MUC5AC*) are high in nasal and bronchial brushings;
#   polyp/turbinate tissue shows immune infiltrate (*PTPRC*). The data are what they claim to be.

# %% [markdown]
# ## Stage 4 – Exploratory data analysis

# %%
fig, ax = plt.subplots(figsize=(7.5, 3))
s = summary.set_index("dataset")[["healthy", "allergic"]].copy()
s["healthy"] = [int((datasets[k]["label"] == 0).sum()) for k in s.index]
s["allergic"] = [int(datasets[k]["label"].sum()) for k in s.index]
y = np.arange(len(s))
ax.barh(y - 0.2, s["healthy"], 0.38, color=C_CTRL, label="healthy")
ax.barh(y + 0.2, s["allergic"], 0.38, color=C_DIS, label="allergic")
for yi, (h, a) in enumerate(zip(s["healthy"], s["allergic"])):
    ax.text(h + 0.5, yi - 0.2, str(h), va="center", fontsize=7, color="#52514e")
    ax.text(a + 0.5, yi + 0.2, str(a), va="center", fontsize=7, color="#52514e")
ax.set_yticks(y, s.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set(xlabel="samples after QC", title="Samples per group, per contrast"); ax.legend(loc="lower right")
savefig(fig, "eda_sample_counts")

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, (name, d) in zip(axes.flat, datasets.items()):
    x = d["expr"]; labl = d["label"].loc[x.columns]
    order = labl.sort_values().index
    bp = ax.boxplot(x[order].values, showfliers=False, patch_artist=True, widths=0.7,
                    medianprops=dict(color=C_INK, lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0))
    for patch, l in zip(bp["boxes"], labl.loc[order]):
        patch.set(facecolor=C_DIS if l else C_CTRL, edgecolor="white", lw=0.4)
    ax.set(title=name, xticks=[], ylabel="log2 expression"); ax.grid(axis="x", visible=False)
ax = axes.flat[-1]
if "libsize" in datasets["CRSwNP polyp (RNA-seq)"]:
    d = datasets["CRSwNP polyp (RNA-seq)"]
    ax.scatter(d["libsize"] / 1e6, d["size_factors"].reindex(d["libsize"].index), s=14, c=C_VIOLET, edgecolor="white", lw=0.5)
    ax.set(title="GSE136825: library size vs size factor", xlabel="reads (millions)", ylabel="median-of-ratios size factor")
fig.suptitle("Per-sample expression distributions after normalisation (blue = healthy, red = allergic)", y=1.01)
fig.tight_layout()
savefig(fig, "eda_distributions")

# %% [markdown]
# **Interpretation.** Within each study, per-sample distributions are well aligned after normalisation,
# with no group-wise shift that would masquerade as differential expression. For the RNA-seq study,
# size factors track library depth, as expected – depth differences are removed by normalisation.

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
pca_store = {}
for ax, (name, d) in zip(axes.flat, datasets.items()):
    x = d["expr"]
    top = x.var(axis=1).sort_values(ascending=False).index[:2000]
    z = x.loc[top].sub(x.loc[top].mean(axis=1), axis=0).T
    pca = PCA(n_components=2, random_state=SEED).fit(z)
    pcs = pca.transform(z)
    labl = d["label"].loc[x.columns].values
    for v, colr, lab in [(0, C_CTRL, "healthy"), (1, C_DIS, "allergic")]:
        ax.scatter(pcs[labl == v, 0], pcs[labl == v, 1], s=22, c=colr, label=lab, edgecolor="white", lw=0.6)
    r = pca.explained_variance_ratio_
    rpc = [abs(stats.pointbiserialr(labl, pcs[:, i])[0]) for i in range(2)]
    ax.set(title=f"{name}\n|r(PC1,disease)|={rpc[0]:.2f}  |r(PC2,disease)|={rpc[1]:.2f}",
           xlabel=f"PC1 ({r[0]:.0%})", ylabel=f"PC2 ({r[1]:.0%})")
    pca_store[name] = dict(pc1=pcs[:, 0].round(2).tolist(), pc2=pcs[:, 1].round(2).tolist(),
                           label=labl.tolist(), var=[round(float(v), 3) for v in r],
                           samples=list(x.columns), r=[round(float(v), 2) for v in rpc])
axes.flat[0].legend(fontsize=7)
axes.flat[-1].axis("off")
fig.suptitle("PCA of the 2,000 most variable genes in each study", y=1.01)
fig.tight_layout()
savefig(fig, "eda_pca")
DASH["pca"] = pca_store

# %% [markdown]
# **Interpretation.** Disease status is the dominant source of variation in tissue-level studies with
# strong inflammation (atopic dermatitis lesional skin, nasal polyps, EoE) – PC1 separates the groups
# almost completely. In the epithelial-brushing studies (asthma, childhood rhinitis) disease explains
# much less of the total variance (|r| with PC1/PC2 is low): inter-individual variation dominates and the
# disease signal is subtler. This tells us to expect many more DEGs in the tissue studies and to weight
# evidence across studies rather than by DEG counts.

# %%
gene_degree = ctd.groupby("GeneSymbol").ChemicalName.nunique()
chem_size = ctd.groupby("ChemicalName").GeneSymbol.nunique()

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})
ax = axes[0]
ax.hist(np.log10(gene_degree), bins=50, color=C_CTRL, edgecolor="white", lw=0.4)
ax.set(xlabel="log10(number of chemicals per gene)", ylabel="genes",
       title=f"CTD gene degree is highly skewed\nmedian {gene_degree.median():.0f}, max {gene_degree.max():,} ({gene_degree.idxmax()})")
ax = axes[1]
ax.hist(np.log10(chem_size), bins=50, color=C_MUTED, edgecolor="white", lw=0.4, label="all CTD chemicals")
for c in BIOACTIVES:
    ax.axvline(np.log10(chem_size[c]), color=C_AQUA, lw=0.7, alpha=0.8)
ax.plot([], [], color=C_AQUA, label="dietary bioactives")
ax.set(xlabel="log10(human target genes per chemical)", ylabel="chemicals", title="Target-set size")
ax.legend(fontsize=7)
ax = axes[2]
bs = bio_targets.rename(index={k: v[0] for k, v in BIOACTIVES.items()}).sort_values()
ax.barh(bs.index, bs.values, color=C_AQUA, height=0.7)
ax.set_xscale("log"); ax.tick_params(axis="y", labelsize=6); ax.grid(axis="y", visible=False)
ax.set(xlabel="human target genes in CTD (log scale)", title="Dietary bioactive panel")
fig.tight_layout()
savefig(fig, "eda_ctd")

# %% [markdown]
# **Interpretation.** CTD knowledge is extremely uneven: a few genes (e.g. *CYP1A1*, *TNF*, *IL6*, *HMOX1*,
# *CASP3*) have been studied with thousands of chemicals, while most genes have a handful. Inflammatory
# genes are both the most studied *and* the ones we expect to change in allergy, so a naive
# hypergeometric test would find "enrichment" for almost any chemical. **This is the key confounder the
# core analysis must correct for** (degree-matched null). Bioactive target-set sizes span two orders of
# magnitude (resveratrol and quercetin ≈ thousands; hesperetin ≈ 60), so we use size-aware statistics.
