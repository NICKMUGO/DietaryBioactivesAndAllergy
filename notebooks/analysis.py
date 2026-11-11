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
