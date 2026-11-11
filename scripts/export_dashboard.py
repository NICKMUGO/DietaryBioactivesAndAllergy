import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "results" / "dashboard_data.json").read_text())
OUT = ROOT / "results" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, df):
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:14s} {len(df):6d} rows")


studies = pd.DataFrame(D["datasets"]).rename(columns={"data type": "data_type", "genes (raw)": "genes_raw"})
de = pd.DataFrame(D["de_summary"])[["dataset", "genes tested", "up", "down"]].rename(columns={"genes tested": "genes_tested"})
studies = studies.merge(de, on="dataset")
studies["data_type"] = studies.data_type.map({"array": "Microarray", "rnaseq_counts": "RNA-seq (counts)",
                                              "rnaseq_fpkm": "RNA-seq (FPKM)"})
save("studies", studies)

qc = pd.DataFrame(D["qc"]).rename(columns={"genes before": "genes_before", "genes after": "genes_after",
                                           "samples before": "samples_before", "samples after": "samples_after",
                                           "outliers removed": "outliers_removed", "max value": "max_value"})
sex = pd.DataFrame(D["sex"]).set_index("dataset")
qc["sex_check"] = qc.dataset.map(lambda n: (
    "not testable: no sex signal (values look sex-adjusted)" if n in sex.index and sex.loc[n, "mismatches"] == "not testable"
    else (f"{sex.loc[n, 'mismatches']} mismatch ({sex.loc[n, 'mismatched samples']})" if n in sex.index and sex.loc[n, "mismatches"] not in (0, "0")
          else ("passed" if n in sex.index else "no sex recorded"))))
qc = qc.drop(columns=["duplicates"])
save("qc", qc)

pc = D["positive_controls"]
rows = [{"dataset": ds, "gene": g, "log2FC": pc["lfc"][i][j], "FDR": pc["fdr"][i][j]}
        for i, ds in enumerate(pc["datasets"]) for j, g in enumerate(pc["genes"]) if pc["lfc"][i][j] is not None]
posctrl = pd.DataFrame(rows)
save("posctrl", posctrl)

vol = pd.concat([pd.DataFrame({"dataset": k, "gene": v["g"], "log2FC": v["x"], "neglog10p": v["y"], "status": v["c"]})
                 for k, v in D["volcano"].items()])
save("volcano", vol)

c = D["consensus"]
hm = c["heatmap"]
sig_hm = pd.DataFrame([{"gene": g, "dataset": ds, "log2FC": hm["lfc"][i][j], "rank": i}
                       for i, g in enumerate(hm["genes"]) for j, ds in enumerate(hm["datasets"]) if hm["lfc"][i][j] is not None])
save("sig_heatmap", sig_hm)
sig = pd.DataFrame(c["table"])
sig["direction"] = sig.Z.map(lambda z: "up" if z > 0 else "down")
save("signature", sig)

e = D["enrichment"]
enr = pd.DataFrame(e["table"])
save("enrichment", enr)
chem = pd.concat([pd.DataFrame(e["scatter_other"], columns=["n_targets", "log2FE"]).assign(compound="", group="Other CTD chemical"),
                  pd.DataFrame(e["scatter_diet"], columns=["n_targets", "log2FE", "compound"]).assign(group="Dietary bioactive")])
save("chemicals", chem)
degbins = pd.DataFrame(e["deg_bins"]).rename(columns={"degree range": "degree_range", "signature genes": "signature_genes",
                                                       "signature %": "signature_pct"})
degbins.insert(0, "decile", range(1, len(degbins) + 1))
save("degbins", degbins)

r = D["reversal"]
save("reversal", pd.DataFrame(r["table"]))

rea = pd.concat([pd.DataFrame(v).assign(gene_set=k) for k, v in D["reactome"].items()])
rea = rea[["gene_set", "pathway", "id", "size", "overlap", "FE", "FDR", "genes"]]
save("reactome", rea)

val = pd.DataFrame(D["validation"]).rename(columns={"gene set": "gene_set", "curated genes in universe": "curated_genes",
                                                    "in signature": "in_signature", "FE naive": "FE_naive", "p naive": "p_naive",
                                                    "FE degree-matched": "FE_degmatched", "p degree-matched": "p_degmatched",
                                                    "example genes": "example_genes"})
save("validation", val)

n = D["network"]
save("net_nodes", pd.DataFrame(n["nodes"]))
save("net_edges", pd.DataFrame(n["edges"]).rename(columns={"s": "source", "t": "target", "e": "effect"}))
hubs = pd.DataFrame(n["hubs"])
for col in ["reverses", "reinforces", "none"]:
    hubs[col] = hubs[col].astype(int)
save("hubs", hubs)

m = D["ml"]
save("ml_within", pd.DataFrame(m["within"]))
rnd = pd.DataFrame([{"dataset": k.split("||")[0], "featureset": k.split("||")[1], "AUROC": a}
                    for k, vals in m["random"].items() for a in vals])
save("ml_random", rnd)
lodo = pd.DataFrame(m["lodo"]).rename(columns={
    "LR on bioactive hub targets": "lr_hub", "LR on random genes (mean)": "lr_random_mean", "LR on random genes (sd)": "lr_random_sd",
    "Signature score (LOO, no training)": "sig_score", "Bioactive-targeted signature score (LOO)": "bio_sig_score"})
save("ml_lodo", lodo)
save("stability", pd.DataFrame(m["stability"]).rename(columns={"signature genes": "signature_genes",
                                                               "Jaccard with full signature": "jaccard"}))

within = pd.DataFrame(m["within"])
hub = within[within.features == "Bioactive hub targets"]
sulf = enr.set_index("ctd_name").loc["sulforaphane"]
measured = posctrl.dropna(subset=["FDR"])
metrics = [
    ("n_contrasts", len(studies), "allergic-vs-healthy contrasts"),
    ("n_series", studies.GEO.nunique(), "GEO series"),
    ("n_tissues", studies.tissue.nunique(), "tissues"),
    ("n_samples", D["n_samples"], "samples analysed after QC"),
    ("n_bioactives", len(D["bioactives"]), "dietary bioactives"),
    ("posctrl_up", int(((measured.log2FC > 0) & (measured.FDR < 0.05)).sum()), "positive-control results significantly up"),
    ("posctrl_down", int(((measured.log2FC < 0) & (measured.FDR < 0.05)).sum()), "positive-control results significantly down"),
    ("posctrl_measured", len(measured), "positive-control results measured"),
    ("n_meta_genes", c["n_universe"], "genes in the meta-analysis"),
    ("n_sig", c["n_sig"], "consensus signature genes"),
    ("n_sig_up", c["n_up"], "signature genes up in allergy"),
    ("n_sig_down", c["n_down"], "signature genes down in allergy"),
    ("n_enrich_universe", e["n_universe"], "genes in the enrichment universe"),
    ("n_naive_sig", int((enr.FDR_naive < 0.05).sum()), "bioactives enriched, naive test (FDR<0.05)"),
    ("n_deg_sig", int((enr.FDR_degmatched < 0.05).sum()), "bioactives enriched, degree-matched (FDR<0.05)"),
    ("pct_all_chem_naive", round(100 * e["frac_naive_all"], 1), "% of all CTD chemicals 'enriched' by the naive test"),
    ("sulf_fe", round(float(sulf.FE_degmatched), 2), "sulforaphane degree-matched fold enrichment"),
    ("sulf_fdr", round(float(sulf.FDR_degmatched), 3), "sulforaphane degree-matched FDR"),
    ("mw_p", round(e["mw_p"], 3), "dietary vs other chemicals, Mann-Whitney p"),
    ("n_other_chem", e["n_other"], "size-matched non-dietary chemicals"),
    ("ols_coef", round(e["ols_coef"], 3), "regression: dietary effect on log2 FE"),
    ("ols_lo", round(e["ols_ci"][0], 3), "regression 95% CI low"),
    ("ols_hi", round(e["ols_ci"][1], 3), "regression 95% CI high"),
    ("rev_pct", round(100 * r["pooled"][0] / r["pooled"][1], 1), "% of bioactive effects opposing disease"),
    ("rev_expected_pct", round(100 * r["pooled"][2], 1), "% expected from direction biases"),
    ("rev_n_compounds", len(r["table"]), "bioactives with >=10 directional signature targets"),
    ("rev_n_sig", int((pd.DataFrame(r["table"]).FDR < 0.05).sum()), "bioactives with significant reversal"),
    ("rev_mw_p", round(r["mw_p"], 3), "reversal, dietary vs other chemicals, Mann-Whitney p"),
    ("n_hub", m["n_hub"], "bioactive hub target genes (>=4 compounds)"),
    ("hub_pct_random", round(float(hub.pct_beats_random.mean()), 0), "% of random gene sets beaten by hub targets"),
    ("lodo_hub", round(float(lodo.set_index("dataset").loc["mean", "lr_hub"]), 3), "LODO mean AUROC, hub targets"),
    ("lodo_random", round(float(lodo.set_index("dataset").loc["mean", "lr_random_mean"]), 3), "LODO mean AUROC, random genes"),
    ("lodo_p", round(m["lodo_p"], 2), "LODO hub vs random empirical p"),
    ("lodo_sig", round(float(lodo.set_index("dataset").loc["mean", "sig_score"]), 3), "LODO mean AUROC, untrained signature score"),
    ("n_net_genes", int((pd.DataFrame(n["nodes"]).kind == "gene").sum()), "hub genes drawn in the network"),
]
summary = pd.DataFrame(metrics, columns=["metric", "value", "label"])
save("summary", summary)
