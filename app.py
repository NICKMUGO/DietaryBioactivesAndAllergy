from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DASH = ROOT / "results" / "dashboard"
FIG = ROOT / "results" / "figures"

UP, DOWN, GOOD, WARN, MUTED = "#e34948", "#2a78d6", "#1baf7a", "#eb6834", "#b4b2a9"

st.set_page_config(page_title="Bioactives & Allergy", layout="wide")


@st.cache_data
def load(name):
    return pd.read_csv(DASH / f"{name}.csv")


if not (DASH / "summary.csv").exists():
    st.error("No results found. Run the notebook first, then `python scripts/export_dashboard.py`.")
    st.stop()

S = load("summary").set_index("metric")["value"].astype(str)
studies = load("studies")
DSN = list(studies.dataset)


def metric(k):
    return S.get(k, "—")


PAGES = ["Overview", "Data & QC", "Differential expression", "Consensus signature", "Bioactive enrichment",
         "Direction of effect", "Pathways", "Network", "Machine learning", "Conclusions"]
st.sidebar.title("Bioactives & Allergy")
page = st.sidebar.radio("Section", PAGES)
st.sidebar.caption("Stages 1–8 of notebooks/analysis.ipynb. Every number here is read from results/.")

if page == "Overview":
    st.title("Dietary bioactives and the allergy transcriptome")
    st.markdown("**Research question.** Are genes consistently dysregulated across human allergic tissues over-represented "
                "among the curated targets of dietary bioactives, beyond gene study bias, and do these compounds shift "
                "their targets against the disease direction?")
    c = st.columns(5)
    c[0].metric("Signature genes", metric("n_sig"), f"{metric('n_sig_up')} up / {metric('n_sig_down')} down", delta_color="off")
    c[1].metric("Enriched, naive test", f"{metric('n_naive_sig')} / {metric('n_bioactives')}")
    c[2].metric("Enriched, degree-matched", f"{metric('n_deg_sig')} / {metric('n_bioactives')}", "sulforaphane", delta_color="off")
    c[3].metric("Effects opposing disease", f"{metric('rev_pct')}%", f"expected {metric('rev_expected_pct')}%", delta_color="off")
    c[4].metric("AUROC on unseen studies", metric("lodo_hub"), f"random genes {metric('lodo_random')}", delta_color="off")
    st.info(f"Across {metric('n_contrasts')} contrasts and {metric('n_samples')} samples, a {metric('n_sig')}-gene allergy "
            f"signature replicates across tissues. After correcting for study bias only {metric('n_deg_sig')} of "
            f"{metric('n_bioactives')} dietary bioactives stays enriched in it (vs {metric('n_naive_sig')} with a naive test).")
    st.subheader("Hypotheses")
    st.table(pd.DataFrame([
        ["H1", "Signature enriched for bioactive targets (degree-matched)", "Not supported (1 of 44)"],
        ["H2", "Bioactives more enriched than other chemicals", f"Not supported (p = {metric('mw_p')})"],
        ["H3", "Bioactives oppose the disease direction", f"Not supported ({metric('rev_pct')}% vs {metric('rev_expected_pct')}%)"],
        ["H4", "Bioactive targets classify disease better than baselines", "Partly: beat majority class, not random genes"],
    ], columns=["", "Hypothesis", "Result"]))

elif page == "Data & QC":
    st.title("Data and quality control")
    st.dataframe(studies, hide_index=True, width="stretch")
    fig = go.Figure([go.Bar(y=studies.dataset, x=studies.healthy, name="healthy", orientation="h", marker_color=DOWN),
                     go.Bar(y=studies.dataset, x=studies.allergic, name="allergic", orientation="h", marker_color=UP)])
    fig.update_layout(barmode="group", height=320, xaxis_title="samples after QC", yaxis_autorange="reversed", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.subheader("QC log")
    st.dataframe(load("qc"), hide_index=True, width="stretch")
    st.warning("GSE36830 was on a linear scale (log2 applied). GSE41861 has no sex-chromosome signal (values were adjusted "
               "before deposition). One GSE19187 sample recorded as female expresses Y-chromosome genes.")
    cols = st.columns(2)
    for col, f in zip(cols * 2, ["qc_sex_check", "qc_tissue_markers", "eda_pca", "eda_distributions"]):
        if (FIG / f"{f}.png").exists():
            col.image(str(FIG / f"{f}.png"), width="stretch")

elif page == "Differential expression":
    st.title("Differential expression per study")
    ds = st.selectbox("Contrast", DSN, index=DSN.index("Asthma bronchial") if "Asthma bronchial" in DSN else 0)
    v = load("volcano")
    v = v[v.dataset == ds]
    fig = px.scatter(v, x="log2FC", y="neglog10p", color="status", hover_name="gene",
                     color_discrete_map={"up": UP, "down": DOWN, "ns": MUTED}, opacity=0.7,
                     labels={"log2FC": "log2 fold change (allergic − healthy)", "neglog10p": "−log10 p"})
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray"); fig.add_vline(x=-0.5, line_dash="dash", line_color="gray")
    fig.update_traces(marker_size=5); fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")
    row = studies.set_index("dataset").loc[ds]
    st.caption(f"{row.up} genes up, {row.down} down (FDR < 0.05, |log2FC| ≥ 0.5). All DEGs shown plus 1,500 sampled non-significant genes.")
    st.subheader("Positive-control type-2 genes")
    pc = load("posctrl").pivot(index="dataset", columns="gene", values="log2FC").reindex(DSN)
    fig = px.imshow(pc, color_continuous_scale="RdBu_r", color_continuous_midpoint=0, text_auto=".1f", aspect="auto",
                    labels=dict(color="log2FC"))
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{metric('posctrl_up')} of {metric('posctrl_measured')} measured results significantly up, {metric('posctrl_down')} down.")

elif page == "Consensus signature":
    st.title("Consensus cross-tissue allergy signature")
    st.write(f"Weighted Stouffer meta-analysis over {metric('n_meta_genes')} genes → **{metric('n_sig')} signature genes** "
             f"({metric('n_sig_up')} up, {metric('n_sig_down')} down).")
    hm = load("sig_heatmap")
    order = hm.drop_duplicates("gene").sort_values("rank").gene
    mat = hm.pivot(index="gene", columns="dataset", values="log2FC").reindex(order)[DSN]
    fig = px.imshow(mat, color_continuous_scale="RdBu_r", color_continuous_midpoint=0, zmin=-3, zmax=3, aspect="auto",
                    labels=dict(color="log2FC"))
    fig.update_layout(height=820)
    st.plotly_chart(fig, width="stretch")
    q = st.text_input("Find a gene (top 300 signature genes)", "").strip().upper()
    sig = load("signature")
    st.dataframe(sig[sig.gene.str.upper().str.contains(q)] if q else sig, hide_index=True, width="stretch")
    st.subheader("Stability when one study is left out")
    st.dataframe(load("stability"), hide_index=True)

elif page == "Bioactive enrichment":
    st.title("Are bioactive targets enriched in the signature?")
    st.write(f"A naive test calls **{metric('pct_all_chem_naive')}%** of all CTD chemicals 'enriched', because inflammatory "
             "genes are the most studied. The degree-matched null compares against equally well-studied genes.")
    e = load("enrichment").sort_values("FE_degmatched")
    fig = go.Figure()
    for _, r in e.iterrows():
        fig.add_trace(go.Scatter(x=[r.FE_naive, r.FE_degmatched], y=[r.compound] * 2, mode="lines",
                                 line=dict(color="#d4d3cc", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=e.FE_naive, y=e.compound, mode="markers", name="naive",
                             marker=dict(symbol="circle-open", size=9, color="gray")))
    fig.add_trace(go.Scatter(x=e.FE_degmatched, y=e.compound, mode="markers", name="degree-matched",
                             marker=dict(size=10, color=np.where(e.FDR_degmatched < 0.05, GOOD, MUTED)),
                             customdata=e[["n_targets", "overlap", "FDR_degmatched"]],
                             hovertemplate="%{y}<br>FE %{x:.2f}<br>targets %{customdata[0]}, overlap %{customdata[1]}<br>FDR %{customdata[2]:.3f}"))
    fig.add_vline(x=1, line_color="black")
    fig.update_layout(height=900, xaxis_title="fold enrichment (green = degree-matched FDR < 0.05)")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    db = load("degbins")
    c1.plotly_chart(px.bar(db, x="decile", y="signature_pct", hover_data=["degree_range", "genes", "signature_genes"],
                           labels={"signature_pct": "% signature genes", "decile": "CTD degree decile (10 = most studied)"},
                           color_discrete_sequence=[UP], title="The confounder"), width="stretch")
    ch = load("chemicals")
    fig = px.scatter(ch, x="n_targets", y="log2FE", color="group", hover_name="compound", log_x=True,
                     color_discrete_map={"Dietary bioactive": GOOD, "Other CTD chemical": MUTED},
                     title=f"Dietary vs other chemicals (Mann–Whitney p = {metric('mw_p')})")
    fig.add_hline(y=0, line_color="black")
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(load("enrichment"), hide_index=True, width="stretch")

elif page == "Direction of effect":
    st.title("Do compounds push signature genes against the disease?")
    r = load("reversal").sort_values("reversal")
    fig = go.Figure([go.Bar(y=r.compound, x=r.reversal, orientation="h", name="observed",
                            marker_color=np.where(r.FDR < 0.05, GOOD, MUTED), text=r.n_reversed.astype(int).astype(str) + "/" + r.n_dir_sig.astype(int).astype(str)),
                     go.Scatter(y=r.compound, x=r.expected_reversal, mode="markers", name="expected",
                                marker=dict(symbol="line-ns-open", size=18, color="black", line_width=3))])
    fig.add_vline(x=0.5, line_dash="dash")
    fig.update_layout(height=520, xaxis=dict(range=[0, 1], tickformat=".0%", title="share of directional signature targets reversed"))
    st.plotly_chart(fig, width="stretch")
    st.write(f"Pooled: {metric('rev_pct')}% reversed vs {metric('rev_expected_pct')}% expected; "
             f"{metric('rev_n_sig')} of {metric('rev_n_compounds')} compounds significant.")
    st.dataframe(load("reversal"), hide_index=True, width="stretch")

elif page == "Pathways":
    st.title("Reactome pathway enrichment")
    rea = load("reactome")
    gs = st.radio("Gene set", rea.gene_set.unique().tolist(), horizontal=True)
    sub = rea[rea.gene_set == gs].head(15).iloc[::-1].assign(neglog10FDR=lambda d: -np.log10(d.FDR))
    fig = px.bar(sub, x="neglog10FDR", y="pathway", orientation="h", hover_data=["overlap", "size", "FE", "genes"],
                 color_discrete_sequence=[UP if gs.startswith("Up") else DOWN if gs.startswith("Down") else GOOD])
    fig.add_vline(x=-np.log10(0.05), line_dash="dash")
    fig.update_layout(height=560, xaxis_title="−log10 FDR", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.subheader("External check: CTD curated disease genes")
    st.dataframe(load("validation"), hide_index=True, width="stretch")
    st.caption("Negative result: curated allergy genes are mostly susceptibility loci and T-cell cytokines barely expressed in bulk tissue.")

elif page == "Network":
    st.title("Compound–gene network (NetworkX)")
    nodes, edges = load("net_nodes"), load("net_edges")
    pos = nodes.set_index("id")[["x", "y"]]
    fig = go.Figure()
    for eff, col in [("none", "#e1e0d9"), ("reinforces", WARN), ("reverses", GOOD)]:
        ex, ey = [], []
        for _, e in edges[edges.effect == eff].iterrows():
            ex += [pos.loc[e.source, "x"], pos.loc[e.target, "x"], None]
            ey += [pos.loc[e.source, "y"], pos.loc[e.target, "y"], None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=col, width=1), name=f"edge: {eff}", hoverinfo="skip"))
    g = nodes[nodes.kind == "gene"]
    fig.add_trace(go.Scatter(x=g.x, y=g.y, mode="markers+text", text=g.label, textposition="top center", name="gene",
                             marker=dict(size=8 + g.degree / 2, color=np.where(g.dir > 0, UP, DOWN), line=dict(color="white", width=1)),
                             hovertext=g.label + " · targeted by " + g.degree.astype(str) + " bioactives", hoverinfo="text"))
    c = nodes[nodes.kind != "gene"]
    fig.add_trace(go.Scatter(x=c.x, y=c.y, mode="markers+text", text=c.label.str.split(" \\(").str[0], textposition="bottom center",
                             name="compound", marker=dict(symbol="square", size=11, color="#52514e"), hoverinfo="text", hovertext=c.label))
    fig.update_layout(height=760, xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, width="stretch")
    h = load("hubs").head(20)
    fig = px.bar(h, y="gene", x=["reverses", "reinforces", "none"], orientation="h",
                 color_discrete_map={"reverses": GOOD, "reinforces": WARN, "none": MUTED}, title="Top hub genes: reversed or reinforced?")
    fig.update_layout(height=560, yaxis_autorange="reversed", xaxis_title="dietary compounds targeting the gene")
    st.plotly_chart(fig, width="stretch")

elif page == "Machine learning":
    st.title("Classifying allergic vs healthy samples")
    w = load("ml_within")
    keep = ["Bioactive hub targets", "Random genes (same size as hub set)", "Bioactive-targeted signature (LOO)",
            "All genes + in-fold top-100", "Bioactive hub targets – random forest", "Majority class"]
    fig = px.scatter(w[w.features.isin(keep)], x="AUROC", y="dataset", color="features", error_x="AUROC_sd",
                     category_orders={"dataset": DSN}, hover_data=["n_genes", "balanced_acc", "perm_p"])
    fig.update_traces(marker_size=11); fig.update_layout(height=520, xaxis_range=[0.4, 1.05])
    st.plotly_chart(fig, width="stretch")
    st.write(f"Hub targets ({metric('n_hub')} genes hit by ≥ 4 bioactives) beat {metric('hub_pct_random')}% of random gene sets of the same size on average.")
    st.subheader("Leave one study out")
    lo = load("ml_lodo")
    lo = lo[lo.dataset != "mean"].melt(id_vars="dataset", value_vars=["lr_hub", "lr_random_mean", "sig_score"],
                                      var_name="model", value_name="AUROC")
    lo["model"] = lo.model.map({"lr_hub": "LR on hub targets", "lr_random_mean": "LR on random genes", "sig_score": "Signature score (no training)"})
    fig = px.bar(lo, x="AUROC", y="dataset", color="model", barmode="group", orientation="h",
                 color_discrete_map={"LR on hub targets": UP, "LR on random genes": MUTED, "Signature score (no training)": DOWN})
    fig.add_vline(x=0.5, line_dash="dash"); fig.update_layout(height=520, xaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.write(f"Mean held-out AUROC: hub targets {metric('lodo_hub')}, random genes {metric('lodo_random')} "
             f"(p = {metric('lodo_p')}), untrained signature score {metric('lodo_sig')}.")

elif page == "Conclusions":
    st.title("Conclusions")
    st.markdown(f"""
**Answer:** a real, reproducible cross-tissue allergy signature exists, but dietary bioactives have no special link to it in curated data.

- **O1 – signature:** {metric('n_sig')} genes anchored by mast-cell and IL-13 genes (CPA3, CDH26, POSTN, CCL26, ALOX15); transfers to unseen studies with no training (AUROC {metric('lodo_sig')}).
- **O2 – enrichment & direction:** only sulforaphane survives the degree-matched null ({metric('sulf_fe')}×, FDR {metric('sulf_fdr')}); bioactives are no more enriched than {metric('n_other_chem')} other chemicals (p = {metric('mw_p')}); {metric('rev_pct')}% of effects oppose disease vs {metric('rev_expected_pct')}% expected.
- **O3 – biology:** up = interleukin signalling, immune interactions, proliferation; down = ceramide/sphingolipid synthesis (skin-barrier and ORMDL3 link). Network hubs are generic stress genes (BAX, NFE2L2, MMP9).
- **O4 – prediction:** hub targets beat the majority baseline but not random genes ({metric('lodo_hub')} vs {metric('lodo_random')}).

**Limitations:** CTD reflects what was studied (mostly cancer cell lines, high doses); some disease changes are protective (NRF2), so "reversal" is crude; heterogeneous small studies; GSE41861 tissues share subjects.

**Next steps:** perturbation data (LINCS L1000) at realistic doses; follow sulforaphane, curcumin and EGCG in single-cell data; test the ceramide barrier axis.
""")
