from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GEO = "https://ftp.ncbi.nlm.nih.gov/geo"


def series_matrix(gse: str) -> str:
    stub = gse[:-3] + "nnn"
    return f"{GEO}/series/{stub}/{gse}/matrix/{gse}_series_matrix.txt.gz"


def series_suppl(gse: str, fname: str) -> str:
    stub = gse[:-3] + "nnn"
    return f"{GEO}/series/{stub}/{gse}/suppl/{fname}"


def platform_annot(gpl: str) -> str:
    stub = gpl[:-3] + "nnn" if len(gpl) > 6 else "GPLnnn"
    return f"{GEO}/platforms/{stub}/{gpl}/annot/{gpl}.annot.gz"


DOWNLOADS = [
    ("geo", "GSE19187_series_matrix.txt.gz", series_matrix("GSE19187")),
    ("geo", "GSE36830_series_matrix.txt.gz", series_matrix("GSE36830")),
    ("geo", "GSE136825_genecounts.txt.gz",
     series_suppl("GSE136825", "GSE136825_genecounts_20190903.txt.gz")),
    ("geo", "GSE136825_series_matrix.txt.gz", series_matrix("GSE136825")),
    ("geo", "GSE41861_series_matrix.txt.gz", series_matrix("GSE41861")),
    ("geo", "GSE32924_series_matrix.txt.gz", series_matrix("GSE32924")),
    ("geo", "GSE58640_RNAseq_processed.txt.gz",
     series_suppl("GSE58640", "GSE58640_RNAseq_processed_GEOsubmission.txt.gz")),
    ("geo", "GSE58640_series_matrix.txt.gz", series_matrix("GSE58640")),
    ("geo", "GPL570.annot.gz", platform_annot("GPL570")),
    ("geo", "GPL6244.annot.gz", platform_annot("GPL6244")),
    ("ctd", "CTD_chem_gene_ixns.tsv.gz", "https://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz"),
    ("ctd", "CTD_chemicals.tsv.gz", "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"),
    ("ctd", "CTD_curated_genes_diseases.tsv.gz",
     "https://ctdbase.org/reports/CTD_curated_genes_diseases.tsv.gz"),
    ("hgnc", "hgnc_complete_set.txt",
     "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"),
    ("reactome", "ReactomePathways.gmt.zip", "https://reactome.org/download/current/ReactomePathways.gmt.zip"),
    ("reactome", "ReactomePathwaysRelation.txt",
     "https://reactome.org/download/current/ReactomePathwaysRelation.txt"),
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get  {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    print(f"       -> {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    failed = []
    for sub, name, url in DOWNLOADS:
        try:
            fetch(url, RAW / sub / name)
        except Exception as exc:
            print(f"  FAILED {name}: {exc}")
            failed.append(name)
    if failed:
        print("\nFailed downloads:", ", ".join(failed))
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
