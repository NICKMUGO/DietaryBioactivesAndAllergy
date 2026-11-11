import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "analysis.py"
DST = ROOT / "notebooks" / "analysis.ipynb"


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(SRC.read_text())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": []}
    if "--execute" in sys.argv:
        from nbconvert.preprocessors import ExecutePreprocessor
        ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    nbformat.write(nb, DST)
    print(f"wrote {DST.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
