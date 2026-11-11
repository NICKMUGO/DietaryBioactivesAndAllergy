#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m pip install --quiet --user streamlit plotly pandas
if [ ! -f results/dashboard/summary.csv ]; then
  echo "No results yet - running the analysis first (about 5-15 minutes)..."
  python3 scripts/download_data.py
  python3 scripts/build_notebook.py --execute
  python3 scripts/export_dashboard.py
fi
python3 -m streamlit run app.py --server.headless true
