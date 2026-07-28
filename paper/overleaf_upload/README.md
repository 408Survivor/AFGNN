# AFGNN Paper — Overleaf Upload Package

This folder contains everything needed to compile the AFGNN paper on Overleaf or a local LaTeX installation.

## Files

- `main.tex` — main manuscript (copy of `paper/afgnn_bibm.tex`).
- `afgnn_bibm.bib` — bibliography.
- `figures/` — all figure PDFs:
  - `afgnn_framework.pdf` — Figure 1: overall AFGNN framework.
  - `figure2_face_graph.pdf` — Figure 2: face graph construction.
  - `figure3_audio_graph.pdf` — Figure 3: audio temporal graph.
  - `figure4_ablation.pdf` — Figure 4: ablation study bar charts.

## Compile instructions

### Overleaf
1. Zip this folder (`overleaf_upload.zip`).
2. In Overleaf, click **New Project → Upload Project** and upload the zip.
3. Make sure the compiler is set to **pdfLaTeX** (Menu → Compiler).
4. The main file is `main.tex`; the bibliography is processed automatically.

### Local (if you have a working TeX Live/MacTeX)
```bash
cd overleaf_upload
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

> Note: The current development environment has a broken `pdflatex` format file and missing `bibtex`, so local compilation should be done on a standard TeX distribution or Overleaf.

## Key numbers in the paper

- Final 3-seed ensemble (seeds 42, 43, 44): **F1 = 0.8077**, **AUC = 0.8041**.
- All table numbers have been cross-checked against `experiments/results/*.json` in the project repository.
- See `paper/table_number_verification.md` for the full verification report.

## Citation style

The paper uses `IEEEtran` bibliography style (`\bibliographystyle{IEEEtran}`).

## Before submission

- Replace the anonymous author block in `main.tex` with real author/affiliation information.
- Remove any commented-out acknowledgments unless it is the camera-ready version.
- Double-check that all figures render correctly and that no placeholder boxes appear.
