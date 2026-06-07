# INT 7623 Final Project

This workspace contains a complete K-NN final project built around the UCI Raisin dataset.

## Main files
- `knn_raisin_project.py`: end-to-end analysis script
- `data/raw/`: original downloaded dataset files
- `data/processed/raisin_cleaned.csv`: cleaned dataset used in the project
- `outputs/`: generated metrics, split summary, and confusion matrices
- `report/final_report.md`: written report
- `report/presentation_outline.md`: slide-by-slide speaking outline

## Run
```bash
MPLCONFIGDIR=/Users/sathwik/Documents/New\\ project/.mplconfig python3 knn_raisin_project.py
```

After running the script, convert the HTML report to Word if needed:

```bash
textutil -convert docx report/final_report.html -output report/final_report.docx
textutil -convert doc report/final_report.html -output report/final_report.doc
```
