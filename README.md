# elcardiocc-baseline-ner

This repository contains the official **baseline Named Entity Recognition (NER) system** for the **ElCardioCC** task, part of the **BioASQ 2025** shared task. It is developed and provided by the **task organizers** for post-submission evaluation and comparison with participant systems.

## 🔍 About the Task

The ElCardioCC task focuses on entity recognition within clinical cardiology texts. This baseline provides a pipeline for training and evaluating an NER model using the annotated dataset distributed to participants.

## 📂 Repository Structure

```text
├── data/
│   ├── labelset.txt               # Full list of codes available in Doccano
│   ├── codes_list.txt             # TBD
│   ├── train_dataset.jsonl        # Raw dataset export from Doccano
│   ├── final_dataset.pickle       # Final dataset after train/val split
│   └── code_frequencies.xlsx      # Frequency distribution of codes
├── notebooks/
│   ├── preprocessing/
|       ├── validate_icd10_annotations.py  # Code & span validation for annotated dataset
│   │   └── preprocessing.ipynb    # Notebook for data preprocessing and formatting
│   ├── mBERT/
│   │   └── mBERT_training.ipynb   # TBD
│   ├── XLM-R/
│   │   └── XLMR_training.ipynb    # TBD
└── README.md
```

## ✅ Baseline Features

- Preprocessing of Doccano-annotated data  
- Validation of ICD-10 codes and span consistency  
- Simple NER training pipeline  
- Excel export of predictions and code frequency analysis

## 🚫 Note

This baseline is **not** a competition submission. It is provided **by the organizers** after the participant submission phase, to serve as a reference point for evaluation and transparency.

---

