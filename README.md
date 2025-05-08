# elcardiocc-baseline-ner

This repository contains the official **baseline Named Entity Recognition (NER) system** for the **ElCardioCC** task, part of the **BioASQ 2025** shared task. It is developed and provided by the **task organizers** for post-submission evaluation and comparison with participant systems.

## 🔍 About the Task

The ElCardioCC task focuses on entity recognition within clinical cardiology texts. This baseline provides a pipeline for training and evaluating an NER model using the annotated dataset distributed to participants.

## 📂 Repository Structure

- `labelset.txt` – Full list of codes available to annotators in Doccano  
- `codes_list.txt` – *(Optional: clarify usage if included)*  
- `train_dataset.jsonl` – Raw training dataset exported from Doccano  
- `final_dataset.pickle` – Final dataset after train/validation split  
- `code_frequencies.xlsx` – Code frequency statistics for exploration  
- `validate_icd10_annotations.py` – Validation script for code correctness and span overlaps  
- `src/` – (Optional) Python scripts for preprocessing, training, evaluation

## ✅ Baseline Features

- Preprocessing of Doccano-annotated data  
- Validation of ICD-10 codes and span consistency  
- Simple NER training pipeline  
- Excel export of predictions and code frequency analysis

## 🚫 Note

This baseline is **not** a competition submission. It is provided **by the organizers** after the participant submission phase, to serve as a reference point for evaluation and transparency.

---

