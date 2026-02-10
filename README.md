# elcardiocc-baseline-ner
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Stoikopoulou/elcardiocc-mbert-ner-baseline)

This repository contains the official **baseline Named Entity Recognition (NER) system** for the **ElCardioCC** task, part of the **BioASQ 2025** shared task. It is developed and provided by the **task organizers** for post-submission evaluation and comparison with participant systems.

**Disclaimer**: This baseline **is not** a competition submission. It is provided by the organizers post-submission to serve as a standardized reference point for transparency, and comparative analysis of participant systems.

##  Experimental Setup
Due to the computational intensity of fine-tuning the **mBERT** model, this project utilized **Google Colab Pro** resources:
- **Compute:** All experiments for finetuning were performed on **Google Colab** using an **NVIDIA Tesla T4 GPU** (16GB VRAM).
- **Reproducibility:** To ensure deterministic results, a fixed random seed of `42` was used for all PyTorch, NumPy, and Python random operations.
- **Tracking:** Experiment tracking, hyperparameter logging, and metric visualization were managed via **Weights & Biases (W&B)**.

## Inference

**Subword-to-Entity Reconstruction**
1.	**Subword Aggregation:** Rebuilds original words by merging Lexical Heads with their subsequent Sub-units (prefixed with ##).
2.	**Label Resolution:** Assigns a single class to the reconstructed word via Majority Voting across all fragments.
3.	**Coordinate Alignment:** Maps words to character offsets using Regex Search with a Stateful Pointer to ensure unique indexing of duplicate terms.

**Word-to-Phrase Grouping**
1.	**Neighbour Check**: Merges adjacent words if they are separated by exactly one character.
2.	**Grouping Logic**: Joins consecutive words into a single phrase whenever they are predicted as entities, regardless of whether they follow a strict B or I sequence.

**Known Limitations** 
1.	**The Strict +1 Gap:** Phrases are split if words are separated by more than one character (e.g., double spaces or newlines), as the logic requires an exact 1-character distance.
2.	**Character Mismatch:** The tokenizer does modify symbols like /, (, and + by isolating or stripping them. It also frequently normalizes Greek accents. These modifications cause the Regex Search to fail, as it cannot find an exact match in the original text, resulting in the permanent deletion of the entity from the results.

## Citation: 
If you use this code or the ElCardioCC dataset, please cite the following:

**BioASQ 2025 Task Overview**
```bash
@inproceedings{Dimitriadis2025OverviewOE,
  title={Overview of ElCardioCC Task on Clinical Coding in Cardiology at BioASQ 2025},
  author={Dimitris Dimitriadis and Vasiliki Patsiou and Eleonora Stoikopoulou and Achilleas Toumpas and Alkis Kipouros and Alexandra Bekiaridou and Konstantinos Barmpagiannos and Anthi Vasilopoulou and Antonios Barmpagiannos and Athanasios Samaras and Dimitrios Papadopoulos and George Giannakoulas and Grigorios Tsoumakas},
  booktitle={Conference and Labs of the Evaluation Forum},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:281670004}
}
```

**Master's Thesis**
```bash
@article{stoikopoulou2024,
      author        = "Stoikopoulou, E.",
      title         = "{Weakly Supervised NER for Cardiology Using Multilingual Transformers}",
      year          = "2024",
}
```

