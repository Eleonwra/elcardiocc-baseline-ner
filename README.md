# elcardiocc-baseline-ner
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Stoikopoulou/elcardiocc-mbert-ner-baseline)

This repository contains the official **baseline Named Entity Recognition (NER) system** for the **ElCardioCC** task, part of the **BioASQ 2025** shared task. It is developed and provided by the **task organizers** for post-submission evaluation and comparison with participant systems.

**Disclaimer**: This baseline **is not** a competition submission. It is provided by the organizers post-submission to serve as a standardized reference point for transparency, and comparative analysis of participant systems.

##  Experimental Setup
Due to the computational intensity of fine-tuning the **mBERT** model, this project utilized **Google Colab Pro** resources:
- **Compute:** All experiments for finetuning were performed on **Google Colab** using an **NVIDIA Tesla T4 GPU** (16GB VRAM).
- **Reproducibility:** To ensure deterministic results, a fixed random seed of `42` was used for all PyTorch, NumPy, and Python random operations.
- **Tracking:** Experiment tracking, hyperparameter logging, and metric visualization were managed via **Weights & Biases (W&B)**.




## Citation: If you use this code or the ElCardioCC dataset, please cite the original BioASQ 2025 task overview: 

```bash
@inproceedings{Dimitriadis2025OverviewOE,
  title={Overview of ElCardioCC Task on Clinical Coding in Cardiology at BioASQ 2025},
  author={Dimitris Dimitriadis and Vasiliki Patsiou and Eleonora Stoikopoulou and Achilleas Toumpas and Alkis Kipouros and Alexandra Bekiaridou and Konstantinos Barmpagiannos and Anthi Vasilopoulou and Antonios Barmpagiannos and Athanasios Samaras and Dimitrios Papadopoulos and George Giannakoulas and Grigorios Tsoumakas},
  booktitle={Conference and Labs of the Evaluation Forum},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:281670004}
}
```
