# A Retrieval-Augmented Generation (RAG) Framework for Xenophobic Sentiment Classification and Spatio-Temporal Resource Allocation in South Africa

GitHub repository name, shortened to the 100-character limit:

`retrieval-augmented-generation-xenophobic-sentiment-spatio-temporal-resource-allocation-south-africa`

Classification of hostile sentiment in South African tweets (2017–2022), and a town-week Green / Amber / Red priority model.

The working pipeline is in `xeno_rag_code_and_results_2/`. Cleaning notes are in `cleaned/data_cleaning.ipynb`. Run instructions, including the later binary-label and town-rate scripts, are in `xeno_rag_code_and_results_2/README.md`.

## Setup

```bash
pip install -r xeno_rag_code_and_results_2/requirements.txt
```

Place `SA_XenoDataset_model_ready.csv` in `xeno_rag_code_and_results_2/` before running the scripts. The cleaning notebook expects the raw export as `SA_XenoDataset_2017-2022.csv` in this folder.

## What is not in git

Tweet-level files stay local. They include post text, user locations, and contact details that should not be in a public repository. Town-level result tables, figures, and the implementation report are included. Generated indexes (`*.pkl`), the unused Chroma store, zip archives, and caches are ignored.

## Layout

- `cleaned/` — cleaning notebook and cleaning log
- `xeno_rag_code_and_results_2/src/` — pipeline scripts
- `xeno_rag_code_and_results_2/outputs/` — comparison tables and the report
- `xeno_rag_code_and_results_2/figures/` — maps and the rate-model chart
- `A-proposal.docx`, `RAG_Xenophobia_Proposal.pptx` — the proposal this code implements
