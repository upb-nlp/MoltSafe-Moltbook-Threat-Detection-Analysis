# Moltbook Threat Detection

This repository accompanies the study "Evaluating Safety Embedding Prefiltering for Analyzing Millions of LLM Agent Social Network Messages for Security and Safety Harm" and the Hugging Face dataset repository `upb-nlp/MoltSafe-10K`. It contains the code used to construct the Moltbook corpus, obtain LLM-judge labels, run embedding-based contrast evaluations, and train and evaluate the lightweight classifier heads described in the paper.

Python commands below are written in module form. To see the available CLI commands:

```powershell
python -m moltbook_poc.cli --help
```

## Study Overview

The study evaluates whether local embedding-based prefilters can reduce the number of Moltbook nodes sent to a frontier LLM judge while preserving high recall for unsafe content. The pipeline first constructs the Moltbook node corpus, filters it to English-dominant nodes, and labels a sampled annotation pool with an LLM judge. The judged data are then converted into a fixed train/test split and a fixed 5-fold assignment used by all downstream evaluations.

We evaluate two families of prefilters. The contrast method is training-free: it embeds judged reference nodes, builds safe and unsafe centroids, and scores each candidate node by its relative similarity to those centroids. The classifier-head baseline trains only a lightweight 2-class linear head on top of a frozen Qwen3-Embedding encoder. In both cases, thresholds are selected within each fold to target recall 0.80, and reported fold summaries use the mean and sample standard deviation across the 5 folds.

## Results

The table below summarizes the trade-off between local throughput and projected frontier-judge cost for the selected prefilters and the trained classifier.

| System | Precision | Cost | Throughput (nodes/s) |
| --- | ---: | ---: | ---: |
| MiniLM-L12-v2, 128 / 16 | 0.205 +/- 0.011 | $3,117 | 332.18 +/- 0.50 |
| BGE-M3, 128 / 16 | 0.229 +/- 0.018 | $2,653 | 57.98 +/- 0.02 |
| Qwen3-0.6B classifier | 0.341 +/- 0.046 | $1,843 | 10.3 +/- 0.4 |

Relative to judging the full corpus without prefiltering, MiniLM-L12-v2 reduces the projected cost by 56.0%, BGE-M3 by approximately 62.6%, and the trained classifier by 74.0%.
MiniLM-L12-v2 is approximately 5.7 times faster than BGE-M3, while BGE-M3 is approximately 5.6 times faster than the trained classifier.
The trained classifier reduces unnecessary judge calls most effectively, while MiniLM-L12-v2 prioritizes local throughput and BGE-M3 provides an intermediate trade-off.

As agents become more capable and increasingly serve as personal assistants operating in environments with access to sensitive data, their exposure to social content also becomes a security concern. Moltbook provides a useful view of the kinds of malicious instructions, scams, unsafe requests, and adversarial content that may appear in agentic interactions. Because failures in these settings can have high-stakes consequences, we hope this release eases access to the study of malicious content on Moltbook by providing the full codebase together with the Hugging Face dataset.

## Environment

Run the setup commands from the repository root:

```powershell
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r req.txt
python -m pip install -e .
```

## Corpus Construction

Build the Moltbook node corpus and apply the language filter:

```powershell
python -m moltbook_poc.cli prep-data
python -m moltbook_poc.cli language-filter
```

The English-filtered parquet is the input to the judge sampling step.

## LLM Judge

Sample the judge input from the English-filtered corpus:

```powershell
python -m moltbook_poc.cli judge-sample `
  --n 10000 `
  --seed 42 `
  --parquet data\corpus\language_analysis_high_accuracy\moltbook_nodes_english.parquet `
  --output data\judge\judge_input.csv
```

Set the OpenAI key in the shell session before submitting judge batches:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Create the Batch API request file without submitting it:

```powershell
python -m moltbook_poc.cli judge-batch-submit `
  --input data\judge\judge_input.csv `
  --output-dir data\judge `
  --dry-run
```

Submit the batch:

```powershell
python -m moltbook_poc.cli judge-batch-submit `
  --input data\judge\judge_input.csv `
  --output-dir data\judge
```

Check the batch status:

```powershell
python -m moltbook_poc.cli judge-batch-status `
  --output-dir data\judge
```

Fetch and parse the completed batch:

```powershell
python -m moltbook_poc.cli judge-batch-fetch `
  --output-dir data\judge
```

This writes `data\judge\results.csv`, `data\judge\raw_responses.jsonl`, and `data\judge\manifest.json`.
For the paper-scale flow, `data\judge\results.csv` should contain the completed 10,000-row judge output before the fold data are prepared.

If the reader wishes to adapt the prompt used in our study, they can do so by inspecting the following file:

```text
data_prep\judge_api\prompt.txt
```

## Published Label Sync

To reproduce the paper evaluations without rerunning the OpenAI Batch API, use the published labelled dataset `upb-nlp/MoltSafe-10K`. The sync command downloads `dataset.csv` from Hugging Face, resolves those node IDs against the reconstructed English Moltbook corpus, verifies text hashes when present, and writes the local inputs expected by the fold-data builder:

```powershell
python -m moltbook_poc.cli sync-published-nodes `
  --hf-repo upb-nlp/MoltSafe-10K `
  --corpus data\corpus\language_analysis_high_accuracy\moltbook_nodes_english.parquet
```

## Fold Data

The contrast evaluation and classifier-head evaluation consume a shared fold-data directory. Build it either from freshly judged labels, or from the published-label sync outputs above.

Using the freshly judged labels:

```powershell
python -m moltbook_poc.cli prepare-contrast-data `
  --results data\judge\results.csv `
  --corpus data\corpus\language_analysis_high_accuracy\moltbook_nodes_english.parquet `
  --overwrite
```

Using the published-label sync path:

```powershell
python -m moltbook_poc.cli prepare-contrast-data `
  --results data\judge\results_merged.csv `
  --corpus data\synced\moltbook_nodes_10k.parquet `
  --overwrite
```

## Contrast Evaluation

Contrast configurations are stored under `config\contrast`. The configuration files associated with the experiments in our study are included in this repository.

For example, to run the MiniLM embedder, the reader can use the following command:

```powershell
python -m moltbook_poc.cli eval-contrast `
  --config config\contrast\minilm_128_16.yaml `
  --overwrite
```

## Classifier-Head Training

The classifier head is trained as a 2-class linear probe on top of frozen `Qwen/Qwen3-Embedding-0.6B`. The training procedure is intended to run in a Kaggle notebook with GPU enabled.

Prepare the data required:

```powershell
python -m moltbook_poc.cli prepare-head-eval-data
```

Additional files:

```text
data\fold_data\train.parquet
data\fold_data\test.parquet
data\fold_data\train_node_ids.txt
data\fold_data\test_node_ids.txt
data\fold_data\published_head_config.json
data\fold_data\contrast_summary_header.json
```

To upload only the Kaggle training package, stage the generated fold data beside the Kaggle scripts:

```powershell
New-Item -ItemType Directory -Force kaggle_classifier_training\data | Out-Null
Copy-Item data\fold_data\* kaggle_classifier_training\data\ -Recurse -Force
```

Upload `kaggle_classifier_training\` as a Kaggle Dataset. Create a Kaggle notebook with GPU enabled and Internet access, attach that Dataset, and run:

```text
kaggle_classifier_training\kaggle_notebook.ipynb
```

After the training is completed, download the completed notebook outputs:

```powershell
kaggle kernels output <kaggle-user>/<kernel-slug> -p _kaggle_heads_download
```

The expected local head directory is:

```text
_kaggle_heads_download\head_clf_stratified\runs_kfold\noprefix_h0
```

## Classifier-Head Evaluation

Evaluate downloaded heads without retraining:

```powershell
python -m moltbook_poc.cli eval-head `
  --eval-only-from _kaggle_heads_download\head_clf_stratified\runs_kfold\noprefix_h0 `
  --model-config-name kaggle_noprefix_h0_eval_only `
  --overwrite
```

## Citation

If you find this work useful, please cite:

TO DO: add citation when the paper is published.
