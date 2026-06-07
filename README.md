# vad-boundary-margin-stt

Official implementation for the KCC 2026 paper:

**Analysis of the Effects of VAD Boundary Margin Adjustment on STT Performance**  
Hyungwon Yang, Taeho Kim, Younsou Choi, Suhan Son

## Overview
This repository contains code for the experiments reported in the KCC paper on boundary margin adjustment for VAD-based STT.

The study examines how leading and trailing boundary margins affect STT performance and whether controlled findings can improve practical VAD-based decoding.

The codebase supports two experiment stages used in the paper:

- **Step 1: Controlled Boundary Margin Analysis**
  - Uses forced-alignment (MFA) TextGrid boundaries
  - Sweeps leading and trailing margins around each utterance
  - Evaluates CER and boundary-sensitive error metrics
  - Produces the analysis tables and figures used in the paper

- **Step 2: Practical VAD-Based Decoding**
  - Uses Silero VAD for practical segmentation
  - Compares original-utterance baseline, default VAD decoding, and FA-informed VAD decoding
  - Evaluates whether margin settings derived from Step 1 improve practical decoding

## Main paper findings
According to the KCC paper, the main findings are:

1. Leading cuts are more harmful than trailing cuts.
2. Exact boundary trimming is suboptimal for stable STT decoding.
3. Moderate positive margins reduce degradation caused by aggressive boundary trimming.
4. Applying FA-informed margin settings to VAD-based decoding improves CER over default VAD segmentation.

The practical VAD experiment reported relative CER improvements of:
- **18.1%** on clean
- **14.9%** on other

## Repository structure

```text
conf/       configuration files
scripts/    entry-point scripts for preparation, decoding, and summarization
src/        core source code
artifacts/  generated intermediate files (kept empty in the public repo)
exp/        experiment outputs (kept empty in the public repo)
logs/       runtime logs (kept empty in the public repo)
data/       dataset root (kept empty in the public repo)
```

## Requirements

```bash
pip install -r requirements.txt
```

This repository assumes:
- LibriSpeech audio is already prepared
- MFA / forced-alignment TextGrid files are already prepared
- MFA itself is not installed or executed inside this repository

## Data preparation
The paper experiments use LibriSpeech evaluation splits only.

Expected layout:

```text
data/
  LibriSpeech/
    dev-clean/
    dev-other/
    test-clean/
    test-other/
    textgrid/
```

The STT backbone used in the paper is `wav2vec2-base-960h`.

## Configuration
The default experiment settings are defined in `conf/base.yaml`.

Important fields include:
- `datasets.librispeech.data_path`
- `datasets.librispeech.textgrid_path`
- `oracle.leading_ms`
- `oracle.trailing_ms`
- `silero_vad`
- `vad_policy`

The paper uses the following FA-informed VAD policy in Step 2:
- clean: leading 400 ms / trailing 400 ms
- other: leading 500 ms / trailing 500 ms

## Running the paper experiments

### Step 1: Controlled Boundary Margin Analysis
This step:
1. builds manifests
2. generates oracle margin variants
3. decodes baseline utterances
4. decodes oracle variants
5. produces per-utterance and summary CSV files
6. renders analysis tables and figures

Run:

```bash
bash scripts/run_experiment_pipeline.sh step1 librispeech
```

### Step 2: Practical VAD-Based Decoding
This step:
1. reuses manifests generated in Step 1
2. reuses baseline decoding from Step 1
3. runs `vad_default` decoding
4. runs `vad_fa_informed` decoding
5. produces per-utterance and summary CSV files
6. renders analysis tables and figures

Run:

```bash
bash scripts/run_experiment_pipeline.sh step2 librispeech
```

### Run all paper experiments

```bash
bash scripts/run_experiment_pipeline.sh all_steps librispeech
```

## Outputs
Generated files are saved under `exp/step1` and `exp/step2`.

### Step 1 outputs
- decoding results: `exp/step1/decodes/`
- evaluation results: `exp/step1/eval/`
- analysis tables and figures: `exp/step1/analysis/`

Frequently used Step 1 outputs:
- `exp/step1/analysis/tables/leading_trailing_cer.csv`
- `exp/step1/analysis/tables/leading_trailing_combined_error_profile.csv`
- `exp/step1/analysis/tables/leading_trailing_first_last_error_rate_profile.csv`
- `exp/step1/analysis/figures/leading_trailing_cer.pdf`
- `exp/step1/analysis/figures/leading_trailing_first_last_error_rate_profile.pdf`

### Step 2 outputs
- decoding results: `exp/step2/decodes/`
- evaluation results: `exp/step2/eval/`
- analysis tables and figures: `exp/step2/analysis/`

Frequently used Step 2 outputs:
- `exp/step2/analysis/tables/clean_other_table.csv`
- `exp/step2/analysis/tables/cer_boundary_changed_table.csv`
- `exp/step2/analysis/figures/vad_cer.pdf`
- `exp/step2/analysis/figures/vad_first_last_word_error.pdf`

## Citation
If you use this repository, please cite the corresponding paper.

```bibtex
@inproceedings{yang2026kcc,
  title={Analysis of the Effects of VAD Boundary Margin Adjustment on STT Performance},
  author={Yang, Hyungwon and Kim, Taeho and Choi, Younsou and Son, Suhan},
  booktitle={Proceedings of KCC},
  year={2026}
}
```
