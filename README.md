# vad-boundary-margin-stt

Code release for our KCC 2026 paper:

**Analysis of the Effects of VAD Boundary Margin Adjustment on STT Performance**  
Hyungwon Yang, Taeho Kim, Younsou Choi, Suhan Son

## Overview
This repository provides the code used in our KCC 2026 paper on boundary margin adjustment for VAD-based STT.

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
The paper reports the following findings:

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
- LibriSpeech audio is prepared under `data/LibriSpeech`
- MFA / forced-alignment TextGrid files are prepared under `data/LibriSpeech/textgrid`
- MFA is installed in a separate conda environment

## Data preparation
The paper experiments use LibriSpeech evaluation splits only.

Download the LibriSpeech evaluation splits:

```bash
bash scripts/data/download_librispeech.sh
```

This downloads and extracts the following splits from OpenSLR SLR12:
- `dev-clean`
- `dev-other`
- `test-clean`
- `test-other`

For forced alignment, create a separate MFA environment:

```bash
conda create -n mfa-aligner -c conda-forge montreal-forced-aligner pyyaml
conda activate mfa-aligner
```

Download the MFA pretrained dictionary and acoustic model:

```bash
mfa model download dictionary english_us_mfa
mfa model download acoustic english_mfa
```

Then prepare the LibriSpeech corpus for MFA and generate TextGrid files:

```bash
python scripts/data/run_mfa_librispeech.py --jobs 8
```

By default, this runs all four splits listed in `conf/base.yaml`: `dev-clean`, `dev-other`, `test-clean`, and `test-other`.
The script passes MFA's cleanup option by default so that temporary alignment files are reset before each run.
The generated TextGrid files are saved under `data/LibriSpeech/textgrid/<split>`.
After MFA finishes, return to the main experiment environment before running the STT pipeline.

The expected layout after data preparation is:

```text
data/
  LibriSpeech/
    dev-clean/
    dev-other/
    test-clean/
    test-other/
    textgrid/
      dev-clean/
      dev-other/
      test-clean/
      test-other/
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
- `exp/step1/analysis/tables/first_and_last_word_errors.csv`
- `exp/step1/analysis/figures/leading_trailing_cer.pdf`
- `exp/step1/analysis/figures/first_and_last_word_errors.pdf`

### Step 2 outputs
- decoding results: `exp/step2/decodes/`
- evaluation results: `exp/step2/eval/`
- analysis table: `exp/step2/analysis/`

Frequently used Step 2 outputs:
- `exp/step2/analysis/tables/cer_boundary_changed_table.csv`

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
