# KIVI: Knowledge-Intensive Video Generation

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

KIVI evaluates text-to-video models on **factuality** and **helpfulness** — shifting the question from *"Does the video look good?"* to *"Does the video communicate correct and useful information?"*

Given a short instructional prompt (e.g., *"How to set up cellular service on a Google Pixel 10"*), models must generate videos that are factually accurate and practically useful. KIVI-Bench includes 1,080 prompts across 18 knowledge-intensive categories, with automatic metrics that achieve ~70% agreement with human evaluation.

<p align="center">
  <img src="assets/overall_pipeline.png" alt="KIVI Pipeline" width="800">
</p>

---

## Installation

```bash
git clone <repo-url>
cd KIVI
conda create -n kivi python=3.10 -y && conda activate kivi
pip install -r requirements.txt
conda install ffmpeg
```

**API keys.** Set your LLM API key for script generation and evaluation:

```bash
export OPENAI_API_KEY={your_key}
```

The pipeline uses the OpenAI-compatible API format. By default it calls Gemini 3.1 Pro, but any provider that supports this format is supported (OpenAI, Gemini API, etc.).

For API-based video generation models, set the provider-specific key:

| Model | Variable |
|---|---|
| Seedance 2.0 | `ARK_API_KEY` |
| HappyHorse 1.0 | `DASHSCOPE_API_KEY` |

---

## Model Repositories

Each open-source video generation model requires its own code repository and conda environment. **Clone only the model(s) you need** into `video_generation_models/`. API-based models (Seedance 2.0, HappyHorse 1.0) do **not** require local code.

| Model | Clone Command |
|---|---|
| Wan 2.2 | `git clone https://github.com/Wan-Video/Wan2.2 video_generation_models/Wan2.2` |
| HunyuanVideo 1.5 | `git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 video_generation_models/HunyuanVideo-1.5` |
| LongCat-Video | `git clone https://github.com/meituan-longcat/LongCat-Video video_generation_models/LongCat-Video` |
| Helios | `git clone https://github.com/PKU-YuanGroup/Helios video_generation_models/Helios` |
| LongLive | `git clone https://github.com/NVlabs/LongLive video_generation_models/LongLive` |

Install model dependencies and download weights following each model's official documentation. For example, setting up Wan 2.2:

```bash
conda create -n {model_env} python=3.10 -y && conda activate {model_env}
pip install -r video_generation_models/Wan2.2/requirements.txt

# Download weights
huggingface-cli login
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir models_cache/Wan2.2_models/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir models_cache/Wan2.2_models/Wan2.2-I2V-A14B
```

---

## Usage

List available models:

```bash
python run_evaluation.py --list-models
```

### Main Arguments

| Argument | Description | Default |
|---|---|---|
| `--model {model}` | Model name to evaluate (see `--list-models`) | *required* |
| `--step {stage}` | Pipeline stage: `all`, `script`, `generate`, `extract`, `verify`, `score` | `all` |
| `--gpu {ids}` | GPU device IDs (e.g., `0` or `0,1` for multi-GPU) | `0` |
| `--category {name}` | Filter by category (e.g., `"Cars_Other_Vehicles"`). Runs all if omitted. | `None` |
| `--prompt-index {n}` | Filter to a specific prompt within a category (1-based). Requires `--category`. | `None` |
| `--prompts-json {path}` | Path to prompts JSON file | `experiment_prompts.json` |
| `--video-path {path}` | Evaluate a user-provided video. Requires `--prompt`. Skips script/generate. | `None` |
| `--prompt {text}` | Prompt text when using `--video-path` | `None` |

### Full Pipeline

```bash
python run_evaluation.py --model {model} --gpu 0
```

This runs all five stages in order: outline + script → video generation → claim extraction → claim verification → scoring.

### Individual Stages

Each stage reads/writes cached intermediates on disk and can be resumed independently:

```bash
python run_evaluation.py --model {model} --step script              # outline + script (LLM only)
python run_evaluation.py --model {model} --step generate --gpu 0     # video generation
python run_evaluation.py --model {model} --step extract             # claim extraction
python run_evaluation.py --model {model} --step verify              # claim verification
python run_evaluation.py --model {model} --step score               # compute final scores
```

### Custom Video Evaluation

Evaluate your own video without running model generation:

```bash
python run_evaluation.py --video-path {path/to/video.mp4} --prompt {your_video_prompt}
```

Results are saved to `evaluation/` next to the video file.

### Output Structure

```
outputs/{model}/{category}/Q{idx}_{prompt}/
├── final_video.mp4
├── outline.json
├── segment_prompts.json
└── evaluation/
    ├── extracted_claims.json
    ├── verification_results.json
    ├── helpfulness_score.json
    └── score.json
```
---
### Cross-Model Comparison

After evaluating multiple models, compare results:

```bash
python compare_scores.py
```

This prints per-prompt, per-category, and pairwise comparison tables, and exports CSVs.

---

## Prompt Sets

Two prompt sets are provided in the repository root:

| File | Description |
|---|---|
| `experiment_prompts.json` | 54 prompts (3 per category), used in the paper experiments. **Default.** |
| `kivi_bench_prompts.json` | Full KIVI-Bench: 1,080 prompts across 18 categories. |

Switch with `--prompts-json kivi_bench_prompts.json`.

---

## Evaluation Metrics

### Factual Precision (FactP)

```
FactP = correct claims / total claims × 100%
```

1. **Claim Extraction:** An LLM reviews the generated video and extracts atomic, externally verifiable statements about what the video depicts.
2. **Claim Verification:** Each claim is verified against world knowledge and classified as *Correct*, *Incorrect*, or *Uncertain*.

### Helpfulness Score (HelpS)

```
HelpS = (Relevance + Completeness + Clarity) / 3 × 100%
```

| Dimension | Description |
|---|---|
| Relevance | Does the video directly address the prompt and its explicit constraints? |
| Completeness | Are the key steps and required information covered? |
| Clarity | Is the procedure easy to follow, with logical sequence and clear pacing? |

---

## Results

Results on the 54-prompt subset using Gemini 3.1 Pro Preview as the evaluator:

| Model | FactP (%) | HelpS (%) |
|---|---|---|
| *Human (reference)* | 97.8 | 81.9 |
| Seedance 2.0 | 81.6 | 66.6 |
| HappyHorse 1.0 | 83.2 | 61.6 |
| Wan 2.2 | 73.1 | 48.4 |
| HunyuanVideo 1.5 | 63.2 | 32.9 |
| Helios-Base | 64.2 | 27.0 |
| LongCat-Video | 50.8 | 15.3 |
| LongLive 1.0 | 46.5 | 15.9 |

---

## Adding a New Video Generation Model

Four generation modes are supported: `segment` (T2V+I2V), `interactive` (prompt switching), `single_prompt`, and `api` (REST).

```bash
cp configs/_template_segment.yaml configs/{my-model}.yaml
# edit the YAML — fill in name, mode, code_dir, model_path, and generation commands
python run_evaluation.py --model {my-model} --gpu 0
```

---

## Repository Structure

```
├── run_evaluation.py            # Main entry point
├── compare_scores.py            # Cross-model score comparison
├── kivi/                        # Core framework
│   ├── llm_client.py            # LLM API client
│   ├── generation/              # Video generation (outline, script, models)
│   └── evaluation/              # Claim extraction, verification, helpfulness
├── configs/                     # Per-model YAML configurations
├── prompts/                     # LLM prompt templates
├── video_generation_models/     # Model code (clone manually)
├── outputs/                     # Evaluation outputs
└── models_cache/                # Downloaded model weights (gitignored)
```

---

## Citation

```bibtex
@article{wang2026kivi,
  title={KIVI: Knowledge-Intensive Video Generation},
  author={Wang, Chenxu and Chen, Mingda},
  journal={arXiv preprint},
  year={2026}
}
```