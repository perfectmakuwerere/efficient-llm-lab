# efficient-llm-lab

Fine-tuning, RL, and quantization experiments for small, efficient open-source LLMs that match or beat larger models on narrow tasks.

This repo collects an ongoing series of projects exploring how far you can push small open-source models — through fine-tuning, RL post-training, distillation, and quantization — before they match or outperform much larger general-purpose models on specific, well-defined tasks. The throughline across all of it: **train → fine-tune → RL → quantize → benchmark**, applied to a different task or model family each time, with results checked against standard public leaderboards wherever one exists rather than custom metrics alone.

Models and datasets are published under the [`nakue`](https://huggingface.co/nakue) Hugging Face account. Code for each project lives either in this repo (`notebooks/`) or linked out to its own repo where noted.

---

## Projects

### Function-calling: small model vs. large model
Fine-tuning `Qwen2.5-0.5B-Instruct` with Unsloth (LoRA) on `Salesforce/xlam-function-calling-60k` to produce a cheap, accurate "router" model — given a user query and a set of tool schemas, output the correct function call as JSON. Benchmarked against zero-shot `Qwen2.5-7B-Instruct` and Salesforce's own `xLAM-1b-fc-r` using the real **Berkeley Function-Calling Leaderboard (BFCL)** harness, for numbers directly comparable to the public leaderboard rather than a homemade metric.
- **Notebooks:** [`unsloth_funccall_finetune.ipynb`](notebooks/unsloth_funccall_finetune.ipynb), [`bfcl_eval_notebook.ipynb`](notebooks/bfcl_eval_notebook.ipynb)
- **Stack:** Unsloth, TRL (SFTTrainer), BFCL eval harness, vLLM
- **Status:** in progress — SFT done, BFCL evaluation underway, GRPO stage planned

### SmolLM2-1.7B quantization series
W4A16 and W8A8 quantized variants of SmolLM2-1.7B, benchmarked with `lm-eval-harness` on HellaSwag, ARC-Easy, and Winogrande to measure accuracy retention under compression.
- **Models:** [huggingface.co/nakue](https://huggingface.co/nakue)
- **Stack:** lm-eval-harness, quantization tooling (AWQ/GPTQ/llm-compressor)
- **Status:** complete

### Roofline Oracle
A Hugging Face Space (Gradio + Qwen2.5-7B-Instruct) that explains roofline analysis and arithmetic-intensity tradeoffs in plain English, separating the LLM's explanatory role from the underlying arithmetic. Planned extension: an agent loop that ingests a model config + target GPU and auto-estimates compute-bound vs. memory-bound regime and max batch size.
- **Space:** [huggingface.co/nakue](https://huggingface.co/nakue)
- **Status:** complete (v1); agent-loop extension planned

### Ollama benchmarking CLI
A command-line tool benchmarking Ollama model variants on TTFT, throughput, cosine similarity, and ROUGE-L.
- **Status:** complete

### Reasoning, tool-calling, and distillation training notebooks
Three complete training notebooks covering different post-training objectives:
- **Gemma-2-2b-it GSM8K reasoning pipeline** implementing the VibeThinker SSP shape
- **VibeThinker-3B LoRA tool-calling fine-tune** on `heegyu/glaive-function-calling-v2-formatted`
- **GLM-4-9B distillation / Qwen-format conversion pipeline**
- **Status:** complete

### Shona bilingual reasoning fine-tune
`Qwen3-1.7B` SFT on the `vamboai/fikira` dataset with W8A8 quantized variants, reframed as a bilingual Shona-English code-switching math explanation assistant rather than a replacement for English-medium instruction.
- **Models:** `nakue/Qwen3-1.7B-shona-reasoning-BF16` and W8A8 variants
- **Status:** complete

---

## Why this structure

Most fine-tuning portfolios are a pile of disconnected one-off uploads. The goal here is the opposite: each project should stand alone as a complete, benchmarked artifact, but read together as one coherent investigation into model efficiency — what cheap interventions (LoRA fine-tuning, RL with a verifiable reward, post-training quantization) buy you, and where a small specialized model can legitimately out-perform a much larger general one.

## Repo structure

```
notebooks/
  unsloth_funccall_finetune.ipynb
  bfcl_eval_notebook.ipynb
  ...
README.md
```

Each notebook is self-contained and documents its own setup, dataset, and eval methodology in markdown cells at the top.
