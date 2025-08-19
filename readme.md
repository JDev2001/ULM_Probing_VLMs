# ULM Project: Probing VLMs [![Build LaTeX document](https://github.com/JDev2001/ULM_Probing_VLMs/actions/workflows/build-pdf.yml/badge.svg?branch=main)](https://github.com/JDev2001/ULM_Probing_VLMs/actions/workflows/build-pdf.yml)

[Report](https://github.com/JDev2001/ULM_Probing_VLMs/blob/main/report.pdf)


## Install
- Download UV: https://docs.astral.sh/uv/getting-started/installation/
- Install Dependencies: ```uv sync```
- Install Pre-Commit Hooks: ``` uv run pre-commit install```
- Run files: ```uv run main.py```
- For Cuda Support: ```uv pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu128``` (https://pytorch.org/get-started/locally/)
- Access Gemma (Preview) Mode:
    - Create token at: https://huggingface.co/settings/tokens
    - ```huggingface-cli login```

## Description
This projects implements a probing pipeline with the mscoco-Dataset for different VLMs.

### Pipeline
- Dataset Loading
- Model Forward Pass
- Hidden-Layer Feature Extraction
- Train Probe Classifiers
- Evaluate & Visualize

### Currently Supported Models are
- "Qwen/Qwen2-VL-2B-Instruct"
- "google/gemma-3-4b-it",device="cpu"
- "microsoft/Phi-4-multimodal-instruct"
- In theory all models with AutoProcessor & AutoModelForCausalLM sould be supported (not guaranteed)
