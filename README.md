# 🔍 PaliGemma Vision-Language Model — From Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-22C55E?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**A clean, from-scratch PyTorch implementation of Google's PaliGemma multimodal vision-language model.**

[Getting Started](#installation) · [Usage](#usage) · [Architecture](#technologies) · [Contributing](#contributing)

</div>

---

## Introduction

PaliGemma is a state-of-the-art **vision-language model (VLM)** developed by Google, capable of understanding both images and text to generate descriptive, contextual responses. This repository provides a **complete ground-up PyTorch implementation** of PaliGemma — built without relying on pre-built model classes — making it an ideal resource for researchers, students, and engineers who want to deeply understand how modern multimodal AI systems work.

Rather than treating the model as a black box, this codebase exposes every architectural component: the SigLIP vision encoder, the Gemma language decoder, the multimodal projector, KV-caching, rotary positional embeddings, and the token-merging pipeline that fuses vision and language representations.

**What problem does it solve?**
Most VLM tutorials abstract away the internals. This project strips that away so you can see exactly how image patches become tokens, how attention flows between visual and textual representations, and how autoregressive decoding works in a multimodal context.

---

## Features

-  **Built from scratch** — every layer (attention, MLP, RMS norm, RoPE) implemented in pure PyTorch with no high-level wrappers
-  **SigLIP Vision Encoder** — patch embedding, multi-head self-attention, and feed-forward layers for visual feature extraction
-  **Gemma Language Decoder** — full decoder-only transformer with grouped query attention (GQA) and RMS normalization
-  **Multimodal Projector** — linear projection bridge that aligns visual embeddings into the language model's token space
-  **KV-Cache** — efficient key-value caching for fast autoregressive inference
-  **Rotary Positional Embeddings (RoPE)** — position-aware attention without absolute positional encoding
-  **Flexible Sampling** — supports greedy decoding and nucleus (top-p) sampling with temperature control
-  **HuggingFace Compatible** — loads weights directly from `.safetensors` checkpoints on the HuggingFace Hub
-  **Multi-device Support** — runs on CPU, CUDA, and Apple Silicon (MPS)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+ (with CUDA for GPU acceleration, or MPS for Apple Silicon)
- A HuggingFace account to download model weights

### 1. Clone the Repository

```bash
git clone https://github.com/AmmarMohamed0/siglipVisionFromScratch.git
cd siglipVisionFromScratch
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision pillow numpy transformers safetensors fire
```

Or, `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

Download the PaliGemma pretrained weights from HuggingFace. You will need to accept the model's usage terms on the Hub.

```bash
# Install the HuggingFace CLI if you haven't already
pip install huggingface_hub

# Log in to your account
huggingface-cli login

# Download the model
huggingface-cli download google/paligemma-3b-pt-224 --local-dir ./paligemma-3b-pt-224
```

---

## Usage

### Running Inference via Shell Script

The easiest way to run inference is using the provided shell script. Edit `launch_inference.sh` to set your paths and parameters:

```bash
#!/bin/bash

MODEL_PATH="./paligemma-3b-pt-224"
PROMPT="this building is "
IMAGE_FILE_PATH="test_images/pic1.jpeg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"
```

Then run:

```bash
bash launch_inference.sh
```

### Running Inference via Python

You can also invoke `inference.py` directly using the `fire` CLI:

```bash
python inference.py \
  --model_path "./paligemma-3b-pt-224" \
  --prompt "describe what you see: " \
  --image_file_path "test_images/pic1.jpeg" \
  --max_tokens_to_generate 150 \
  --temperature 0.7 \
  --top_p 0.9 \
  --do_sample True \
  --only_cpu False
```

### Inference Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | required | Path to the downloaded model directory |
| `prompt` | `str` | required | Text prompt to condition generation |
| `image_file_path` | `str` | required | Path to the input image |
| `max_tokens_to_generate` | `int` | `100` | Maximum number of new tokens to generate |
| `temperature` | `float` | `0.8` | Sampling temperature (higher = more random) |
| `top_p` | `float` | `0.9` | Nucleus sampling probability mass threshold |
| `do_sample` | `bool` | `False` | Use sampling; if `False`, uses greedy decoding |
| `only_cpu` | `bool` | `False` | Force CPU inference even if GPU is available |

### Example Prompts

```bash
# Image captioning
--prompt "caption en: "

# Visual question answering
--prompt "what color is the sky in this image? "

# Object detection description
--prompt "describe all objects visible: "

# Scene completion
--prompt "this landmark is located in "
```

### Programmatic Usage

```python
from PIL import Image
import torch
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = load_hf_model("./paligemma-3b-pt-224", device)
model = model.to(device).eval()

num_image_tokens = model.config.vision_config.num_image_tokens
image_size = model.config.vision_config.image_size
processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

image = Image.open("test_images/pic1.jpeg")
inputs = processor(text=["this image shows "], images=[image])
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference ...
```

---

## Technologies

### Core Framework

- **[PyTorch](https://pytorch.org/)** — tensor computation and neural network primitives

### Model Architecture

- **SigLIP Vision Encoder** — Vision Transformer (ViT) backbone with patch embeddings and multi-head self-attention
- **Gemma Language Decoder** — decoder-only transformer with Grouped Query Attention (GQA), RMSNorm, and gated MLP (GeGLU)
- **Rotary Positional Embeddings (RoPE)** — position-aware attention mechanism
- **KV-Cache** — stateful key/value cache for efficient autoregressive decoding

### Utilities & Tooling

- **[HuggingFace Transformers](https://huggingface.co/docs/transformers)** — tokenizer loading
- **[Safetensors](https://github.com/huggingface/safetensors)** — fast and safe model weight loading
- **[Pillow (PIL)](https://pillow.readthedocs.io/)** — image loading and preprocessing
- **[NumPy](https://numpy.org/)** — image normalization and array operations
- **[Fire](https://github.com/google/python-fire)** — automatic CLI generation

---

## Project Structure

```
paligemma-from-scratch/
│
├── modeling_siglip.py          # SigLIP Vision Encoder (ViT backbone)
├── modeling_gemma.py           # Gemma LM decoder + PaliGemma multimodal model
├── processing_paligemma.py     # Image preprocessing and prompt formatting
├── utils.py                    # Model and tokenizer loading from HuggingFace
├── inference.py                # End-to-end inference pipeline
├── launch_inference.sh         # Shell script for easy inference runs
│
├── test_images/                # Place your test images here
│   └── pic1.jpeg
│
└── README.md
```

---

## Contributing

Contributions are warmly welcome! Here's how to get involved:

1. **Fork** the repository and create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — fix a bug, add a feature, improve documentation, or add test cases.

3. **Follow code style** — keep implementations clean, well-commented, and consistent with the existing patterns. Dimension annotations in comments (e.g., `# [Batch_Size, Seq_Len, Hidden_Size]`) are especially encouraged.

4. **Commit and push**:
   ```bash
   git commit -m "feat: add flash attention support"
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** with a clear description of what was changed and why.

### Good First Issues

- Add support for batched multi-image inference
- Implement Flash Attention as an optional backend
- Add a Gradio or Streamlit demo interface
- Write unit tests for individual model components
- Add support for additional PaliGemma model sizes (448, 896)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Contact

**Author:** Ammar Mohamed Amin
**Email:** ammarmohamedamin0@gmail.com
**GitHub:** [@AmmarMohamed0](https://github.com/AmmarMohamed0)
**LinkedIn:** [ammar-mohamed-amin](https://www.linkedin.com/in/ammar-mohamed-amin/)

---

## Acknowledgments

- **[Google DeepMind](https://deepmind.google/)** — for developing and open-sourcing the PaliGemma model family
- **[HuggingFace](https://huggingface.co/)** — for model hosting, tokenizer infrastructure, and the `safetensors` library
- **Umar Jamil** — whose educational content on transformer internals inspired this implementation style
- The broader open-source ML community for papers, reference implementations, and discussions that made this possible

---

<div align="center">

Made with ❤️ and PyTorch

⭐ Star this repo if you found it helpful!

</div>