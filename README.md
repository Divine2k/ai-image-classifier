# AI Image Classifier

**Detecting AI-Generated Images vs Real Photos**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is an **ensemble-based classifier** designed to distinguish between **AI-generated images** and **real photographs**. It leverages three powerful pre-trained convolutional neural network models: **ResNet50**, **EfficientNetB0**, and **ConvNeXt Tiny**. By combining predictions from these models, the system achieves good detection performance.

## ✨ Features

- **Ensemble of 3 State-of-the-Art Models**: ResNet50 (85%), EfficientNetB0 (90%), ConvNeXt Tiny (92%)
- **High-Quality Training Data**: ~30K high-resolution images
- **Progressive Fine-Tuning**: Head → Head + Final Layer approach
- **Multiple Generator Detection**: SD1.5, SDXL, Midjourney, Grok, DALL·E
- **Command-line & Python API**: Easy to integrate

## 📊 Model Performance

# AI Image Classifier

**Detect AI-Generated Images vs Real Photographs**

This project provides a small ensemble classifier built with `ResNet50`, `EfficientNet-B0`, and `ConvNeXt Tiny` to help identify AI-generated images. A user-facing Gradio app is available in `app.py` for quick demos.

## Features

- Ensemble inference using three architectures
- Simple Gradio demo (`app.py`) for image uploads
- Trained model weights stored under `src/models/` (not included in repo by default)

## Detection Capabilities & Limitations

- Works well for many generator families (Stable Diffusion variants, Midjourney, DALL·E) but is not perfect.
- Performance varies with generator version, image post-processing, and dataset bias. Use results for research/preview purposes only.

## Quick Start

### Prerequisites

- Python 3.8+
- Recommended: a GPU for faster inference; CPU also works.

### Installation (Windows example)

```powershell
git clone <your-repo-url>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run the demo

```powershell
python app.py
```

The Gradio interface will open locally; in a Hugging Face Space, the same `requirements.txt` will be used to install dependencies.

## Pretrained Models & Demo

- **Model weights / download:** https://huggingface.co/divine2k/ai-image-detectors
- **Live demo (Hugging Face Space):** https://huggingface.co/spaces/divine2k/ai_image-classification

You can download the pretrained weights into `src/models/` (create the folder if missing) or use `huggingface_hub` to pull files programmatically. When deploying the Gradio app, ensure the model files are available under `src/models/` or update `app.py` to fetch them from the Hugging Face repo at startup.

## Requirements

This project only requires the minimal runtime packages used by `app.py`. See `requirements.txt` in the repo root. Example contents:

```text
torch>=2.0.0,<3.0
torchvision>=0.15.0,<1.0
gradio>=3.34.0,<4.0
Pillow>=9.0.0
numpy>=1.23.0
```

Notes:
- On Hugging Face Spaces, avoid pinning CUDA-specific `torch` wheels — the platform will install the appropriate build.

## If large files were already tracked

After adding `.gitignore`, untrack large artifacts with:

```bash
git rm -r --cached src/models/ models/ data/ logs/ notebooks/
git commit -m "Remove large artifacts from tracking; add .gitignore"
```

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit and open a PR

## License

MIT

---

Made with ❤️ using PyTorch and Gradio