# FL-GradInversion: Privacy Attack in Federated Learning

## 🎯 Overview

This repository implements **GradInversion**, a privacy attack that recovers original training images from batch-averaged gradients in federated learning. Based on the NVIDIA paper:

**"See through Gradients: Image Batch Recovery via GradInversion"**
- Authors: Hongxu Yin, Arun Mallya, Arash Vahdat, Jose M. Alvarez, Jan Kautz, Pavlo Molchanov
- Category: 1 - Privacy & Inference Attacks
- Course: Federated Learning Advanced Topics, IIT Patna

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/mig2002/fl-gradinversion-privacy.git
cd fl-gradinversion-privacy

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt