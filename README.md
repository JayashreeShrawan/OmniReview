# OmniReview: Leveraging VAEs, GANs, Diffusion Models, and Normalizing Flows for Joint Review Generation

## IE7374 – Special Topics: Generative AI

---

## Team Members

- Inchara Ashok Kumar
- Jayashree Sambasivam Muralidharan
- Saketh Mandava
- SenthilKumaran Ramanathan

---

# Project Overview

OmniReview is a Generative AI framework designed to generate high-quality, controllable product reviews by integrating multiple generative models within a shared latent space. The system jointly generates review headlines and review bodies while conditioning on product category, star rating, and review helpfulness.

Unlike traditional text generation approaches that rely on a single model, OmniReview combines Variational Autoencoders (VAEs), Transformer-based language models (T5 with LoRA), Diffusion Models, Generative Adversarial Networks (GANs), and Normalizing Flows to improve semantic consistency, fluency, diversity, and realism.

---

# Project Objectives

The objectives of this project are to:

- Learn a shared latent representation of product reviews using a Variational Autoencoder (VAE).
- Generate coherent review headlines and review bodies using T5-small with LoRA fine-tuning.
- Improve generated text quality through Diffusion-LM refinement.
- Estimate review helpfulness using a Normalizing Flow model.
- Evaluate generated reviews using adversarial learning through a GAN discriminator.
- Compare the proposed multi-model architecture with baseline approaches through comprehensive experiments.

---

# Dataset

**Dataset:** Amazon US Customer Reviews Dataset

The dataset contains millions of customer reviews collected across multiple product categories.

### Selected Categories

- Electronics
- Books
- Kitchen
- Apparel
- Digital Video Games

### Key Attributes

- Review Headline
- Review Body
- Star Rating
- Helpfulness Votes
- Verified Purchase
- Product Category

The dataset is filtered to verified purchases with sufficient voting information to improve label quality and reduce noise.

---

# Repository Structure

```text
OmniReview/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── README.md
│   ├── preprocess.py
│   └── create_splits.py
│
├── models/
│   ├── vae/
│   ├── transformer/
│   ├── diffusion/
│   ├── gan/
│   └── flow/
│
├── evaluation/
├── experiments/
├── docs/
│   └── Group01_Week9_Proposal.docx
│
├── outputs/
└── presentation/
```

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Sentence Transformers
- Diffusers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# Model Architecture

The OmniReview framework consists of five major components:

1. Variational Autoencoder (VAE)
2. Transformer (T5-small + LoRA)
3. Diffusion Model
4. GAN Discriminator
5. Normalizing Flow

Each model contributes a unique capability toward generating coherent, fluent, diverse, and controllable product reviews.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/JayashreeShrawan/OmniReview.git
cd OmniReview
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Current Usage

At the current stage of development, the data preprocessing pipeline has been initiated.

Run the preprocessing script:

```bash
python data/preprocess.py
```

Additional scripts for model training, fine-tuning, inference, and evaluation will be added as the project progresses.

---

# Evaluation Metrics

The project will evaluate model performance using:

- Conditional Perplexity
- BERTScore
- Headline–Body Similarity
- Rating–Sentiment Agreement
- Helpfulness Calibration Error
- Embedding-space FID
- GAN Discriminator AUC

---

# Project Timeline

| Phase | Status |
|--------|--------|
| Project Proposal | ✅ Completed |
| Repository Setup | ✅ Completed |
| Data Preprocessing | 🚧 In Progress |
| Model Development | ⏳ Planned |
| Model Training | ⏳ Planned |
| Evaluation | ⏳ Planned |
| Final Report & Presentation | ⏳ Planned |

---

# Current Project Status

- [x] Project Proposal
- [x] Repository Created
- [x] Repository Structure
- [x] Documentation Setup
- [x] Data Preprocessing Pipeline (Initial)
- [ ] Dataset Preparation
- [ ] VAE Implementation
- [ ] Transformer (T5 + LoRA)
- [ ] Diffusion Model
- [ ] GAN Integration
- [ ] Normalizing Flow
- [ ] Model Training
- [ ] Evaluation
- [ ] Final Report

---

# Documentation

Additional project documentation is available in the **docs** folder, including:

- Project Proposal
- Literature Review *(to be added)*
- Benchmarking *(to be added)*
- Model Architecture *(to be added)*
- Training Procedure *(to be added)*

---

# References

This project is based on recent research in:

- Variational Autoencoders (VAEs)
- Transformer Models (T5)
- LoRA (Low-Rank Adaptation)
- Diffusion Models
- Generative Adversarial Networks (GANs)
- Normalizing Flows
- Sentence-BERT

The complete literature review and reference list will be maintained in the **docs** directory as the project progresses.

---

# License

This project was developed for the **IE7374 – Special Topics: Generative AI** course at **Northeastern University** and is intended for academic purposes only.
