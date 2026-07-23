# OmniReview: Controllable Multi-Attribute Product Review Generation

Leveraging Variational Autoencoders (VAEs), Normalizing Flows, Diffusion Models, GANs, and Transformer-based Language Models for controllable product review generation.

---

## Project Overview

OmniReview is a controllable, multi-attribute text generation framework that generates synthetic Amazon product reviews while simultaneously controlling:

- Product Category
- Star Rating
- Helpfulness Ratio

Unlike conventional language models, OmniReview coordinates multiple generative models within a shared latent space to improve semantic consistency, controllability, and text quality.

---

## Project Objectives

The project investigates three research questions:

- Can a shared latent representation improve headline-body consistency?
- Can Normalizing Flows accurately model continuous helpfulness distributions?
- Can latent diffusion refine VAE-generated embeddings and improve generation quality?

---

## Model Architecture

The OmniReview pipeline integrates six deep learning models:

| Model | Purpose |
|--------|---------|
| OmniVAE | Learns shared latent representations |
| Normalizing Flow (MAF) | Models helpfulness distributions |
| Latent Diffusion (DDPM) | Refines latent embeddings |
| GAN Discriminator | Filters unrealistic generations |
| LatentConditionedT5 | Generates fluent review text |
| TextCNN / CharRNN | Baseline comparison models |

---

## Dataset

Dataset:
Amazon US Customer Reviews Dataset

Preprocessing includes:

- Verified purchases only
- Removal of Vine reviews
- Helpfulness ratio computation
- Sentence-BERT embeddings
- Stratified Train/Validation/Test split

---

# Installation

## 1. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Prepare Dataset

Ensure the following files are available inside `data/`

- train_clean.parquet
- val_clean.parquet
- test_clean.parquet
- train_body_emb.npy
- train_head_emb.npy
- val_body_emb.npy
- val_head_emb.npy
- test_body_emb.npy
- test_head_emb.npy

These files are generated using

```bash
python data/preprocess.py
python data/embed.py
```

---

## 4. Train Models (Optional)

```bash
python src/train.py --model all
```

If pretrained checkpoints are available locally, this step can be skipped.

---

## 5. Run with Docker (Optional)

```bash
docker build -t omnireview .
docker run --rm omnireview
```

---

## 6. Run Inference

Generate default samples

```bash
python src/model_runner.py
```

Generate multiple samples

```bash
python src/model_runner.py --samples 10
```

Generate category-specific reviews

```bash
python src/model_runner.py --category Electronics --rating 5
```

Generated outputs are stored inside

```
outputs/
```

including

- generated_reviews.csv
- samples.txt

---

# Repository Structure

```
OmniReview/
│
├── src/
│   ├── data_loader.py
│   ├── model_runner.py
│   ├── train.py
│   └── models/
│       ├── vae.py
│       ├── transformer.py
│       ├── diffusion.py
│       ├── gan.py
│       ├── flow.py
│       ├── classifier.py
│       └── baseline.py
│
├── utils/
│   └── helpers.py
│
├── configs/
│   └── model_config.yaml
│
├── data/
│
├── outputs/
│   ├── generated_reviews.csv
│   ├── samples.txt
│   └── README.md
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Preliminary Results

Example generated reviews

| Category | Rating | Generated Review |
|-----------|--------|------------------|
| Electronics | ⭐⭐⭐⭐⭐ | Great product with excellent quality and performance... |
| Books | ⭐☆☆☆☆ | Disappointing storyline and weak character development... |
| Apparel | ⭐⭐⭐☆☆ | Comfortable fit but average material quality... |

### Evaluation Metrics

| Metric | Result |
|----------|--------|
| T5 Validation Perplexity | 30.34 |
| GAN Discriminator AUC | 1.0000 |
| Flow KS-Test Statistic | 0.1680 |

---

# Outputs

Successful execution generates:

- generated_reviews.csv
- samples.txt

Additional evaluation results, logs, and visualizations may also be stored inside the `outputs/` directory.

---

# Known Limitations

- Posterior collapse in the VAE may reduce latent diversity.
- Beam search can generate repetitive text; temperature sampling improves diversity.
- Model checkpoints are excluded from the repository because of their size.

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- LoRA (PEFT)
- Sentence-BERT
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Docker

---

# Team Members

- Jayashree Sambasivam Muralidharan
- Inchara Ashok Kumar
- Saketh Mandava
- SenthilKumaran Ramanathan

---

# License

This project was developed as part of the Northeastern University Generative AI course and is intended for academic and research purposes.