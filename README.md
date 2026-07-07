# OmniReview: Leveraging VAEs, GANs, Diffusion Models, and Normalizing Flows for Joint Review Generation

## IE7374 – Special Topics: Generative AI

### Team Members

* Inchara Ashok Kumar
* Jayashree Sambasivam Muralidharan
* Saketh Mandava
* SenthilKumaran Ramanathan

---

## Project Overview

OmniReview is a generative AI framework designed to generate high-quality, controllable product reviews by integrating multiple generative models within a shared latent space. The system jointly generates review headlines and review bodies while conditioning on product category, star rating, and review helpfulness.

Unlike traditional text generation approaches that rely on a single model, OmniReview combines Variational Autoencoders (VAEs), Transformer-based language models (T5 with LoRA), Diffusion Models, Generative Adversarial Networks (GANs), and Normalizing Flows to improve semantic consistency, fluency, diversity, and realism.

---

## Project Objectives

The objectives of this project are to:

* Learn a shared latent representation of product reviews using a Variational Autoencoder.
* Generate coherent review headlines and review bodies using T5-small with LoRA fine-tuning.
* Improve generated text quality through Diffusion-LM refinement.
* Estimate review helpfulness using a Normalizing Flow model.
* Evaluate generated reviews using adversarial learning through a GAN discriminator.
* Compare the proposed multi-model architecture with baseline approaches through comprehensive experiments.

---

## Dataset

**Dataset:** Amazon US Customer Reviews Dataset

The dataset contains millions of customer reviews across multiple product categories.

Selected Categories:

* Electronics
* Books
* Kitchen
* Apparel
* Digital Video Games

Key attributes include:

* Review Headline
* Review Body
* Star Rating
* Helpfulness Votes
* Verified Purchase
* Product Category

The dataset is filtered to verified purchases with sufficient voting information to improve label quality.

---

## Repository Structure

```text
OmniReview/
│
├── data/
├── models/
├── evaluation/
├── experiments/
├── docs/
├── outputs/
├── presentation/
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* Sentence Transformers
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Model Architecture

The OmniReview framework consists of five major components:

1. Variational Autoencoder (VAE)
2. Transformer (T5-small + LoRA)
3. Diffusion Model
4. GAN Discriminator
5. Normalizing Flow

These models work together to generate fluent, diverse, and controllable product reviews.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/JayashreeShrawan/OmniReview.git
cd OmniReview
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Data Preprocessing

```bash
python data/preprocess.py
```

### Train the VAE

```bash
python models/vae/train.py
```

### Fine-tune T5 with LoRA

```bash
python models/transformer/train.py
```

### Evaluate the Model

```bash
python evaluation/evaluate_all.py
```

---

## Evaluation Metrics

The project evaluates model performance using:

* Conditional Perplexity
* BERTScore
* Headline–Body Similarity
* Rating–Sentiment Agreement
* Helpfulness Calibration Error
* Embedding-space FID
* GAN Discriminator AUC

---

## Current Project Status

* [x] Project Proposal
* [ ] Data Preprocessing
* [ ] VAE Training
* [ ] Transformer Fine-tuning
* [ ] Diffusion Model
* [ ] GAN Integration
* [ ] Normalizing Flow
* [ ] Evaluation
* [ ] Final Report

---

## References

The project is based on recent research in VAEs, Transformers, Diffusion Models, GANs, Normalizing Flows, LoRA, and Sentence-BERT. A complete reference list is available in the project documentation.

---

## License

This project was developed for the IE7374 – Special Topics: Generative AI course and is intended for academic purposes.

