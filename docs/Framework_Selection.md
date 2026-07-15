# Framework Selection

The OmniReview project utilizes the following frameworks and libraries to support data preprocessing, model development, training, and evaluation.

| Framework / Library | Purpose |
|---------------------|---------|
| **PyTorch** | Implements the VAE, GAN, Diffusion Model, and Normalizing Flow architectures. |
| **Hugging Face Transformers** | Provides the pretrained T5-small model for review generation. |
| **PEFT (LoRA)** | Enables parameter-efficient fine-tuning of the Transformer model. |
| **Sentence Transformers** | Generates semantic embeddings for review headlines and bodies. |
| **Diffusers** | Supports diffusion-based refinement of latent representations. |
| **Pandas** | Performs data loading, preprocessing, and manipulation. |
| **NumPy** | Supports numerical computations during preprocessing and model development. |
| **Scikit-learn** | Used for dataset splitting, preprocessing, and evaluation utilities. |
| **Matplotlib** | Visualizes training progress and evaluation results. |
| **Jupyter Notebook** | Used for experimentation, model prototyping, and preliminary analysis. |

### Rationale

PyTorch was selected for its flexibility in implementing custom deep learning architectures. Hugging Face Transformers and PEFT (LoRA) provide efficient fine-tuning of pretrained language models, while Sentence Transformers generate high-quality semantic embeddings. Diffusers support diffusion-based refinement, and supporting libraries such as Pandas, NumPy, Scikit-learn, and Matplotlib facilitate data preprocessing, experimentation, and evaluation throughout the project.
