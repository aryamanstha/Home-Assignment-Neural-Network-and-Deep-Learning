# Home Assignment 4: Neural Network and Deep Learning

### Name: Aryaman Shrestha 

### Student ID: 700788013

## Contents

- [`nlp_preprocessing.ipynb`](./nlp_preprocessing.ipynb) - tokenization,
  English stop-word removal, and Porter stemming using NLTK.
- [`entity_recognition.ipynb`](./entity_recognition.ipynb) - named-entity
  recognition with spaCy's `en_core_web_sm` model.
- [`attention.ipynb`](./attention.ipynb) - a NumPy implementation of
  scaled dot-product attention, including attention weights and the final
  weighted output.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Jupyter Notebook or JupyterLab

The Python dependencies are declared in [`pyproject.toml`](./pyproject.toml)
and locked in [`uv.lock`](./uv.lock).

## Setup

From the repository root, create the project environment and install the
dependencies:

```bash
uv sync
```

Install the spaCy English model required by
`entity_recognition.ipynb`:

```bash
uv run python -m spacy download en_core_web_sm
```

The preprocessing notebook downloads the NLTK `punkt_tab` and `stopwords`
resources when it is run. If you prefer to download them in advance:

```bash
uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Running the notebooks

Launch Jupyter from the project environment:

```bash
uv run --with jupyter jupyter notebook
```

Open any notebook and run its cells from top to bottom. The notebooks use the
same Python 3.11 environment created by `uv sync`.

## Attention example

The attention notebook computes:

1. Query-key similarity scores with `Q @ K.T`.
2. Scaled scores by dividing by the square root of the key dimension.
3. Softmax attention weights.
4. The output as the weighted sum of the value matrix.

This is a compact NumPy demonstration of the core operation used in
attention-based neural-network architectures.
