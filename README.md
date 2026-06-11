# NASA Galaxies Morphology Classification with CNNs

Cluster galaxy / star-cluster imagery from the public
[NASA Image and Video Library](https://images.nasa.gov) using a convolutional
neural network, and demonstrate a supervised CNN classifier end-to-end.

## 📓 Main deliverable

**[`galaxy_star_cnn_analysis.ipynb`](galaxy_star_cnn_analysis.ipynb)** — a fully
executed notebook (outputs and plots embedded) that:

1. **Downloads sample images** of galaxies and star clusters from the NASA Images
   REST API — **no API key / authentication required**
   (`https://images-api.nasa.gov/search?q=galaxy&media_type=image`).
2. Pre-processes them into a uniform image tensor dataset.
3. Trains a **convolutional auto-encoder (CNN)** to learn a compact visual
   feature embedding without labels.
4. **Clusters** those features with K-Means (cluster count chosen by silhouette
   score) and inspects each morphological group.
5. Trains a small **CNN classifier** for an *illustrative* "could-this-host-life?"
   target and reports a confusion matrix and classification report.

## ⚠️ Scientific honesty disclaimer

There is **no scientifically valid way to determine whether a galaxy or star
hosts life from an optical image**, and no ground-truth "images that contain
life" dataset exists. Real biosignature detection requires *spectroscopy of
individual exoplanet atmospheres*, not whole-galaxy pictures.

Therefore the **"contains life" / habitability section is an explicitly synthetic,
educational exercise**: it fabricates a proxy label from image statistics purely
to demonstrate the mechanics of training and evaluating a CNN classifier. Those
predictions carry **no astrobiological meaning**. The scientifically honest part
of this project is the **unsupervised morphology clustering**.

## Running it yourself

```bash
pip install tensorflow-cpu numpy pandas matplotlib scikit-learn pillow scipy jupyter
jupyter notebook galaxy_star_cnn_analysis.ipynb
```

The data-loading cell first tries to download **real NASA imagery**. If the
network is unavailable (sandbox / CI / restrictive firewall), it automatically
falls back to a **procedurally generated synthetic dataset** so every cell still
runs end-to-end. The outputs currently saved in the notebook were produced with
that synthetic fallback; re-run on a normal internet connection to use live NASA
images.

## Files

| File | Purpose |
|------|---------|
| `galaxy_star_cnn_analysis.ipynb` | The analysis notebook (executed, with outputs) |
| `build_notebook.py` | Script that regenerates the notebook via `nbformat` |
| `README.md` | This file |

## Possible next steps

- Replace the toy auto-encoder with a pretrained backbone (e.g. ResNet) fine-tuned
  on a labelled morphology dataset such as **Galaxy Zoo / Galaxy10**.
- Replace the fabricated life label with a *real* scientific target — spiral-vs-
  elliptical morphology, redshift bins, or exoplanet-transit light-curve classification.
