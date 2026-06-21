# NASA Galaxies Morphology Classification with CNNs

Cluster galaxy / star-cluster imagery from the public
[NASA Image and Video Library](https://images.nasa.gov) using a convolutional
neural network, and demonstrate a supervised CNN classifier end-to-end.

## Main deliverable

**[`galaxy_star_cnn_analysis.ipynb`](galaxy_star_cnn_analysis.ipynb)** - a fully
executed notebook (outputs and plots embedded) that:

1. **Downloads sample images** of galaxies and star clusters from the NASA Images
   REST API - **no API key / authentication required**
   (`https://images-api.nasa.gov/search?q=galaxy&media_type=image`).
2. Pre-processes them into a uniform image tensor dataset.
3. Trains a **convolutional auto-encoder (CNN)** to learn a compact visual
   feature embedding without labels.
4. **Clusters** those features with K-Means (cluster count chosen by silhouette
   score) and inspects each morphological group.
5. Trains a small **CNN classifier** for an *illustrative* "could-this-host-life?"
   target and reports a confusion matrix and classification report.

## Project structure

| File | Purpose |
|------|---------|
| `galaxy_star_cnn_analysis.ipynb` | The analysis notebook (executed, with outputs) |
| `build_notebook.py` | Script that regenerates the notebook via `nbformat` |
| `README.md` | This file |

## Data pipeline

The notebook is self-contained and does not require any cached data. On launch
it runs the following pipeline:

1. **Query** the NASA Images search endpoint for each category in `CATEGORIES`
   (default: `"galaxy"` and `"star cluster"`).
2. For each search hit, **follow the `collection.json` asset listing** to find
   the actual JPEG/PNG asset URLs. Small/thumbnail variants are preferred to
   keep downloads light.
3. **Decode each asset** with Pillow, convert to RGB, resize to `IMG_SIZE`
   (default 64x64), and normalise pixels to `[0, 1]`.
4. **Shuffle and stack** the per-category arrays into an `(N, 64, 64, 3)`
   tensor with integer category labels.

If the NASA endpoint is unreachable (sandbox, CI, restrictive firewall) the
pipeline transparently switches to a **procedurally generated synthetic
dataset** so every downstream cell still has data to operate on:

- **Galaxies** - a bright central bulge plus logarithmic spiral arms (spiral)
  or a smooth elongated light profile (elliptical), over a faint star field.
- **Star clusters** - a dark sky scattered with point-like sources (2-D
  Gaussian PSFs) of varying brightness.

## Model architecture

### Convolutional auto-encoder (unsupervised feature learning)

The encoder compresses each `64x64x3` image down to a `LATENT_DIM`-dimensional
vector (default 32). The decoder mirrors the encoder with transposed
convolutions:

```
Input (64,64,3)
  Conv2D 16 -> MaxPool   ->  (32,32,16)
  Conv2D 32 -> MaxPool   ->  (16,16,32)
  Conv2D 32 -> MaxPool   ->  ( 8, 8,32)
  Flatten -> Dense(32, relu)  = latent embedding
  Dense -> Reshape -> 3 x Conv2DTranspose -> Conv2D(3, sigmoid)
Output (64,64,3)
```

Trained with `mse` reconstruction loss, Adam optimizer, 40 epochs,
`batch_size=16`, `validation_split=0.15`. After training, the encoder alone is
reused to produce a feature matrix `(N, LATENT_DIM)` that is fed to clustering.

If TensorFlow is not installed, the notebook falls back to PCA on flattened
pixels so the clustering pipeline still runs.

### K-Means clustering

The latent features are standardised with `StandardScaler` and clustered with
`KMeans`. The notebook sweeps `k = 2..6` and picks the `k` with the best
**silhouette score**, then plots a PCA projection coloured by cluster and by
the original NASA search category for visual comparison. A contingency table
(`pd.crosstab`) quantifies how the unsupervised groups line up with the search
terms.

### Habitability CNN classifier (illustrative)

A small supervised CNN (Conv 16 -> Conv 32 -> Conv 64 -> GAP -> Dense 32 ->
sigmoid) trained with binary cross-entropy on a **fabricated** habitability
label produced from image statistics (brightness, contrast, local gradient
structure). Reports a confusion matrix and `classification_report` on a 25%
held-out test split.

## Scientific honesty disclaimer

There is **no scientifically valid way to determine whether a galaxy or star
hosts life from an optical image**, and no ground-truth "images that contain
life" dataset exists. Real biosignature detection requires *spectroscopy of
individual exoplanet atmospheres*, not whole-galaxy pictures.

Therefore the **"contains life" / habitability section is an explicitly
synthetic, educational exercise**: it fabricates a proxy label from image
statistics purely to demonstrate the mechanics of training and evaluating a
CNN classifier. Those predictions carry **no astrobiological meaning**. The
scientifically honest part of this project is the **unsupervised morphology
clustering**.

## Running it yourself

```bash
pip install tensorflow-cpu numpy pandas matplotlib scikit-learn pillow scipy jupyter
jupyter notebook galaxy_star_cnn_analysis.ipynb
```

To regenerate the notebook from scratch (useful after editing the source
script):

```bash
pip install nbformat
python build_notebook.py
```

This rewrites `galaxy_star_cnn_analysis.ipynb` from the cells defined in
`build_notebook.py`.

The data-loading cell first tries to download **real NASA imagery**. If the
network is unavailable (sandbox / CI / restrictive firewall), it automatically
falls back to a **procedurally generated synthetic dataset** so every cell
still runs end-to-end. The outputs currently saved in the notebook were
produced with that synthetic fallback; re-run on a normal internet connection
to use live NASA images.

## Configuration knobs

All tunables live at the top of the setup cell in
`galaxy_star_cnn_analysis.ipynb`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `IMG_SIZE` | `64` | Side length of the square resized images |
| `PER_CATEGORY` | `60` | Images requested per NASA search category |
| `CATEGORIES` | `["galaxy", "star cluster"]` | NASA search queries |
| `LATENT_DIM` | `32` | Size of the CNN feature embedding |
| `SEED` | `42` | Random seed for NumPy / TensorFlow / Python `random` |

Increase `PER_CATEGORY` for a larger dataset (slower download, better-trained
auto-encoder). Increase `IMG_SIZE` for more visual detail (heavier model,
slower training). Add new entries to `CATEGORIES` to explore other NASA search
terms (e.g. `"nebula"`, `"supernova remnant"`).

## Reproducibility notes

- All randomness is seeded via `SEED = 42` (`random`, NumPy, TensorFlow).
- NASA search results are not perfectly stable over time: the API may return a
  different ordering or slightly different assets between runs. The
  `random.shuffle(urls)` step further randomises selection within `SEED`.
- The synthetic fallback is fully deterministic given `SEED` and
  `PER_CATEGORY`.

## Dependencies

- Python 3.10+
- `tensorflow-cpu` (or `tensorflow`) - optional but recommended for the real
  CNN path; the notebook degrades gracefully to scikit-learn if absent
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn` (StandardScaler, KMeans, silhouette_score, PCA, MLPClassifier
  fallback, train_test_split, classification_report, confusion_matrix)
- `pillow` for image decoding
- `requests` for the NASA HTTP calls
- `jupyter` to run the notebook
- `nbformat` only if you intend to rebuild the notebook via
  `build_notebook.py`

## License

The code in this repository is released for educational use. NASA imagery is
in the public domain unless otherwise noted on the individual asset page;
attribution to NASA is appreciated when republishing.
