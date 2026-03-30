# Neural Contraction Metrics

<video src="outputs/combined.mp4" autoplay loop muted playsinline width="100%"></video>

The code accompanying the submission "Learning Certified Neural Network Controllers Using Contraction and Interval Analysis"

## Setup

### 1. Clone (recursive)

```bash
git clone --recursive <repo-url>
cd neural_contraction
```

### 2. Install JAX 0.9.2

**CPU only:**
```bash
pip install "jax[cpu]==0.9.2"
```

**GPU (CUDA 12):**
```bash
pip install "jax[cuda12]==0.9.2"
```

**GPU (CUDA 13):**
```bash
pip install "jax[cuda13]==0.9.2"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for further details.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install immrax

```bash
pip install --no-deps ./immrax
```

## Usage

### Generate plots from a trained model

```bash
python plots.py
```

This saves per-maneuver plots and videos under `outputs/`. To tile all four maneuver videos into a single side-by-side video:

```bash
bash gen_stack.sh
```

Output: `outputs/combined.mp4`

### Train from scratch

```bash
python training.py
```

Trained weights are saved to `NCM/model.eqx` and `Controller/model.eqx`.
