# Q-stage: Attention-Based Foundation Model for Quantum States

This repository contains the research code for **Q-stage**, an attention-based foundation model architecture for quantum many-body states, developed for the project/paper

> **“Attention-Based Foundation Model for Quantum States”**

Q-stage is a neural quantum state (NQS) that treats quantum systems and Hamiltonian parameters in a unified, transformer-like architecture. The goal is to learn a **single model** that can generalize across different lattice sizes, couplings, and phases by viewing “which Hamiltonian?” as part of the input, in the spirit of foundation models.

---

## Repository layout

The code is intentionally minimal and self-contained. The main components are:

- **`q_stage.py`**  
  Core implementation of the Q-stage architecture:
  - Builds the attention-based neural quantum state.
  - Encodes local degrees of freedom (e.g. spins/sites) and Hamiltonian information into model tokens.
  - Produces (log-)wavefunction amplitudes or related outputs used in the training loss.

- **`lattice.py`**  
  Utilities for specifying and encoding lattice models:
  - Defines lattice geometry and indexing.
  - Encodes Hamiltonian couplings (e.g. on-site fields, bond couplings) into numerical features the model consumes.
  - Provides helpers for constructing input tensors for Q-stage.

- **`loss.py`**  
  Loss functions and evaluation metrics:
  - Supervised losses against exact diagonalization (ED) targets (e.g. amplitude/phase or probability distributions).
  - Optional energy-based or overlap-based objectives.
  - Basic evaluation helpers (e.g. tracking training curves, overlaps, or fidelities).

- **`optimizer.py`**  
  Training utilities:
  - Optimizer and learning-rate schedule setup.
  - Single-step training/evaluation routines.
  - Lightweight logging hooks (e.g. printing or saving metrics).

- **`main_run.py`**  
  Top-level script to reproduce the project experiments:
  - Loads datasets from `ED_data/`.
  - Builds the Q-stage model and lattice encoding.
  - Runs training and evaluation loops, saving metrics/checkpoints as configured.

- **`ED_data/`**  
  Example **exact diagonalization datasets** used in this project:
  - Small-system benchmarks used to train and test Q-stage.
  - Filenames encode the underlying model / parameters (see the folder contents for details).
  - You can plug in your own ED data by following the same conventions.

---

## Installation

Q-stage is a small, pure-Python research codebase. A typical installation workflow is:

1. **Clone the repository**
   ```bash
   git clone https://github.com/tzaklama1/Q-stage.git
   cd Q-stage```

<!-- <details>
<summary>Q-stage setup, usage, and extension details</summary> -->

## 2. Create and activate a virtual environment

Pick your favorite tool (`venv` or `conda`):

Using `venv`:
```bash
python -m venv .qstage-env
source .qstage-env/bin/activate   # On Windows: .qstage-env\Scripts\activate
```

Or using `conda`:
```bash
conda create -n qstage python=3.11
conda activate qstage
```

---

## 3. Install dependencies

The project uses a standard scientific Python stack. Install the packages used in the imports of `q_stage.py`, `main_run.py`, and related modules. For example, a minimal setup might look like:
```bash
pip install numpy
# plus whichever deep-learning / optimization libraries are used in the code
# (e.g. jax, optax, pytorch, etc. – see the imports in the .py files)
```

> **Tip:** Once you have a working environment, you can export it into a `requirements.txt` and later reinstall via:
> ```bash
> pip install -r requirements.txt
> ```

---

## 4. (Optional) GPU support

If your deep-learning backend supports GPU (e.g. CUDA builds of JAX or PyTorch), install the corresponding GPU-enabled packages following their official instructions. This is not required for small ED benchmarks but can greatly speed up experiments.

---

## Quickstart

Once dependencies are installed and you are inside the repository root:

### 1. Run the main script

```bash
python main_run.py
```

- `main_run.py` is the entry point that wires together:
  - the lattice description in `lattice.py`,
  - the Q-stage model in `q_stage.py`,
  - the datasets in `ED_data/`,
  - the losses in `loss.py`, and
  - the optimizer utilities in `optimizer.py`.

- If the script expects command-line flags, you can inspect them with:
  ```bash
  python main_run.py --help
  ```

### 2. Inspect outputs

Depending on how you configured `main_run.py`, you can expect:

- Printed training curves (loss, overlap, energy error, etc.).
- Saved checkpoints of model parameters.
- Optional logs or arrays saved to disk (e.g. in a `results/` directory or one defined in the script).

### 3. Modifying an experiment

Typical ways to change an experiment:

- Edit hyperparameters directly in `main_run.py` (e.g. model depth, embedding dimension, learning rate, number of training steps).
- Point to a different dataset in `ED_data/`.
- Change the lattice / Hamiltonian encoding in `lattice.py`.

---

## Typical workflow

A common workflow with Q-stage is:

1. **Choose a model family and dataset**
   - For example, a family of spin or fermionic models at different couplings, sizes, or boundary conditions.
   - Prepare or select the corresponding ED data in `ED_data/`.

2. **Encode the lattice**
   - Use `lattice.py` to build a numerical representation (site indices, coupling matrices, boundary condition flags, etc.) consistent with the rest of the code.
   - Make sure the ED dataset and the lattice encoding agree on the Hilbert space and ordering.

3. **Configure the Q-stage architecture**
   - Adjust attention depth, number of heads, embedding dimensions, and any conditioning on Hamiltonian parameters inside `q_stage.py`.
   - Optionally introduce symmetry or equivariance constraints if appropriate for your model family.

4. **Pick a loss**
   - For supervised learning from ED, you might:
     - Minimize overlap or fidelity error between the predicted state and the ED ground state.
     - Minimize amplitude/phase errors in a suitable representation.
   - For energy-based learning, you could plug in a variational energy estimator (if present) using `loss.py`.

5. **Train & evaluate**
   - Launch training with `main_run.py`.
   - Track how well Q-stage:
     - interpolates within the training distribution of Hamiltonians, and
     - generalizes to new couplings / sizes not seen during training.

---

## Extending Q-stage

Some natural extensions of this codebase:

- **New lattice families**  
  Add new encodings or helper functions in `lattice.py` for:
  - different geometries (chains, ladders, 2D lattices),
  - new types of interactions / disorder.

- **Alternative losses and tasks**  
  Extend `loss.py` to:
  - support new metrics (e.g. entanglement entropy, correlation functions),
  - support multi-task setups (e.g. predicting energies *and* observables).

- **Model variations**  
  Modify `q_stage.py` to:
  - test different attention biases or positional encodings,
  - add FiLM-style conditioning on Hamiltonian parameters,
  - experiment with depth/width scaling towards more “foundation-like” behavior.

- **Data generation scripts**  
  If you generate new ED datasets externally, you can standardize their format and add a short description to `ED_data/README` (you can create this file) to document how to reuse them.

---

## Citing this work

If you use this code or ideas from Q-stage in academic work, please consider citing the associated project/paper and this repository.

### Paper (update once bibliographic details are final)

```bibtex
@article{Zaklama2025AttentionQStage,
  title   = {Attention-Based Foundation Model for Quantum States},
  author  = {Zaklama, Timothy, Guerci, Daniele, and Fu, Liang},
  year    = {2025},
  journal = {arXiv:2512.XXXXX}
}
```

### Software

```bibtex
@software{Zaklama2025QStage,
  author  = {Zaklama, Timothy},
  title   = {{Q-stage: Foundation Model Architecture for Quantum States}},
  url     = {https://github.com/tzaklama1/Q-stage},
  year    = {2025}
}
```

---

## License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file in the root of this repository for the full license text.



</details>

