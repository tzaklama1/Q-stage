# Copyright (c) 2025 Timothy Zaklama, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests model for interacting 1d chain of fermions and beyond.

This file is the main file from which Q-stage can be run.
"""
import jax, jax.numpy as jnp
from flax.training import train_state
import flax.serialization as serialization
import time
import numpy as np
from pathlib import Path
from .lattice import enumerate_fock, mask_to_array, my_readcsv, write_run_summary_gpu
from .loss import overlap_loss_multi, overlap_loss
from .optimizer import create_optimizer
from .q_stage import LatticeTransFormer


# ---------------------------------------------------------------------------
# 1)  Initialize and set parameters
# ---------------------------------------------------------------------------


# --- hyperparameters -------------------------------------------------------
DEPTH = 2
F_DIM = 80
HEADS = 4
MLP_DIMS = 80
# --- input/output ----------------------------------------------------------
dir = "/your/path/here/"   # adjust to your path
out_dir = dir + "run_output"
# --- lattice ---------------------------------------------------------------
LATTICE   = "1d"      # specify lattice type
N_SITES   = 12        # total number of sites

# --- Hamiltonian parameters ------------------------------------------------
V_LIST = [0.0, 1.0, 4.0]
T_LIST = [1.0 for _ in V_LIST]
N_LIST = [int(N_LIST/2) for _ in V_LIST]      

# --- optimisation ----------------------------------------------------------
LOSS_TYPE = "overlap_multi"     # "overlap_multi"  or  "overlap_loss" for single wavefunction optimization
TOTAL_STEPS    = 1200*7          # SGD steps in the demo script 
SEED           = 42        # RNG seed


# ---------------------------------------------------------------------------
# 2)  Build concatenated training set
# ---------------------------------------------------------------------------

assert len(T_LIST)==len(V_LIST)==len(N_LIST)
G = len(T_LIST)              # number of Hamiltonians

OCC_ALL, LAM_ALL, TARGET_ALL, GID_ALL = [], [], [], []
for gid,(t,v,npart) in enumerate(zip(T_LIST, V_LIST, N_LIST)):
    # basis for this particle number --------------------
    basis = enumerate_fock(N_SITES, npart)
    occ   = jnp.array([mask_to_array(m, N_SITES) for m in basis],
                        dtype=jnp.int32)

    # λ-vector extended to include N --------------------
    lam_vec = jnp.array([t, v, npart], dtype=jnp.float32)
    lam     = jnp.tile(lam_vec, (len(basis),1))

    coeffs, masks, _ = my_readcsv(f"./ED_data/gsWaveFn_Ns{int(N_SITES)}_Np{npart}_Vnn{round(v/t,2)}.csv")

    # TARGET aligned to OCC (Re, Im)
    targ = jnp.stack([jnp.array(coeffs.real), jnp.array(coeffs.imag)], axis=1)

    gid_vec = jnp.full((len(basis),), gid, dtype=jnp.int32)

    OCC_ALL.append(occ)
    LAM_ALL.append(lam)
    TARGET_ALL.append(targ)
    GID_ALL.append(gid_vec)

# concatenate everything --------------------------------
OCC     = jnp.concatenate(OCC_ALL,    axis=0)
LAM     = jnp.concatenate(LAM_ALL,    axis=0)   # shape (B,3)
TARGET  = jnp.concatenate(TARGET_ALL, axis=0)
GIDS    = jnp.concatenate(GID_ALL,    axis=0)   # shape (B,)

print("Total training states:", OCC.shape[0], "| Hamiltonians:", G, "V/t:", V_LIST)

# ---------------------------------------------------------------------------
# 3)  Define Model
# ---------------------------------------------------------------------------
model = LatticeTransFormer(n_sites=N_SITES, depth=DEPTH, d_model=F_DIM, n_heads=HEADS, mlp_dims=(MLP_DIMS, ))

# ---------------------------------------------------------------------------
# 4)  Loss & training step
# ---------------------------------------------------------------------------
OPTIMIZER = "adamw"  # choose optimizer scheme: "adam", "adamw", "sgd_nesterov"
LEARNING_RATE = 1e-3
OPT_KWARGS = {}   

def create_state(rng):
    params = model.init(rng, OCC, LAM, train=False)
    tx = create_optimizer(OPTIMIZER, learning_rate=LEARNING_RATE,**OPT_KWARGS)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

def loss_fn(params):
    preds = model.apply(params, OCC, LAM, train=False)
    if LOSS_TYPE == "overlap_multi":
        loss, overlap = overlap_loss_multi(preds, TARGET, GIDS, num_groups=G, return_overlap=True)
    else:
        raise ValueError("Unknown LOSS_TYPE")
    return loss, overlap

@jax.jit
def train_step(state):
    (loss, overlap), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, overlap

# ---------------------------------------------------------------------------
# 5)  Run training loop
# ---------------------------------------------------------------------------
print("JAX backend:", jax.default_backend().upper(), # e.g., CPU, GPU, TPU
    "| lattice:", LATTICE,
    f"| N={N_SITES}, Np={list(set(N_LIST))}, Attn Depth: {DEPTH}, features: {F_DIM}, heads: {HEADS}, MLP: {MLP_DIMS}")
print(f"Training for MLP V-pooling, dropout of 0.1 (and MLP), with {OPTIMIZER}, lr={LEARNING_RATE} LOSS={LOSS_TYPE}")

state = create_state(jax.random.PRNGKey(42))

# early stopping / convergence settings
PATIENCE = 5000               # number of epochs to wait for improvement
MIN_DELTA = 1e-3            # minimal loss improvement to reset patience
OV_THRESHOLD = 0.9999         # cumulative overlap threshold for immediate stop
PRINT_EVERY = 100          # print progress every N epochs

loss_hist = []
cumov_hist = []
pergroup_hist = []

best_loss = np.inf
best_cumov = 0.0
best_params_bytes = serialization.to_bytes(state.params)
epochs_since_improve = 0

start_time = time.time()
epoch = 0
while True:
    state, loss, overlap = train_step(state)

    loss_val = float(loss)
    loss_hist.append(loss_val)

    if overlap is None:
        cumov_hist.append(np.nan)
        pergroup_hist.append([np.nan] * G)
        cum_ov = np.nan
    else:
        abs_ov = np.asarray(np.abs(overlap))
        cum_ov = float(abs_ov.mean())
        cumov_hist.append(cum_ov)
        pergroup_hist.append(abs_ov.tolist())

    # track best and decide whether to save params
    improved = False
    if not np.isnan(cum_ov) and (cum_ov > best_cumov + 1e-8):
        best_cumov = cum_ov
        improved = True
    if loss_val + MIN_DELTA < best_loss:
        best_loss = loss_val
        improved = True

    if improved:
        best_params_bytes = serialization.to_bytes(state.params)
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1

    if epoch % PRINT_EVERY == 0:
        if overlap is None:
            ov_str = " ".join("nan" for _ in range(G))
            print(f"epoch {epoch:4d}  loss = {loss_val:.4e}  cum_overlap = nan  overlaps = {ov_str}")
        else:
            ov_str = " ".join(f"{o:.4e}" for o in np.asarray(np.abs(overlap)))
            print(f"epoch {epoch:4d}  loss = {loss_val:.4e}  cum_overlap = {cum_ov:.4e}  overlaps = {ov_str}")

    # convergence checks (break based on convergence or other criteria)
    if not np.isnan(cum_ov) and cum_ov >= OV_THRESHOLD:
        print(f"Converged at epoch {epoch}: cumulative overlap = {cum_ov:.4e} >= {OV_THRESHOLD}")
        break
    if epochs_since_improve >= PATIENCE:
        print(f"Stopping at epoch {epoch}: no sufficient loss/overlap improvement for {PATIENCE} epochs.")
        break
    # safety cap to avoid infinite loop
    if epoch + 1 >= 2 * TOTAL_STEPS:
        print(f"Reached maximum allowed steps ({TOTAL_STEPS}), stopping.")
        break

    epoch += 1

total_time = time.time() - start_time
print(f"Training finished after {epoch+1} epochs, elapsed {total_time:.1f}s. Best loss {best_loss:.4e}, best cum_ov {best_cumov:.4e}")






# ---------------------------------------------------------------------------
# 6)  Post-training metrics & outputs
# ---------------------------------------------------------------------------
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
# Predicted coefficients (concatenated over all groups)
preds_dev  = model.apply(state.params, OCC, LAM, train=False) 
loss_hist  = jnp.asarray(jnp.array(loss_hist))
group_overlap = jnp.asarray(jnp.array(pergroup_hist))   

params_file = f"{out_dir}Netweights_Dim{LATTICE}_L{N_SITES}_N{'_'.join(f'{n}' for n in N_LIST)}_V{'_'.join(f'{v:.1f}' for v in V_LIST)}_D{DEPTH}_F{F_DIM}_H{HEADS}_MLP{MLP_DIMS}_thresh{OV_THRESHOLD}_bestParams.msgpack"
with open(params_file, "wb") as f:
    f.write(best_params_bytes)      
print(f"Saved best parameters to {params_file}")

run_name = f"Fullrun_Dim{LATTICE}_L{N_SITES}_N{'_'.join(f'{n}' for n in N_LIST)}_V{'_'.join(f'{v:.1f}' for v in V_LIST)}_D{DEPTH}_F{F_DIM}_H{HEADS}_MLP{MLP_DIMS}_thresh{OV_THRESHOLD}seed{SEED}"
config_row = dict(
    depth=DEPTH, width=F_DIM,
    lattice=LATTICE, N_SITES=N_SITES,
    loss_type=LOSS_TYPE, epochs=TOTAL_STEPS, print_every=PRINT_EVERY,
    n_hamiltonians=G, seed=SEED
)

# 1) Summary (single device_get per tensor)
summary_path = write_run_summary_gpu(
    out_dir=out_dir,
    run_name=run_name,
    config_dict=config_row,
    loss_history_device=loss_hist,
    overlap_history_device=group_overlap,
    preds_device=preds_dev,
    target_device=TARGET,
    gids_device=GIDS,         
    G=G
)
print(f"Saved run summary to {summary_path}")


# ---------------------------------------------------------------------------
# 7)  Test model on held out data
# ---------------------------------------------------------------------------
v_over_t = 2.0
t = 1.0
ed_coeffs, _, _ = my_readcsv(f"ED_data/gsWaveFn_Sq_Ns16_Np8_Vnn{v_over_t}.csv")
targ = jnp.stack([jnp.array(ed_coeffs.real), jnp.array(ed_coeffs.imag)], axis=1)

# build basis + lam for this (L,N)
basis = enumerate_fock(N_SITES, N_LIST[0])
occ = jnp.array([mask_to_array(m, N_SITES) for m in basis], dtype=jnp.int32)
lam_vec = jnp.array([t, float(v_over_t), float(N_LIST[0])], dtype=jnp.float32)
lam = jnp.tile(lam_vec, (len(basis), 1))

# model prediction and overlap
coeff_pred = model.apply(state.params, occ, lam, train=False)  # (B,2)
_, overlap = overlap_loss(coeff_pred, targ)
ov_abs = float(jnp.abs(overlap))
print(f"Final test overlap for V/t={v_over_t} given training set of V={V_LIST}: |<Ψθ|ΨED>| = {ov_abs:.6f}")



