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
Loss functions for co-trained lattice TNN
----------------------------------------
From the overleaf document:
• overlap_loss  Eq.(2) exact overlap 1 – |⟨Ψθ|Ψ_ED⟩|²
• amp_phase_loss Eq.(3) amplitude + discrete phase-gradient distance
"""
import jax, jax.numpy as jnp
from typing import Tuple
from jax import lax

def _to_complex(x: jnp.ndarray) -> jnp.ndarray:
    return x[..., 0] + 1j * x[..., 1]

# --------------------------------------------------------------------------- #
#   Multi-group overlap loss: new normalization scheme          #
# --------------------------------------------------------------------------- #
def overlap_loss_multi(pred:        jnp.ndarray,   # (B,2)
                       target:      jnp.ndarray,   # (B,2)
                       gIDs:        jnp.ndarray,   # (B,)  int
                       num_groups:  int,           # PYTHON INT
                       return_overlap: bool = False
                       ) -> jnp.ndarray:
    """
    L = 1 - mean_{g=0..G-1} | ⟨Ψθ_g | Ψ_ED_g⟩ |²
    Uses jnp.bincount to avoid dynamic shapes.
    Optionally returns the per-group overlaps.
    """
    ψθ = _to_complex(pred)
    ψ  = _to_complex(target)
    eps = 1e-12

    # compute norms for each group -----------------------------------------
    nθ  = jnp.bincount(gIDs, weights=jnp.abs(ψθ) ** 2,
                       length=num_groups)
    nψ  = jnp.bincount(gIDs, weights=jnp.abs(ψ)  ** 2,
                       length=num_groups)

    # normalise wave-functions per group before taking the overlap
    ψθ = ψθ / jnp.sqrt(nθ[gIDs] + eps)
    ψ  = ψ  / jnp.sqrt(nψ[gIDs] + eps)

    # weighted sums per group using normalised coefficients -----------------
    overlap = jnp.bincount(gIDs, weights=jnp.conj(ψθ) * ψ,
                           length=num_groups)

    loss_g  = 1.0 - jnp.abs(overlap) ** 2
    loss    = jnp.mean(loss_g)
    return (loss, overlap) if return_overlap else loss

# --------------------------------------------------------------------------- #
# Single exact overlap loss                                                       #
# --------------------------------------------------------------------------- #
def overlap_loss(pred: jnp.ndarray,
                 target: jnp.ndarray,
                 return_overlap = True) -> jnp.ndarray:
    """
    pred,target : (H,2)  real-imag coefficients for *one* λ
    returns scalar loss ≥ 0      minimise → max overlap
    """
    ψθ = _to_complex(pred)
    ψ  = _to_complex(target)
    # normalize
    ψθ /= jnp.linalg.norm(ψθ) + 1e-12
    ψ  /= jnp.linalg.norm(ψ)  + 1e-12
    overlap = jnp.vdot(ψθ, ψ)               # ⟨Ψθ | Ψ_ED⟩
    loss = 1.0 - jnp.abs(overlap)**2        # ≤ 1  … best = 0
    return (loss, overlap) if return_overlap else loss
