# -----------------------------------------------------------------------------
# dist_patch.py
# -----------------------------------------------------------------------------

"""Patch ``torch.distributions.MultivariateNormal`` → diagonal ``Normal``.

Road‑tested on the Saxena‑lab SRNN implementation which assumes the *old*
behaviour where a diagonal covariance could be passed.  Call
:pyfunc:`apply_patch` **once** right after importing ``torch``.
"""

import torch
import torch.distributions.multivariate_normal as _mvn_mod
import torch.distributions as _dist

_orig_mvn = _mvn_mod.MultivariateNormal  # keep a reference – could restore

def _patched_mvn(loc, covariance_matrix=None, *a, **k):  # type: ignore[override]
    if covariance_matrix is not None and isinstance(covariance_matrix, torch.Tensor):
        var = covariance_matrix.diagonal(dim1=-2, dim2=-1)
        std = torch.sqrt(var)
    else:
        std = torch.full_like(loc, 1e-2)
    return _dist.Independent(_dist.Normal(loc, std), 1)


def apply_patch() -> None:
    """Globally monkey‑patch both ``torch.distributions`` modules."""
    _mvn_mod.MultivariateNormal = _patched_mvn  # type: ignore[attr-defined]
    _dist.MultivariateNormal = _patched_mvn  # type: ignore[attr-defined]
